#!/usr/bin/env python3
"""
Hebrew Subtitle Translator
Extracts audio from video, transcribes with Whisper, translates to gender-aware Hebrew with Claude.
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional if vars are already set

import anthropic
import openai

WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
BATCH_SIZE = 120
BATCH_OVERLAP = 15
CHUNK_MINUTES = 10
CHUNK_OVERLAP_SECONDS = 15

# Max segments to send for gender analysis in one call (~200K token context is fine for ~2000 segs)
GENDER_ANALYSIS_MAX_SEGMENTS = 3000


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------

_temp_files = []

def _cleanup_temp_files():
    for path in _temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass

atexit.register(_cleanup_temp_files)


def make_temp(suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    _temp_files.append(path)
    return path


# ---------------------------------------------------------------------------
# Step 1 — Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path):
    """Extract 16kHz mono 32kbps MP3 from video. Returns path to temp audio file."""
    _check_ffmpeg()
    audio_path = make_temp(".mp3")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg error:\n{result.stderr}")
    return audio_path


def _check_ffmpeg():
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        sys.exit("ffmpeg not found. Install with: brew install ffmpeg")


def split_audio(audio_path):
    """Split audio into overlapping chunks. Returns list of (path, start_offset_seconds)."""
    chunk_sec = CHUNK_MINUTES * 60
    overlap = CHUNK_OVERLAP_SECONDS

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())

    chunks = []
    start = 0.0
    while start < duration:
        chunk_path = make_temp(".mp3")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_sec + overlap),
            "-c", "copy", chunk_path,
        ]
        subprocess.run(cmd, capture_output=True)
        chunks.append((chunk_path, start))
        start += chunk_sec
        if start >= duration:
            break

    return chunks


# ---------------------------------------------------------------------------
# Step 2 — Whisper transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path):
    """Transcribe audio with Whisper. Returns list of segment dicts {id, start, end, text}."""
    size = os.path.getsize(audio_path)
    if size > WHISPER_MAX_BYTES:
        print(f"Audio is {size / 1024 / 1024:.1f} MB > 25 MB limit — splitting into chunks...")
        return _transcribe_chunked(audio_path)
    return _transcribe_file(audio_path, offset=0.0)


def _transcribe_file(audio_path, offset):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(2):
        try:
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            segments = []
            for i, seg in enumerate(result.segments):
                segments.append({
                    "id": i,
                    "start": round(seg.start + offset, 3),
                    "end": round(seg.end + offset, 3),
                    "text": seg.text.strip(),
                })
            return segments
        except Exception as e:
            if attempt == 0:
                print(f"Whisper error: {e} — retrying in 10s...")
                time.sleep(10)
            else:
                sys.exit(f"Whisper transcription failed: {e}")
    return []


def _transcribe_chunked(audio_path):
    chunks = split_audio(audio_path)
    all_segments = []
    seen_starts = set()

    for chunk_path, offset in chunks:
        segs = _transcribe_file(chunk_path, offset=offset)
        for seg in segs:
            if seg["start"] not in seen_starts:
                seen_starts.add(seg["start"])
                all_segments.append(seg)

    for i, seg in enumerate(sorted(all_segments, key=lambda s: s["start"])):
        seg["id"] = i

    return all_segments


# ---------------------------------------------------------------------------
# Step 3a — Gender analysis pass (new)
# ---------------------------------------------------------------------------

GENDER_ANALYSIS_SYSTEM = """\
You are a script analyst. Your job is to read a transcript and identify every \
speaker/character, determine their gender, and note how each character is \
addressed by others.

Return ONLY a JSON object (no markdown, no explanation) in this exact format:
{
  "speakers": [
    {
      "name": "character name or role (e.g. 'Host', 'Sarah', 'Contestant 1')",
      "gender": "male" | "female" | "unknown",
      "evidence": "brief reason (e.g. 'referred to as she/her', 'name is female')"
    }
  ],
  "address_notes": "any general notes about how characters address each other gender-wise"
}

Be thorough — include every named character and every identifiable unnamed role.
"""

def analyze_genders(segments, hints=""):
    """
    Pass 1: Send full transcript text to Claude to identify speaker genders.
    Returns a formatted string to inject into every translation batch.
    """
    print("Analyzing speaker genders across full transcript...")

    # Build a compact plain-text transcript (no timestamps, just numbered lines)
    lines = []
    for seg in segments[:GENDER_ANALYSIS_MAX_SEGMENTS]:
        lines.append(f"[{seg['id']}] {seg['text']}")
    transcript_text = "\n".join(lines)

    hint_section = f"\n\nUser hints: {hints}" if hints else ""
    user_msg = (
        f"Analyze the following transcript and identify all speakers and their genders.{hint_section}\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=GENDER_ANALYSIS_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            speakers = data.get("speakers", [])
            address_notes = data.get("address_notes", "")

            # Format into a clear reference block for translation prompts
            if not speakers:
                return ""

            lines_out = ["=== CAST & GENDER REFERENCE (apply consistently throughout) ==="]
            for sp in speakers:
                gender_label = sp.get("gender", "unknown").upper()
                name = sp.get("name", "Unknown")
                evidence = sp.get("evidence", "")
                lines_out.append(f"  • {name}: {gender_label}  ({evidence})")
            if address_notes:
                lines_out.append(f"\n  Address notes: {address_notes}")
            lines_out.append("=== END CAST REFERENCE ===")

            result = "\n".join(lines_out)
            print(f"  Found {len(speakers)} speakers:")
            for sp in speakers:
                print(f"    {sp.get('name')}: {sp.get('gender')} — {sp.get('evidence')}")
            return result

        except Exception as e:
            if attempt == 0:
                print(f"  Gender analysis error: {e} — retrying...")
                time.sleep(5)
            else:
                print(f"  Warning: gender analysis failed ({e}), proceeding without it.")
                return ""

    return ""


# ---------------------------------------------------------------------------
# Step 3b — Claude translation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Hebrew translator specializing in gender-aware translation.

{gender_context}

Your task:
1. Use the cast & gender reference above as ground truth for every gender decision.
2. Apply correct Hebrew grammatical gender for BOTH:
   A. SPEAKER GENDER — how the speaker refers to themselves:
      - Past tense: הלכתי (I went, m/f same) but הוא הלך (he) / היא הלכה (she)
      - Self-description adjectives: אני עייף (m) / אני עייפה (f)
   B. ADDRESSEE GENDER — how the speaker addresses the listener:
      - Second-person pronouns: אתה (m) / את (f)
      - Imperatives: לך! (m) / לכי! (f), תגיד / תגידי, בוא / בואי, שב / שבי
      - Adjectives: אתה חזק (m) / את חזקה (f)
      - Possessives: שלך (m) / שלך (f) — same form but pay attention to verb agreement
3. Apply correct third-person gender:
   - הוא/היא, שלו/שלה, אמר/אמרה, הלך/הלכה, etc.
4. Default to masculine ONLY when gender is truly unknown and unresolvable.
5. Produce natural, fluent Hebrew — not word-for-word translations.
6. Preserve segment IDs and timestamps exactly.

Return ONLY a valid JSON array (no markdown, no explanation):
[{{"id": <int>, "start": <float>, "end": <float>, "text": "<Hebrew translation>"}}]
"""

CONTEXT_ONLY_NOTE = " [CONTEXT ONLY — do not include in output]"


def translate(segments, hints=""):
    """Translate all segments to Hebrew using Claude. Returns translated segments."""
    if not segments:
        return []

    # Pass 1: analyze genders across the full transcript
    gender_context = analyze_genders(segments, hints=hints)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        gender_context=gender_context if gender_context else
        "(No cast reference available — infer genders from context as best you can.)"
    )

    if len(segments) <= BATCH_SIZE:
        return _translate_batch(segments, hints=hints, system_prompt=system_prompt)

    print(f"Translating {len(segments)} segments in batches of {BATCH_SIZE} (overlap {BATCH_OVERLAP})...")
    return _translate_batched(segments, hints=hints, system_prompt=system_prompt)


def _translate_batched(segments, hints, system_prompt):
    results = {}
    total = len(segments)
    i = 0
    batch_num = 0

    while i < total:
        batch_num += 1
        end = min(i + BATCH_SIZE, total)
        batch = segments[i:end]
        is_last = end >= total

        context_tail = segments[end:end + BATCH_OVERLAP] if not is_last else []

        print(f"  Batch {batch_num}: segments {i}–{end - 1}", end="", flush=True)

        translated = _translate_batch(
            batch, hints=hints, system_prompt=system_prompt,
            context_tail=context_tail,
        )

        for seg in translated:
            if seg["id"] not in results:
                results[seg["id"]] = seg

        print(f" ✓ ({len(translated)} translated)")
        i += BATCH_SIZE - BATCH_OVERLAP

    output = []
    for seg in segments:
        if seg["id"] in results:
            output.append(results[seg["id"]])
        else:
            output.append(seg)
    return output


def _translate_batch(batch, hints, system_prompt, context_tail=None):
    context_tail = context_tail or []

    hint_section = f"\n\nAdditional user hints: {hints}" if hints else ""

    payload = list(batch)
    for seg in context_tail:
        payload.append({**seg, "text": seg["text"] + CONTEXT_ONLY_NOTE})

    user_msg = (
        f"Translate the following transcript segments to Hebrew.{hint_section}\n\n"
        f"Segments to translate (context-only segments at the end must NOT appear in output):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    expected_ids = {seg["id"] for seg in batch}

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            translated = json.loads(raw)

            returned_ids = {seg["id"] for seg in translated}
            if not expected_ids.issubset(returned_ids):
                missing = expected_ids - returned_ids
                raise ValueError(f"Missing segment IDs: {missing}")

            return [seg for seg in translated if seg["id"] in expected_ids]

        except Exception as e:
            if attempt == 0:
                print(f"\n  Translation error: {e} — retrying...")
                if isinstance(e, (json.JSONDecodeError, ValueError)):
                    user_msg = (
                        f"Your previous response had an issue: {e}\n\n"
                        f"Please try again. Return ONLY a valid JSON array for these segments:\n"
                        f"{json.dumps(batch, ensure_ascii=False, indent=2)}"
                    )
                else:
                    time.sleep(10)
            else:
                sys.exit(f"Translation failed after retry: {e}")

    return batch


# ---------------------------------------------------------------------------
# Step 4 — SRT generation
# ---------------------------------------------------------------------------

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path):
    """Write segments to an SRT file with UTF-8 BOM for RTL Hebrew compatibility."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(segments)} subtitles to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate video audio to Hebrew subtitles (.srt)",
    )
    parser.add_argument("input", help="Input video file (e.g. input.mp4)")
    parser.add_argument(
        "-o", "--output",
        help="Output .srt file path (default: <input>.srt)",
    )
    parser.add_argument(
        "--hints",
        default="",
        help='Gender hints for Claude (e.g. "female interviewer, male interviewee")',
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file after completion",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        sys.exit(f"Input file not found: {input_path}")

    output_path = args.output or str(Path(input_path).with_suffix(".srt"))

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Add it to your environment or .env file.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set. Add it to your environment or .env file.")

    # Step 1: Extract audio
    print(f"Extracting audio from: {input_path}")
    audio_path = extract_audio(input_path)
    size_mb = os.path.getsize(audio_path) / 1024 / 1024
    print(f"Audio extracted: {size_mb:.1f} MB")

    if args.keep_audio:
        _temp_files.remove(audio_path)
        kept_path = str(Path(input_path).with_suffix(".mp3"))
        os.rename(audio_path, kept_path)
        audio_path = kept_path
        print(f"Audio saved to: {audio_path}")

    # Step 2: Transcribe
    print("Transcribing with Whisper...")
    segments = transcribe(audio_path)
    print(f"Transcribed {len(segments)} segments")
    if not segments:
        sys.exit("No speech segments found in audio.")

    # Step 3: Translate (with gender analysis pre-pass)
    print("Translating to Hebrew with Claude...")
    translated = translate(segments, hints=args.hints)

    # Step 4: Write SRT
    write_srt(translated, output_path)


if __name__ == "__main__":
    main()
