#!/usr/bin/env python3
"""
Hebrew Subtitle Translator
Extracts audio from video, transcribes with Whisper (or embedded subs),
diarizes speakers with pyannote, translates to gender-aware Hebrew with Claude.
"""

import argparse
import atexit
import json
import os
import re
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
    pass

import anthropic
import openai

WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
BATCH_SIZE = 120
BATCH_OVERLAP = 15
CHUNK_MINUTES = 10
CHUNK_OVERLAP_SECONDS = 15
GENDER_ANALYSIS_MAX_SEGMENTS = 3000

CACHE_VERSION = 1  # bump to invalidate all caches


# ---------------------------------------------------------------------------
# Cache helpers  (saved as <video>.cache.json next to the video file)
# ---------------------------------------------------------------------------

def _cache_path(video_path):
    return str(Path(video_path).with_suffix(".cache.json"))


def _load_cache(video_path):
    path = _cache_path(video_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("cache_version") != CACHE_VERSION:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(video_path, cache):
    cache["cache_version"] = CACHE_VERSION
    path = _cache_path(video_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  [cache saved → {Path(path).name}]")


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
    """Extract 16kHz mono 32kbps MP3. Returns path to temp audio file."""
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
# Step 2a — Embedded subtitle extraction (preferred for timing accuracy)
# ---------------------------------------------------------------------------

def extract_embedded_subtitles(video_path):
    """
    Try to extract an English subtitle stream from the video.
    Returns list of segment dicts or None if no subtitle stream found.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index:stream_tags=language",
         "-of", "json", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None

    try:
        streams = json.loads(probe.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None

    if not streams:
        return None

    # Prefer English, fall back to first available
    target_index = None
    for stream in streams:
        lang = stream.get("tags", {}).get("language", "")
        if lang in ("eng", "en"):
            target_index = stream["index"]
            break
    if target_index is None:
        target_index = streams[0]["index"]

    srt_path = make_temp(".srt")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{target_index}", srt_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.getsize(srt_path):
        return None

    segments = _parse_srt_file(srt_path)
    return segments if segments else None


def _parse_srt_timestamp(ts):
    """Convert HH:MM:SS,mmm to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.replace(".", ",").split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt_file(srt_path):
    """Parse SRT file into segment dicts {id, start, end, text}."""
    with open(srt_path, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    segments = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Find the timestamp line (index might vary)
        ts_line_idx = None
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue

        ts_match = re.match(
            r"(\d+:\d+:\d+[,\.]\d+)\s+-->\s+(\d+:\d+:\d+[,\.]\d+)",
            lines[ts_line_idx],
        )
        if not ts_match:
            continue

        start = _parse_srt_timestamp(ts_match.group(1))
        end = _parse_srt_timestamp(ts_match.group(2))
        text_lines = lines[ts_line_idx + 1:]
        text = " ".join(text_lines).strip()
        # Strip HTML tags and SDH annotations
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\{[^}]+\}", "", text)
        text = re.sub(r"^\s*[\(\[][^\)\]]*[\)\]]\s*$", "", text)
        text = text.strip()
        if not text:
            continue

        segments.append({"id": len(segments), "start": start, "end": end, "text": text})

    return segments


# ---------------------------------------------------------------------------
# Step 2b — Whisper transcription (fallback)
# ---------------------------------------------------------------------------

def transcribe(audio_path):
    """Transcribe audio with Whisper. Returns list of segment dicts."""
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
# Step 3 — Speaker diarization
# ---------------------------------------------------------------------------

def diarize(audio_path, hf_token):
    """
    Run pyannote speaker diarization. Returns list of {start, end, speaker} dicts,
    or None if unavailable.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("  pyannote.audio not installed. Run: pip3 install pyannote.audio")
        return None

    print("  Loading diarization model (first run downloads ~1 GB)...")
    try:
        # PyTorch 2.6+ defaults weights_only=True which breaks pyannote checkpoints.
        # Patch torch.load to trust pyannote's HuggingFace models.
        import torch
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    except Exception:
        pass
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        print(f"  Failed to load diarization model: {e}")
        print("  Make sure HF_TOKEN is set and you accepted the model license at:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        return None

    try:
        torch.load = _orig_load  # restore after loading
        if torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))
    except Exception:
        pass  # CPU fallback is fine

    print("  Running diarization...")
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"  Diarization failed: {e}")
        return None

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return turns


def assign_speakers(segments, turns):
    """
    Assign a speaker label to each segment based on maximum time overlap with diarization turns.
    Adds a 'speaker' key to each segment dict.
    """
    for seg in segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return segments


# ---------------------------------------------------------------------------
# Step 4a — Gender analysis pass
# ---------------------------------------------------------------------------

GENDER_ANALYSIS_SYSTEM = """\
You are a script analyst. Read the transcript and identify every speaker.

If speaker IDs like SPEAKER_00 are present, map each ID to a character name and gender.
If no speaker IDs are present, infer speakers from context (names, dialogue patterns).

Return ONLY a JSON object (no markdown):
{
  "speakers": [
    {
      "id": "SPEAKER_00",
      "name": "Character name or role",
      "gender": "male" | "female" | "unknown",
      "evidence": "brief reason"
    }
  ],
  "address_notes": "any general notes about how characters address each other"
}

Include every identifiable speaker. Be thorough with evidence.
"""


def analyze_genders(segments, hints="", cache=None, cache_save=None):
    """
    Pass 1: send full transcript to Claude to build a speaker→gender map.
    Returns (gender_context_string, speaker_map dict {speaker_id: {name, gender}}).
    """
    if cache and cache.get("gender_analysis"):
        stored = cache["gender_analysis"]
        print(f"Using cached gender analysis: {len(stored.get('speakers', []))} speakers")
        return _build_gender_outputs(stored)

    print("Analyzing speaker genders across full transcript...")

    lines = []
    for seg in segments[:GENDER_ANALYSIS_MAX_SEGMENTS]:
        speaker_prefix = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines.append(f"[{seg['id']}] {speaker_prefix}{seg['text']}")
    transcript_text = "\n".join(lines)

    hint_section = f"\n\nUser hints: {hints}" if hints else ""
    user_msg = (
        f"Analyze this transcript and identify all speakers and their genders.{hint_section}\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
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
            if not data.get("speakers"):
                return "", {}

            if cache is not None:
                cache["gender_analysis"] = data
                if cache_save:
                    cache_save()

            context_str, speaker_map = _build_gender_outputs(data)
            return context_str, speaker_map

        except Exception as e:
            if attempt == 0:
                print(f"  Gender analysis error: {e} — retrying...")
                time.sleep(5)
            else:
                print(f"  Warning: gender analysis failed ({e}), proceeding without it.")
                return "", {}

    return "", {}


def _build_gender_outputs(data):
    """Build context string and speaker_map from gender analysis data dict."""
    speakers = data.get("speakers", [])
    address_notes = data.get("address_notes", "")

    speaker_map = {}
    for sp in speakers:
        speaker_map[sp.get("id", sp.get("name", ""))] = {
            "name": sp.get("name", ""),
            "gender": sp.get("gender", "unknown"),
        }

    out = ["=== CAST & GENDER REFERENCE (apply consistently throughout) ==="]
    for sp in speakers:
        gender_label = sp.get("gender", "unknown").upper()
        name = sp.get("name", "?")
        sid = sp.get("id", "")
        evidence = sp.get("evidence", "")
        id_part = f" ({sid})" if sid and sid != name else ""
        out.append(f"  • {name}{id_part}: {gender_label}  — {evidence}")
    if address_notes:
        out.append(f"\n  Address notes: {address_notes}")
    out.append("=== END CAST REFERENCE ===")

    context_str = "\n".join(out)
    print(f"  {len(speakers)} speakers in cast reference:")
    for sp in speakers:
        print(f"    {sp.get('name')}: {sp.get('gender')} — {sp.get('evidence', '')[:80]}")

    return context_str, speaker_map


# ---------------------------------------------------------------------------
# Step 4b — Claude translation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Hebrew translator specializing in gender-aware translation.

{gender_context}

Your task:
1. Use the cast & gender reference above as ground truth for every gender decision.
2. Each segment may include a SPEAKER label (e.g. "SPEAKER_00 = Donna, FEMALE").
   Use the speaker label to determine:
   A. SPEAKER GENDER — how the speaker refers to themselves:
      - Self-description adjectives: אני עייף (m) / אני עייפה (f)
      - First-person verb agreement: הלכתי is same for m/f, but אני יודע (m) / אני יודעת (f)
   B. ADDRESSEE GENDER — how the speaker addresses the listener, inferred from context:
      - Second-person pronouns: אתה (m) / את (f)
      - Imperatives: לך! (m) / לכי! (f), תגיד / תגידי, בוא / בואי, שב / שבי, תראה / תראי
      - Adjectives in address: אתה חזק (m) / את חזקה (f)
3. Apply correct third-person gender: הוא/היא, שלו/שלה, אמר/אמרה, הלך/הלכה.
4. Default to masculine ONLY when gender is truly unresolvable.
5. Produce natural, fluent Hebrew — not word-for-word.
6. Preserve segment IDs and timestamps exactly.
7. Do NOT include the speaker label in the output text.

Return ONLY a valid JSON array (no markdown, no explanation):
[{{"id": <int>, "start": <float>, "end": <float>, "text": "<Hebrew translation>"}}]
"""

CONTEXT_ONLY_NOTE = " [CONTEXT ONLY — do not include in output]"


def translate(segments, hints="", cache=None, cache_save=None):
    """Translate all segments to Hebrew using Claude. Returns translated segments."""
    if not segments:
        return []

    gender_context, speaker_map = analyze_genders(segments, hints=hints,
                                                   cache=cache, cache_save=cache_save)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        gender_context=gender_context if gender_context
        else "(No cast reference available — infer genders from context.)"
    )

    if len(segments) <= BATCH_SIZE:
        return _translate_batch(segments, hints=hints, system_prompt=system_prompt,
                                speaker_map=speaker_map)

    print(f"Translating {len(segments)} segments in batches of {BATCH_SIZE} (overlap {BATCH_OVERLAP})...")
    return _translate_batched(segments, hints=hints, system_prompt=system_prompt,
                              speaker_map=speaker_map)


def _translate_batched(segments, hints, system_prompt, speaker_map):
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
            speaker_map=speaker_map, context_tail=context_tail,
        )

        for seg in translated:
            if seg["id"] not in results:
                results[seg["id"]] = seg

        print(f" ✓ ({len(translated)} translated)")
        i += BATCH_SIZE - BATCH_OVERLAP

    output = []
    for seg in segments:
        output.append(results.get(seg["id"], seg))
    return output


def _translate_batch(batch, hints, system_prompt, speaker_map, context_tail=None):
    context_tail = context_tail or []
    hint_section = f"\n\nAdditional user hints: {hints}" if hints else ""

    def _enrich(seg, context_only=False):
        """Add speaker label to segment text for Claude's benefit."""
        d = dict(seg)
        speaker_id = seg.get("speaker")
        if speaker_id and speaker_id != "UNKNOWN" and speaker_map:
            info = speaker_map.get(speaker_id, {})
            name = info.get("name", speaker_id)
            gender = info.get("gender", "unknown").upper()
            d["speaker_label"] = f"{name} ({gender})"
        if context_only:
            d["text"] = d["text"] + CONTEXT_ONLY_NOTE
        # Remove raw speaker field from payload (Claude only needs speaker_label)
        d.pop("speaker", None)
        return d

    payload = [_enrich(s) for s in batch]
    payload += [_enrich(s, context_only=True) for s in context_tail]

    user_msg = (
        f"Translate the following transcript segments to Hebrew.{hint_section}\n\n"
        f"Each segment may have a 'speaker_label' indicating who is speaking and their gender. "
        f"Context-only segments at the end must NOT appear in output.\n\n"
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
                        f"{json.dumps([_enrich(s) for s in batch], ensure_ascii=False, indent=2)}"
                    )
                else:
                    time.sleep(10)
            else:
                sys.exit(f"Translation failed after retry: {e}")

    return batch


# ---------------------------------------------------------------------------
# Step 5 — SRT generation
# ---------------------------------------------------------------------------

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path):
    """Write segments to SRT with UTF-8 BOM for Hebrew RTL compatibility."""
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
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output .srt path (default: <input>.srt)")
    parser.add_argument("--hints", default="",
                        help='Gender hints (e.g. "female host, male guest")')
    parser.add_argument("--keep-audio", action="store_true",
                        help="Keep extracted audio file")
    parser.add_argument("--hf-token", default="",
                        help="HuggingFace token for speaker diarization (or set HF_TOKEN env var)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip speaker diarization")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        sys.exit(f"Input file not found: {input_path}")

    output_path = args.output or str(Path(input_path).with_suffix(".srt"))

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    cache = _load_cache(input_path)
    if cache:
        print(f"Loaded cache: {_cache_path(input_path)}")

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

    # Step 2: Get transcript (embedded subs preferred, Whisper fallback)
    if cache.get("segments"):
        print(f"Using cached transcript: {len(cache['segments'])} segments")
        segments = cache["segments"]
    else:
        print("Checking for embedded subtitles...")
        segments = extract_embedded_subtitles(input_path)
        if segments:
            print(f"Using embedded subtitles: {len(segments)} segments (frame-accurate timing)")
        else:
            print("No embedded subtitles found — transcribing with Whisper...")
            segments = transcribe(audio_path)
            print(f"Transcribed {len(segments)} segments")
        cache["segments"] = segments
        _save_cache(input_path, cache)

    if not segments:
        sys.exit("No speech segments found.")

    # Step 3: Speaker diarization
    if not args.no_diarize:
        if cache.get("diarization_turns"):
            print(f"Using cached diarization: {len(cache['diarization_turns'])} speaker turns")
            turns = cache["diarization_turns"]
            segments = assign_speakers(segments, turns)
            unique_speakers = len({s["speaker"] for s in segments})
            print(f"  Assigned {unique_speakers} speakers across {len(segments)} segments")
        elif hf_token:
            print("Running speaker diarization...")
            turns = diarize(audio_path, hf_token)
            if turns:
                cache["diarization_turns"] = turns
                _save_cache(input_path, cache)
                segments = assign_speakers(segments, turns)
                unique_speakers = len({s["speaker"] for s in segments})
                print(f"  Assigned {unique_speakers} speakers across {len(segments)} segments")
            else:
                print("  Diarization unavailable — proceeding without speaker labels")
        else:
            print("Skipping diarization (no HF_TOKEN set — set it for better gender accuracy)")
            print("  Get a free token at: https://huggingface.co/settings/tokens")
            print("  Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1")

    # Step 4: Translate (gender analysis is cached inside translate())
    print("Translating to Hebrew with Claude...")
    translated = translate(segments, hints=args.hints, cache=cache,
                           cache_save=lambda: _save_cache(input_path, cache))

    # Step 5: Write SRT
    write_srt(translated, output_path)


if __name__ == "__main__":
    main()
