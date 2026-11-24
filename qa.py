"""
QA stage for the Hebrew subtitle translator.

Runs after translation and auto-fixes the subtitles:
  1. Structural  — every line keeps the source timestamps, sorted, one per source line.
  2. Timing      — the English source lines are aligned against a word-level Whisper
                   transcript of the real audio. Constant offsets, drift (e.g. a subtitle
                   track from a release with a different frame rate) and individually
                   misplaced lines are corrected.
  3. Language    — Claude reviews every Hebrew line against its English source, the
                   speaker and the cast reference, and fixes wrong gender forms, lines whose
                   text slid into a neighbouring line, untranslated lines and real
                   mistranslations.
  4. Checks      — report-only flags (reading speed, very short lines, empty lines).

Everything that changed is written to a JSON report next to the .srt.
"""

import bisect
import difflib
import json
import os
import re
import statistics
import time

import anthropic

QA_MODEL = "claude-opus-5-5"
QA_REVIEW_BATCH = 120
QA_REVIEW_CONTEXT = 8

# Alignment tuning
ALIGN_BLOCK = 40                # source lines aligned per window
ALIGN_WINDOW_LOCKED = 60.0      # seconds of slack around a block once the offset is known
ALIGN_WINDOW_SEARCH = 300.0     # seconds of slack while (re)finding the offset
ALIGN_NEIGHBOURS = 7            # confident lines on each side used for the local offset
SHIFT_MIN_LOCAL = 0.5           # shift a region only if its offset is at least this (s)
SHIFT_MIN_LINE = 1.5            # shift a single line only if it deviates this much (s)
MIN_COVERAGE = 0.25             # below this, the subtitles don't match the audio
OVERLAP_GAP = 0.04
MIN_READ_S = 1.0                # a neighbour is never squeezed below this...
READ_CPS = 17.0                 # ...or below its length at this reading speed

# Report-only checks
MAX_CPS = 20.0
MIN_DURATION = 0.5


# ---------------------------------------------------------------------------
# SRT I/O (keeps text verbatim — unlike the source parser, nothing is stripped)
# ---------------------------------------------------------------------------

def parse_srt(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        content = f.read()
    out = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s+-->\s+(\d+):(\d+):(\d+)[,.](\d+)", lines[ts_idx])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        out.append({"start": start, "end": end, "text": "\n".join(lines[ts_idx + 1:]).strip()})
    return out


def pair_srt_with_source(srt_entries, source_segments):
    """Map entries of an existing Hebrew .srt back to source segment ids."""
    if len(srt_entries) == len(source_segments):
        return [
            {"id": src["id"], "start": e["start"], "end": e["end"], "text": e["text"]}
            for src, e in zip(source_segments, srt_entries)
        ]
    # Counts differ: match each entry to the nearest unused source line by start time.
    starts = [s["start"] for s in source_segments]
    used = set()
    paired = []
    next_free_id = max((s["id"] for s in source_segments), default=-1) + 1
    for e in srt_entries:
        i = bisect.bisect_left(starts, e["start"])
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(starts) and j not in used and abs(starts[j] - e["start"]) <= 1.0:
                if best is None or abs(starts[j] - e["start"]) < abs(starts[best] - e["start"]):
                    best = j
        if best is None:
            paired.append({"id": next_free_id, "start": e["start"], "end": e["end"],
                           "text": e["text"], "unpaired": True})
            next_free_id += 1
        else:
            used.add(best)
            paired.append({"id": source_segments[best]["id"], "start": e["start"],
                           "end": e["end"], "text": e["text"]})
    return paired


# ---------------------------------------------------------------------------
# 1. Structural
# ---------------------------------------------------------------------------

def restore_structure(source_segments, translated, report):
    """
    One output line per source line, with the source timestamps.
    The translation model sometimes echoes back altered timestamps — those are reverted here.
    Lines the translator never returned are kept as the English source and flagged.
    """
    by_id = {s["id"]: s for s in translated}
    out = []
    for src in source_segments:
        tr = by_id.get(src["id"])
        if tr is None:
            report["structural"].append({"id": src["id"], "issue": "missing_translation"})
            out.append({"id": src["id"], "start": src["start"], "end": src["end"], "text": src["text"]})
            continue
        if abs(tr.get("start", src["start"]) - src["start"]) > 0.001 or \
           abs(tr.get("end", src["end"]) - src["end"]) > 0.001:
            report["structural"].append({
                "id": src["id"], "issue": "timestamp_restored",
                "was": [tr.get("start"), tr.get("end")], "now": [src["start"], src["end"]],
            })
        out.append({"id": src["id"], "start": src["start"], "end": src["end"],
                    "text": str(tr.get("text", "")).strip()})
    # Keep lines from an existing .srt that had no source counterpart.
    for tr in translated:
        if tr.get("unpaired"):
            out.append(dict(tr))
            report["structural"].append({"id": tr["id"], "issue": "no_source_line"})
    out.sort(key=lambda s: s["start"])
    return out


# ---------------------------------------------------------------------------
# 2. Timing alignment against the audio
# ---------------------------------------------------------------------------

def _norm_tokens(text):
    text = re.sub(r"<[^>]+>|\{[^}]+\}|\[[^\]]*\]|\([^)]*\)", " ", text.lower())
    text = text.replace("'", "").replace("’", "")
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def _match_lines_to_words(source_segments, words):
    """
    Returns per-line measurements {idx: (d, matched, ntok)} where d is how far (s) the
    audio is from where the subtitle places the line (positive = speech is later).
    """
    wtok, wtime = [], []
    for w in words:
        for t in _norm_tokens(w[0]):
            wtok.append(t)
            wtime.append(w[1])
    if not wtok:
        return {}

    seg_tokens = [_norm_tokens(s["text"]) for s in source_segments]
    measured = {}
    recent = []
    offset_est = 0.0
    locked = False
    n = len(source_segments)

    for b in range(0, n, ALIGN_BLOCK):
        blk = range(b, min(b + ALIGN_BLOCK, n))
        a_tok, a_own = [], []
        for i in blk:
            for k, t in enumerate(seg_tokens[i]):
                a_tok.append(t)
                a_own.append((i, k))
        if not a_tok:
            continue

        win = ALIGN_WINDOW_LOCKED if locked else ALIGN_WINDOW_SEARCH
        t0 = source_segments[blk[0]]["start"] + offset_est - win
        t1 = source_segments[blk[-1]]["end"] + offset_est + win
        lo, hi = bisect.bisect_left(wtime, t0), bisect.bisect_right(wtime, t1)
        if hi - lo < 2:
            locked = False
            continue

        sm = difflib.SequenceMatcher(None, a_tok, wtok[lo:hi], autojunk=False)
        hits = {}
        for m in sm.get_matching_blocks():
            if m.size < 2:  # single-word matches in a wide window are mostly noise
                continue
            for j in range(m.size):
                i, k = a_own[m.a + j]
                hits.setdefault(i, []).append((k, wtime[lo + m.b + j]))

        found = 0
        for i, pairs in hits.items():
            seg = source_segments[i]
            ntok = len(seg_tokens[i])
            dur = max(seg["end"] - seg["start"], 0.1)
            times = [t for _, t in pairs]
            if max(times) - min(times) > dur + 4.0:
                continue  # matched words are scattered — not a real match
            # Subtitles stay on screen after speech ends, so words are not spread evenly over
            # the line. Prefer the onset (first word) when it matched; otherwise estimate.
            onset = [t for k, t in pairs if k == 0]
            if onset:
                d = onset[0] - seg["start"]
            else:
                d = statistics.median(t - (seg["start"] + dur * k / ntok) for k, t in pairs)
            measured[i] = (d, len(pairs), ntok)
            if len(pairs) >= 2 and len(pairs) / ntok >= 0.5:
                recent.append(measured[i][0])
                found += 1
        if found:
            recent = recent[-25:]
            offset_est = statistics.median(recent)
            locked = True
        else:
            locked = False

    return measured


def _min_read(text):
    """Shortest time a subtitle should stay on screen to be readable."""
    return max(MIN_READ_S, len(re.sub(r"\s+", "", text)) / READ_CPS)


def align_timing(segments, source_segments, words, report):
    """Shift subtitle timings to match the speech in the audio. Mutates and returns segments."""
    src_index = {s["id"]: i for i, s in enumerate(source_segments)}
    measured = _match_lines_to_words(source_segments, words)

    eligible = [i for i, s in enumerate(source_segments) if len(_norm_tokens(s["text"])) >= 2]
    confident = sorted(i for i, (d, m, nt) in measured.items() if m >= 2 and m / nt >= 0.5)
    coverage = len(confident) / max(len(eligible), 1)

    timing = report["timing"]
    timing["lines_matched_to_audio"] = len(confident)
    timing["coverage"] = round(coverage, 3)

    if coverage < MIN_COVERAGE:
        report["warnings"].append(
            f"Only {coverage:.0%} of subtitle lines could be matched to the audio. The source "
            "subtitles probably don't belong to this video (wrong track or release), or the "
            "audio is mostly non-English. Timing was NOT changed — try --subtitle-track, or "
            "delete the cache so Whisper transcribes instead."
        )
        timing["applied"] = False
        return segments

    timing["applied"] = True
    ds = [measured[i][0] for i in confident]

    # Drift summary (linear fit of offset vs. time) for the report.
    xs = [source_segments[i]["start"] for i in confident]
    if len(xs) >= 10 and max(xs) - min(xs) > 60:
        mx, my = statistics.mean(xs), statistics.mean(ds)
        var = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ds)) / var if var else 0.0
        timing["offset_at_start_s"] = round(statistics.median(ds[:30]), 3)
        timing["offset_at_end_s"] = round(statistics.median(ds[-30:]), 3)
        timing["drift_s_per_hour"] = round(slope * 3600, 3)

    def local_offset(i):
        p = bisect.bisect_left(confident, i)
        neigh = confident[max(0, p - ALIGN_NEIGHBOURS):p + ALIGN_NEIGHBOURS]
        return statistics.median(measured[j][0] for j in neigh)

    # Region shift for every line first, so single-line moves can be checked against neighbours.
    plan = {}
    for seg in segments:
        i = src_index.get(seg["id"])
        if i is None:
            continue
        local = local_offset(i)
        plan[seg["id"]] = (local, "region_offset") if abs(local) >= SHIFT_MIN_LOCAL else (0.0, None)
    start_by_id = {s["id"]: s["start"] for s in segments}
    text_by_id = {s["id"]: s["text"] for s in segments}
    base_start = {sid: start_by_id[sid] + sh for sid, (sh, _) in plan.items()}
    # Neighbours in dialogue order (source line order), not current time order.
    ordered = [s["id"] for s in source_segments if s["id"] in base_start]
    rank = {sid: n for n, sid in enumerate(ordered)}

    for seg in segments:
        i = src_index.get(seg["id"])
        if i is None or i not in measured:
            continue
        d, m, nt = measured[i]
        local = plan[seg["id"]][0]
        strong = (m >= 3 and m / nt >= 0.6) or (m >= 2 and m == nt)
        if not strong or abs(d - local) < SHIFT_MIN_LINE:
            continue
        # Repeated nearby text ("No, no." / duplicated lines) can match the wrong occurrence.
        toks = _norm_tokens(source_segments[i]["text"])
        if any(_norm_tokens(source_segments[j]["text"]) == toks
               for j in range(max(0, i - 3), min(len(source_segments), i + 4)) if j != i):
            continue
        # A genuinely misplaced line lands back between its neighbours; a false match
        # (repeated phrase elsewhere) would jump past them.
        new_start = seg["start"] + d
        r = rank[seg["id"]]
        prev_id = ordered[r - 1] if r > 0 else None
        next_id = ordered[r + 1] if r + 1 < len(ordered) else None
        prev_start = base_start[prev_id] if prev_id is not None else float("-inf")
        next_start = base_start[next_id] if next_id is not None else float("inf")
        if not (prev_start <= new_start <= next_start):
            continue
        # Subtitlers often delay a line on purpose so the previous one stays readable.
        # Move toward the speech, but never leave the previous line (or this one, when
        # moving later) shorter than its reading time.
        if d < local and prev_id is not None:      # moving earlier: protect the previous line
            new_start = max(new_start, prev_start + _min_read(text_by_id[prev_id]) + OVERLAP_GAP)
        if d > local and next_id is not None:      # moving later: keep this line readable
            new_start = min(new_start, next_start - _min_read(seg["text"]) - OVERLAP_GAP)
        clamped = new_start - seg["start"]
        if abs(clamped - local) >= SHIFT_MIN_LOCAL and (clamped - local) * (d - local) > 0:
            plan[seg["id"]] = (clamped, "line_misplaced")

    moved = set()
    for seg in segments:
        if seg["id"] not in plan:
            continue
        shift, reason = plan[seg["id"]]
        if reason:
            shift = round(shift, 3)
            new_start = max(0.0, seg["start"] + shift)
            new_end = max(new_start + 0.3, seg["end"] + shift)
            timing["changes"].append({
                "id": seg["id"], "reason": reason, "shift_s": shift,
                "was": [round(seg["start"], 3), round(seg["end"], 3)],
                "now": [round(new_start, 3), round(new_end, 3)],
            })
            seg["start"], seg["end"] = new_start, new_end
            moved.add(seg["id"])

    segments.sort(key=lambda s: s["start"])

    # Trim overlaps that QA itself introduced.
    for a, b in zip(segments, segments[1:]):
        if (a["id"] in moved or b["id"] in moved) and a["end"] > b["start"] - OVERLAP_GAP:
            a["end"] = max(a["start"] + 0.3, b["start"] - OVERLAP_GAP)

    return segments


# ---------------------------------------------------------------------------
# 3. Language review with Claude
# ---------------------------------------------------------------------------

REVIEW_SYSTEM = """\
You are the final QA reviewer for Hebrew subtitles translated from English.

{gender_context}

Each line has: id, speaker (name and gender, when known), en (English source) and he (current Hebrew subtitle).
Lines marked "context": true are only there so you can see the surrounding conversation — never fix them.

Find and fix ONLY real errors in these categories:
- gender: wrong grammatical gender for the speaker (self-reference: אני יודע/יודעת, עייף/עייפה),
  for the addressee (אתה/את, imperatives like בוא/בואי, תגיד/תגידי, adjectives),
  or for a third person (הוא/היא, אמר/אמרה, שלו/שלה). Use the cast reference, the speaker
  label and the conversation to decide who is speaking and who is being addressed. Plural address
  to a mixed or unknown group uses masculine plural.
- misaligned: the Hebrew is a translation of a NEIGHBOURING English line (text slid up/down),
  or two lines were merged/split so this line's Hebrew does not match its own English.
  The fix is the correct Hebrew for THIS line's English.
- untranslated: Hebrew is empty, or the line is still in English (names and brand names are fine).
- mistranslation: the meaning is clearly wrong or a meaningful part of the English is missing.

Do NOT rephrase lines that are already correct, do not "improve" style or word choice, and do not
change punctuation only. When in doubt, leave the line alone. Keep fixes natural, fluent subtitle
Hebrew, similar in length to the original line.

Return every fix in "fixes". Return an empty list if everything is correct.
"""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "category": {"type": "string",
                                 "enum": ["gender", "misaligned", "untranslated", "mistranslation"]},
                    "fixed_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "category", "fixed_text", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["fixes"],
    "additionalProperties": False,
}


def _speaker_label(src, speaker_map):
    sid = src.get("speaker")
    if not sid or sid == "UNKNOWN":
        return None
    info = speaker_map.get(sid, {})
    return f"{info.get('name', sid)} ({info.get('gender', 'unknown').upper()})"


def review_language(segments, source_segments, gender_context, speaker_map, report, log=print):
    """Claude reviews Hebrew vs. English and returns segments with fixes applied."""
    src_by_id = {s["id"]: s for s in source_segments}
    rows = []
    for seg in segments:
        src = src_by_id.get(seg["id"], {})
        row = {"id": seg["id"], "en": src.get("text", ""), "he": seg["text"]}
        label = _speaker_label(src, speaker_map)
        if label:
            row["speaker"] = label
        rows.append(row)

    system = REVIEW_SYSTEM.format(
        gender_context=gender_context or "(No cast reference available — infer genders from context.)"
    )
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    fixes_by_id = {}

    total_batches = (len(rows) + QA_REVIEW_BATCH - 1) // QA_REVIEW_BATCH
    for bn, b in enumerate(range(0, len(rows), QA_REVIEW_BATCH), 1):
        batch = rows[b:b + QA_REVIEW_BATCH]
        before = [dict(r, context=True) for r in rows[max(0, b - QA_REVIEW_CONTEXT):b]]
        after = [dict(r, context=True) for r in rows[b + QA_REVIEW_BATCH:b + QA_REVIEW_BATCH + QA_REVIEW_CONTEXT]]
        allowed = {r["id"] for r in batch}
        user_msg = (
            "Review these subtitle lines.\n\n"
            + json.dumps(before + batch + after, ensure_ascii=False, indent=1)
        )
        log(f"  Review batch {bn}/{total_batches}", end="", flush=True)
        fixes = _review_call(client, system, user_msg)
        kept = 0
        for f in fixes:
            if f.get("id") in allowed and f.get("fixed_text", "").strip():
                fixes_by_id[f["id"]] = f
                kept += 1
        log(f" ✓ ({kept} fixes)")

    for seg in segments:
        f = fixes_by_id.get(seg["id"])
        if not f:
            continue
        new_text = f["fixed_text"].strip()
        if new_text == seg["text"].strip():
            continue
        report["text"].append({
            "id": seg["id"], "category": f["category"], "reason": f["reason"],
            "en": src_by_id.get(seg["id"], {}).get("text", ""),
            "before": seg["text"], "after": new_text,
        })
        seg["text"] = new_text
    return segments


def _review_call(client, system, user_msg):
    for attempt in range(3):
        try:
            with client.beta.messages.stream(
                model=QA_MODEL,
                max_tokens=64000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": REVIEW_SCHEMA},
                },
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                response = stream.get_final_message()
            if response.stop_reason == "refusal":
                print(" (review refused for this batch — skipped)", end="")
                return []
            if response.stop_reason == "max_tokens":
                raise ValueError("review response hit max_tokens")
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text).get("fixes", [])
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
                ValueError, json.JSONDecodeError, StopIteration) as e:
            if attempt == 2:
                print(f" (review failed: {e} — batch skipped)", end="")
                return []
            time.sleep(10 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
# 4. Report-only checks
# ---------------------------------------------------------------------------

def flag_checks(segments, report):
    for seg in segments:
        text = seg["text"].strip()
        dur = seg["end"] - seg["start"]
        chars = len(re.sub(r"\s+", "", text))
        if not text:
            report["flags"].append({"id": seg["id"], "issue": "empty_line"})
            continue
        if dur < MIN_DURATION:
            report["flags"].append({"id": seg["id"], "issue": "very_short", "duration_s": round(dur, 3)})
        if dur > 0 and chars / dur > MAX_CPS:
            report["flags"].append({"id": seg["id"], "issue": "fast_reading_speed",
                                    "chars_per_s": round(chars / dur, 1)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_qa(source_segments, translated, words, gender_context, speaker_map, log=print):
    """
    source_segments: English segments (with ids, timestamps, optional 'speaker').
    translated:      Hebrew segments with matching ids.
    words:           [[word, start, end], ...] from a word-level Whisper pass (may be empty).
    Returns (fixed_segments, report).
    """
    report = {"structural": [], "timing": {"changes": []}, "text": [], "flags": [], "warnings": []}

    log("QA 1/4: structure (ids, timestamps, ordering)...")
    segments = restore_structure(source_segments, translated, report)
    log(f"  {len(report['structural'])} structural issue(s) fixed")

    log("QA 2/4: timing vs. audio...")
    if words:
        segments = align_timing(segments, source_segments, words, report)
        t = report["timing"]
        log(f"  matched {t.get('lines_matched_to_audio', 0)} lines ({t.get('coverage', 0):.0%}), "
            f"{len(t['changes'])} timing change(s)")
        if "drift_s_per_hour" in t:
            log(f"  offset start {t['offset_at_start_s']:+.2f}s, end {t['offset_at_end_s']:+.2f}s, "
                f"drift {t['drift_s_per_hour']:+.2f}s/h")
    else:
        report["warnings"].append("No word-level transcript available — timing check skipped.")
        log("  skipped (no word transcript)")

    log("QA 3/4: language review (gender, alignment, untranslated)...")
    segments = review_language(segments, source_segments, gender_context, speaker_map, report, log=log)
    by_cat = {}
    for f in report["text"]:
        by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
    log(f"  {len(report['text'])} text fix(es) {by_cat if by_cat else ''}")

    log("QA 4/4: readability checks...")
    flag_checks(segments, report)
    log(f"  {len(report['flags'])} line(s) flagged (report only)")

    for w in report["warnings"]:
        log(f"  WARNING: {w}")

    report["summary"] = {
        "lines": len(segments),
        "structural_fixes": len(report["structural"]),
        "timing_changes": len(report["timing"]["changes"]),
        "text_fixes": len(report["text"]),
        "text_fixes_by_category": by_cat,
        "flags": len(report["flags"]),
        "warnings": len(report["warnings"]),
    }
    return segments, report
