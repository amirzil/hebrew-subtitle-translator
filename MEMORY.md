# MEMORY

## 2026-11-24 — QA stage after translation

**Decided:** Add a QA pass (`qa.py`) that runs after translation and auto-fixes the `.srt`, keeping a `.pre-qa.srt` backup and writing a `.qa.json` report.
- Timing: align English source lines against a *word-level* Whisper transcript of the real audio (windowed `difflib` matching, local median offset). Region shifts fix offsets/drift/jumps; single-line moves only with strong evidence, when the line lands back between its dialogue-order neighbours, and not when nearby text repeats.
- If < 25% of lines match the audio → warn (wrong track/release) and don't touch timing.
- Language review: Claude (`claude-opus-5-5`, structured output, server-side refusal fallback) checks Hebrew vs English + speaker + cast reference for gender / misaligned / untranslated / mistranslation. Only real errors, no restyling.
- Translation now always restores source timestamps (model sometimes echoed altered ones); Whisper chunk merge now de-dupes by owned time window and actually sorts.

**Why:** Users saw subtitles going out of sync and wrong masculine/feminine forms.

**Rejected:**
- Speech-activity (VAD) cross-correlation only — catches global offset/drift but not single misplaced lines or wrong-track detection.
- Sending video frames to Claude for gender — much more expensive; can be added later for still-ambiguous lines only.
- Report-only QA — user chose auto-fix with backup.
- Snapping every line to Whisper word times — Whisper word timestamps are less precise than embedded subs; only shift when deviation is ≥ 0.5 s (region) / ≥ 1.5 s (single line).
