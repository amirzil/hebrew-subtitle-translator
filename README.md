# Hebrew Subtitle Translator

A CLI tool that takes any video file and produces a Hebrew `.srt` subtitle file, with Claude AI inferring grammatical gender (speaker gender and addressee gender) from the full conversation context.

## How it works

```
Video → ffmpeg (audio extraction) → Whisper API (transcription) → Claude (translation) → .srt file
```

**Two-pass translation:**
1. **Gender analysis** — Claude reads the full transcript and builds a cast list with each character's gender, backed by evidence from the dialogue
2. **Translation** — every batch is translated with the full cast reference injected into the prompt, ensuring consistent gendered Hebrew forms throughout

This means a character identified as female in minute 2 will be referred to correctly in minute 48.

## What "gender-aware" means

Hebrew grammar encodes gender in verbs, pronouns, imperatives, and adjectives. This tool handles:

| Category | Masculine | Feminine |
|---|---|---|
| 2nd-person pronoun | אתה | את |
| Imperative (go) | לך! | לכי! |
| Past tense (she went) | הוא הלך | היא הלכה |
| Adjective | אתה חזק | את חזקה |

## Requirements

- Python 3.9+
- ffmpeg — `brew install ffmpeg`
- OpenAI API key (Whisper)
- Anthropic API key (Claude)

## Installation

```bash
git clone https://github.com/amirzil/hebrew-subtitle-translator
cd hebrew-subtitle-translator
pip3 install -r requirements.txt
brew install ffmpeg
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

```bash
python3 translator.py input.mp4
```

```bash
# Custom output path
python3 translator.py input.mp4 -o subtitles.srt

# Provide gender hints
python3 translator.py interview.mp4 --hints "female interviewer, male guest"

# Keep the extracted audio file
python3 translator.py input.mp4 --keep-audio
```

## QA pass (runs automatically after translation)

After the `.srt` is written, a QA pass checks it against the real video and auto-fixes it:

1. **Structure** — every subtitle keeps the source timestamp and its own line (the translation model sometimes alters timestamps or slides text into a neighbouring line).
2. **Timing vs. audio** — the audio is transcribed again with Whisper *word* timestamps and every English source line is aligned to the speech. Constant offsets, drift (e.g. a subtitle track from a release with a different frame rate), jumps (cut scenes) and individually misplaced lines are corrected. If most lines don't match the audio at all, the subtitle track is probably wrong — QA warns and leaves timing alone.
3. **Language review** — Claude re-reads every Hebrew line next to its English source, the speaker and the cast reference, and fixes wrong gender forms, misaligned lines, untranslated lines and clear mistranslations. It does not restyle correct lines.
4. **Readability flags** — very fast reading speed, very short or empty lines (reported only).

Files written next to the video:

| File | What |
|---|---|
| `movie.srt` | Final, QA-fixed subtitles |
| `movie.pre-qa.srt` | Backup of the subtitles before QA |
| `movie.qa.json` | Report: every timing/text change (before → after, with reason), flags, warnings |

```bash
# QA an existing .srt without re-translating
python3 translator.py movie.mkv --qa-only

# Translate without QA
python3 translator.py movie.mkv --no-qa
```

The word transcript is cached in `<video>.cache.json`, so re-running QA only costs the Claude review.

## Output

- `.srt` file encoded as UTF-8 BOM for correct RTL Hebrew display in VLC and Windows media players
- Timestamps preserved exactly from Whisper transcription
- Defaults to masculine forms only when gender is truly ambiguous
