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

## Output

- `.srt` file encoded as UTF-8 BOM for correct RTL Hebrew display in VLC and Windows media players
- Timestamps preserved exactly from Whisper transcription
- Defaults to masculine forms only when gender is truly ambiguous
