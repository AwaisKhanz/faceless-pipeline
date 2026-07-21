# Faceless Studio

Turns a script into a finished YouTube video — in English, German and Spanish — on your
own computer. Windows or macOS. No subscriptions, no watermarks, no scene limit.

```
   script  →  production sheets  →  visuals  →  narration  →  MP4
```

---

# 1 · First-time setup

About 20 minutes, once. After this you never do it again.

**This runs on Windows and on macOS.** Follow 1.1W–1.3W for Windows, or 1.1M–1.3M for a Mac.
Everything from section 2 onwards is identical on both.

> **Which machine should you use?** If you have an NVIDIA graphics card, use it. Voice
> generation is many times faster, and — more importantly — the model runs in the card's own
> memory instead of competing with your operating system for RAM. On a Mac the GPU shares
> system memory, so a long voicing job can slow the whole machine to a crawl.

---

## Windows

### 1.1W Python

Install **Python 3.12** from [python.org/downloads](https://www.python.org/downloads/).

On the very first screen of the installer, tick **"Add python.exe to PATH"**. It's easy to
miss and everything else fails without it.

> Newer Python versions often work, but the machine-learning packages lag new releases by
> months. 3.12 is the version this stack is happiest on.

### 1.2W ffmpeg

This does the actual video work. In **PowerShell**:

```powershell
winget install Gyan.FFmpeg
```

Close and re-open PowerShell afterwards, then check it took:

```powershell
ffmpeg -version
```

If `winget` isn't available, download a build from
[gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) (take the *full* build — the
*essentials* build lacks the text libraries, and burned-in captions won't work) and add its
`bin` folder to your PATH.

### 1.3W Everything else

**Double-click `setup.bat`.**

It builds a private Python environment inside the project folder and installs the voice
engine. Several GB — leave it running.

If it finds an NVIDIA card it installs the **CUDA 12.8** build of PyTorch, which is what
RTX 50-series (Blackwell) cards need. Older builds install without complaint and then fail
to launch a single kernel on the card, so the last thing `setup.bat` does is multiply two
matrices on the GPU and confirm the answer came back. If that check fails it tells you the
exact command to fix it.

You need the NVIDIA **driver** installed, but *not* the CUDA Toolkit — the PyTorch package
bundles its own CUDA runtime.

---

## macOS

### 1.1M Homebrew

Open **Terminal** (Cmd+Space, type "Terminal") and check:

```bash
brew --version
```

If that says *command not found*, install it, then re-open Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.2M ffmpeg

```bash
brew install ffmpeg
```

**Install the plain `ffmpeg` formula** — some taps ship a stripped-down build without the
text libraries, and then burned-in captions can't work. Section 6 explains how to tell.

### 1.3M Python packages

```bash
cd ~/Documents/faceless-pipeline
bash setup.sh
```

This creates a `.venv` folder inside the project and installs the voice engine there.
It pulls in PyTorch, so it is a large download and takes a few minutes.

> **Why a .venv?** Homebrew's Python blocks system-wide `pip install` (PEP 668). If you
> run `python3 -m pip install …` yourself you'll get *externally-managed-environment*.
> `setup.sh` sidesteps that, and every script here hands itself over to `.venv`
> automatically — so you keep typing plain `python3` and it just works.

## 1.4 API keys

Two are required, one is optional. All free.

| Key | Needed for | Get it at |
|---|---|---|
| **Pexels** | stock photos and clips | https://www.pexels.com/api/ |
| **Pixabay** | fallback when Pexels has no match | https://pixabay.com/api/docs/ |
| **Gemini** | *optional* — turning a script into sheets | https://aistudio.google.com/apikey |

Copy the template, then open it:

```powershell
copy config.example.json config.json      :: Windows
notepad config.json
```

```bash
cp config.example.json config.json        # macOS
open -e config.json
```

Paste each key **between the existing quote marks**:

```json
"pexels_key": "your-key-here",
"pixabay_key": "your-key-here",
"gemini_key": ""
```

> Use a plain text editor — Notepad on Windows, `open -e` (not a double-click) on macOS.
> TextEdit's rich-text mode and Word both turn `"` into curly quotes, which silently
> breaks the file.

Prefer keys out of files? Set `PEXELS_API_KEY`, `PIXABAY_API_KEY` and `GEMINI_API_KEY`
as environment variables instead — those win over `config.json`.

## 1.5 Check it

```bash
faceless check
```

Everything you need should show `ok`. Anything showing `!!` is explained in section 6.

The two lines that matter most:

```
ok chatterbox  installed · NVIDIA GeForce RTX 5060 Ti · 17.1 GB VRAM · sm_120
ok captions    ass: libass present — full styled captions
```

If `chatterbox` says `CPU` on a machine with an NVIDIA card, stop and fix that first —
section 6 explains why, and it's a one-line fix. Nothing errors; voicing is just twenty
times slower than it should be.

---

# 2 · Making a video

Open the control panel:

```bash
faceless start
```

Your browser opens to the panel. Leave the terminal open while you work; press Ctrl+C in
it when you're finished. `faceless start --no-browser` if you'd rather open the page
yourself.

The panel has four sections, and each is a real page — Back and Forward work, and any
view can be bookmarked or reloaded:

| | |
|---|---|
| **Dashboard** | every project, with a four-step bar per language: sheets, visuals, voice, render. Tells you at a glance what is done and what isn't. |
| **Project** | one video. Source or review visuals; voice, re-voice, render or re-render any single language without redoing the rest. |
| **Activity** | live terminal with timestamps, plus exact counts — scene *n* of *N*, elapsed, estimated remaining, seconds per item, and a per-step breakdown once each finishes. |
| **Voices / Settings** | audition reference clips and choose one per language; check the machine, GPU, keys and finished files. |

There's a light/dark toggle at the bottom of the sidebar; the choice is remembered.

Status is **derived from what's on disk**, never stored. Delete an MP4 or clear a cache by
hand and the dashboard tells the truth on the next refresh — it can't claim something
exists when it doesn't.

`faceless` is installed by the setup script as a normal console command — the same
mechanism behind `npm start` or `git`. It works from any folder inside the project, as
long as the `.venv` is active.

Three ways to do the identical thing, in case one isn't available:

| | |
|---|---|
| `faceless start` | the short form, once setup has run |
| `python3 make_video.py studio` | always works, no install needed |
| double-click `Start.bat` / `Start.command` | no terminal at all |

**A note on `python` vs `python3`:** on Windows type `python`, on macOS type `python3`.
That's the only difference between the platforms once setup is done. `faceless` is
spelled the same on both.

## Step 0 · Start from a script *(optional)*

Paste a finished script, name it (`video06`), tick the languages, click
**Generate the sheets**.

Gemini splits it into scenes and writes the master sheet, the German and Spanish
narration files, and the YouTube packages. They appear in the dropdown below.

- **Your words are never rewritten.** Gemini only decides *where to cut*. Every section
  is checked word-by-word against what you pasted; mismatches are retried twice with the
  error fed back, then shown to you as a diff. Nothing changes silently.
- **The file format can't come out wrong.** Gemini returns structured data, never
  markdown — Python writes every character of the file.
- Uses roughly 15–25 requests per video, against a free allowance of ~250/day.

Skip this step entirely if you write your sheets by hand.

## Step 1 · Find visuals

Pick the video, tick the languages, choose music if you want it, click **Find visuals**.

It sources a photo or clip for every scene. A couple of minutes for 115 scenes.

## Step 2 · Check the visuals

Every scene appears as a card showing its picture next to the line it has to carry.

- **Click any picture that doesn't suit its line.** That's all rejecting means.
- **Gold border = a scene that carries the story** — recurring characters, title cards,
  the payoff line. Worth a proper look; scan the rest.
- **Anything you leave alone is approved.** There's no approve button.

**Get new pictures** fetches the next-best match for just the ones you marked. Click the
same scene three times and you'll get the fourth-best result.

## Step 3 · Build

Click **Looks good — build the videos**. Narration is generated and every ticked language
rendered, with progress as it goes.

Timing depends almost entirely on whether you have an NVIDIA card. On an RTX 5060 Ti,
narration for a 115-scene video takes about 7 minutes per language; on Apple Silicon it
is closer to 100. Rendering the picture is similar on both — it's ffmpeg, not the GPU.
Run `benchtts` once to see where your machine sits.

Finished files land in `out/`, with a **Show files in Finder** button.

---

# 3 · Voices

Narration is cloned from a reference clip by **Chatterbox** — MIT licensed, running
locally on your own machine, free and unlimited. There is no per-character cost and nothing is
sent to a third party.

## Where clips live

One folder per language, named by language code:

```
voices_refs/
  en/    warm-documentary-male.mp3
  de/    ruhige-erzaehlerin.mp3
  es/    narrador-calido.mp3
```

Each language has its own choice, so English can be read by one voice and German by
another. A clip in `en/` is only offered when you're choosing an English voice — a German
list full of English voices is noise, not choice.

Files left loose in `voices_refs/` still work. They appear under **Not sorted yet** and
are offered for every language, and the panel has a button that files them away when the
name makes the language obvious. Nothing breaks if you just drop a file in.

## Adding a voice

1. **Put the file in the language folder** — `voices_refs/en/your-clip.mp3`. Create the
   folder if it isn't there; use the two-letter code from the list at the end of this
   section.

2. **Name it for how it sounds.** The filename becomes the label, tidied up:

   | file | shows as |
   |---|---|
   | `warm-documentary-male.mp3` | Warm documentary male |
   | `calm-older-female.mp3` | Calm older female |
   | `german-narrator-1.mp3` | German narrator 1 |

   Dashes and underscores both work. The filename stays the identity; the label is only
   what you read.

3. **Reload the Voices panel.** The clip appears under that language, with its length.

4. **Preview it.** It reads a real line from your own script, not "hello world" — how a
   voice handles your actual writing is the only thing worth judging.

5. **Use this** saves it, together with the current Expression and Guidance settings.

6. **If that language already had narration, re-voice it.** Audio already generated stays
   cached under the *previous* voice and would otherwise be silently reused. Project page
   → that language → **Redo**.

Formats: `.wav .mp3 .m4a .flac .ogg .aac`. The pipeline writes its own normalised copy
into `cache/refs/` — mono, 24 kHz, silence trimmed, levels evened — so you never need to
edit anything by hand. That cache is disposable and rebuilds itself.

## What makes a good clip

| | |
|---|---|
| **30+ seconds** | clones far better than 10; under 8s is flagged in the panel |
| **one speaker** | no interviews, no overlapping voices |
| **clean** | no music, no background noise, no heavy reverb |
| **the right pace** | it copies delivery, so use the speed you want your videos read |
| **plain speech** | ordinary sentences, not shouting or whispering |

## Rights

**Whatever you put there becomes your channel's voice**, on every video, publicly. It has
to be audio you may use that way.

- **Your own voice** — 30 seconds on a phone. No licensing question at all, and nobody
  else on YouTube has it.
- **Mozilla Common Voice** (https://commonvoice.mozilla.org/) — released CC0, an explicit
  public-domain dedication. Thousands of speakers across many languages.
- **LibriVox** (https://librivox.org/) — public-domain audiobook readings.

Audio from a paid AI voice service, or lifted from someone else's video, is not usable —
the first breaks those services' terms, the second is a person's voice and likeness.

**30+ seconds of clean, continuous speech clones far better than 10.** One speaker, no
music, no background noise, at the pace you want your videos read. Clips under 8 seconds
are flagged in the panel.

## Choosing one

Two places, one setting:

- **On the project page** — each language row has a **Voice** dropdown. Change it and it
  saves immediately. This is the quick way once you know your clips.
- **In the Voices panel** — tabs per language, with **Preview** to hear a clip read *a
  real line from your own script* before committing, plus the Expression and Guidance
  sliders.

Both write the same `voices.json`, so they can't disagree. Voice and Render are disabled
for a language until it has one, rather than failing after you click.

> Changing a language's voice changes what counts as "voiced". Narration is cached per
> voice, so switching from one clip to another resets that language's progress to 0 —
> the old audio is still on disk and comes back if you switch back.

Two sliders:

| | |
|---|---|
| **Expression** | low = calm and even, high = performed. 0.40 suits documentary narration. |
| **Guidance** | how closely it copies the reference. 0.50 is a good starting point. |

## Speed

Chatterbox is slower than a cloud service, and *how much* slower depends enormously on
your hardware. Before committing:

```bash
python3 make_video.py benchtts --lang en
```

It times five real lines from your sheets, measures **seconds per character** (line lengths
vary too much for a per-line average to mean anything), takes the median, and extrapolates
to a full video in three languages.

Two things it deliberately handles:

- **The first line is always slower** — the GPU compiles kernels on its first real call.
  That cost is paid once per session, not per line, so it's reported separately and left
  out of the estimate.
- **If the later lines are slower than the earlier ones**, it says so and refuses to treat
  the estimate as reliable. That direction of drift means the machine is running out of
  memory as it goes, which is a different problem from being slow.

Measured on the two machines this was built against, same 115-scene script:

| | Apple Silicon (MPS) | RTX 5060 Ti (CUDA) |
|---|---|---|
| per line | 53.7s | **3.8s** |
| one language | 103 min | **7 min** |
| all three languages | 5+ hours | **22 min** |

The Mac wasn't just slower — it degraded as it ran, from 27 it/s down to 2.5, because the
GPU shares system memory with the OS. The PC held 55 it/s flat across every line.

**On a Mac**, the GPU shares system memory. A long job can exhaust it, and then everything
slows down together — the model, and your desktop. If that happens, force the CPU:

```bash
FACELESS_DEVICE=cpu python3 make_video.py voice --sheet … --lang en
```

**On an NVIDIA card**, the model lives in the card's own VRAM. Running out gives you a
clean error instead of a frozen computer. 8 GB is comfortable; 16 GB is plenty.

`doctor` tells you which you have, and how much memory it has.

Voicing is cached either way, so a re-render never repeats the work.

## Languages

23 supported: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek,
Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian,
Spanish, Swahili, Swedish, Turkish.

---

# 4 · Command line

Everything the studio does, for scripting or troubleshooting.

`faceless <verb>` is the short form. Every verb maps to a `make_video.py` step, so the
two columns below are interchangeable — use whichever you prefer.

| short | long | what it does |
|---|---|---|
| `faceless start` | `make_video.py studio` | open the control panel |
| `faceless check` | `make_video.py doctor` | check this machine |
| `faceless sheets` | `make_video.py list` | projects found in `sheets/` |
| `faceless bench` | `make_video.py benchtts` | time the voice engine |
| `faceless new` | `make_video.py generate` | script → production sheets |
| `faceless visuals` | `make_video.py stock` | source photos and clips |
| `faceless voice` | `make_video.py voice` | generate narration |
| `faceless render` | `make_video.py render` | build the MP4 |
| `faceless all` | `make_video.py all` | everything, start to finish |

`faceless --help` prints this list. Add `--verbose` to anything to see full tracebacks;
without it, expected failures print a plain sentence instead.

```bash
python3 make_video.py studio         # open the control panel (same as Start.bat)
python3 make_video.py studio --no-browser    # ...without opening a browser

python3 make_video.py doctor         # check this machine — run this first when stuck
python3 make_video.py list           # projects found in sheets/
python3 make_video.py models         # which Gemini models your key can call
python3 make_video.py voices --lang de
python3 make_video.py benchtts --lang en

python3 make_video.py generate --script ~/Desktop/script06.txt --id video06 --langs en,de,es
python3 make_video.py stock  --sheet sheets/video06_MASTER_production_sheet.md
python3 make_video.py voice  --sheet sheets/video06_MASTER_production_sheet.md --lang de
python3 make_video.py render --sheet sheets/video06_MASTER_production_sheet.md --lang de --captions
```

Useful flags: `--music music/piano.mp3`, `--no-zoom` (faster), `--redo 12,45,78`,
`--voice de-DE-KillianNeural`, `--overwrite`.

---

# 5 · How it works

## Your sheet is the video

Every scene is three facts: what's said, what to show, and whether that's a photo or a
clip. Everything downstream is mechanical translation of those rows.

```
**S1 ⬜** · IMAGE
- Narration: "You're wide awake, and it's still dark."
- ALT / search: `senior lying awake in a dark bedroom, eyes open`
```

Drop new sheets into `sheets/`; they appear in the dropdown. Translation files named like
`video06_GERMAN_narration.md` are detected automatically.

## Nothing is ever wasted

Everything expensive is cached by *content*, not filename.

- Stock is keyed by search query + media type + which take
- Narration is keyed by a hash of the exact text + voice + speed + pitch — change one
  word in scene 47 and only scene 47 regenerates
- Scene clips and the crossfaded track survive too

**Visuals are shared across languages.** German and Spanish reuse the English pictures,
which is what you want and makes them much faster.

## Timing

Two numbers at the top of `lib/pipeline.py` control the feel:

```python
TAIL     = 1.0   # seconds the picture holds after the line ends
DISSOLVE = 0.6   # crossfade length between scenes
```

That leaves a 0.4s breath between lines, with every dissolve landing in silence so no word
is faded away. Narration is locked to each clip's *actual* frame-rounded duration —
without that, rounding error slides the voice about a second off the picture across 115
scenes.

---

# 6 · When something goes wrong

**Always start here:**

```bash
python3 make_video.py doctor
```

### `externally-managed-environment` when using pip

Expected — Homebrew's Python blocks system installs. Use `bash setup.sh` (or
`bash setup.sh`) instead of pip directly. Never use `--break-system-packages`;
it can break Homebrew itself.

### doctor says `!! environment  system Python`

Run `bash setup.sh` to create the `.venv`. If it already exists, the scripts switch to it
automatically — you can ignore this when everything else is `ok`.

### Captions aren't burned in

doctor will show why:

```
!! build libass   NOT compiled in
!! captions       none: NO subtitle filter at all
```

Your ffmpeg was built without text rendering. Either reinstall the standard formula:

```bash
brew uninstall --ignore-dependencies ffmpeg
brew install ffmpeg
python3 make_video.py doctor
```

**Or just don't burn them in.** Every render writes a `.srt` next to the MP4, and
uploading that to YouTube is arguably better — viewers can toggle it, YouTube can
auto-translate it, and the text gets indexed for search. Burned-in captions do none of
that.

A caption failure never loses a render. The video is finished before captions run, so it
gets saved either way and the problem is reported as a caveat.

### A scene says "no match found"

The search query is too abstract. Stock libraries index literal, photographic
descriptions.

- Works: `alarm clock glowing in a dark bedroom at night, moody`
- Fails: `the passage of time` → returns clip-art

Edit the `ALT / search:` line in your sheet and click Find visuals again.

### Narration stops partway

A network hiccup. Run it again — finished lines are cached, so it resumes.

### Gemini says a model is unavailable

Google retires model names at short notice. The pipeline detects this and switches to a
live model automatically. To see what your key can reach:

```bash
python3 make_video.py models
```

Leave `"gemini_model": "auto"` in `config.json` unless you want to pin one.

### Render is slow

Normal — it's encoding 115 clips. Untick "Slow zoom on photos" to roughly halve it.
Closing the window loses nothing; finished clips are kept and the crossfade is reused.

### Port 8765 already in use

The studio is already running in another window, or didn't shut down cleanly. Close the
other Terminal window, or change `PORT` at the top of `studio.py`.

### Start.command does nothing when double-clicked *(macOS)*

It lost its executable bit. Once, in Terminal:

```bash
chmod +x Start.command setup.sh
```

### `'python' is not recognized` *(Windows)*

Python isn't on your PATH. Re-run the Python installer, choose **Modify**, and make sure
**"Add python.exe to PATH"** is ticked. Then close and re-open PowerShell.

### doctor says `!! chatterbox installed · CPU` on a machine with an NVIDIA card

**This is the most likely thing to go wrong, and it is silent.** Nothing errors — voicing
just runs roughly twenty times slower than it should.

The cause is that `chatterbox-tts` hard-pins `torch==2.6.0`. When pip honours that pin it
replaces the CUDA build with the CPU-only build from PyPI. `setup.bat` installs Chatterbox
*first* and the CUDA build *second* to work around this, but any later `pip install` that
re-resolves dependencies can undo it again.

Overriding the pin is not optional on an RTX 50-series card: torch 2.6 predates Blackwell
support, so no build of it can drive one at all. Chatterbox runs fine on newer torch — the
pin is simply over-strict.

Fix it:

```powershell
cd $HOME\Documents\faceless-pipeline
.venv\Scripts\python.exe -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

pip will print a dependency-conflict warning about `chatterbox-tts requires torch==2.6.0`.
**Ignore it** — that is the pin being overridden on purpose.

Then `python make_video.py doctor` again — it should name your card and its VRAM.

> **Habit worth having:** run `doctor` after any `pip install` in this project. It is the
> only thing that catches this, because nothing else complains.

### `no kernel image is available for execution on the device`

The build has CUDA but not for *your* card, which means the card is newer than the stable
build supports. Use the nightly, which picks up new GPUs first:

```powershell
.venv\Scripts\python.exe -m pip install --pre --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### The whole computer freezes while voicing *(macOS)*

The GPU shares system memory with everything else, and Chatterbox exhausted it. Force the
CPU — slower per line, but it won't take the machine down with it:

```bash
FACELESS_DEVICE=cpu python3 make_video.py voice --sheet … --lang en
```

This is the single best reason to move voicing to a machine with a discrete NVIDIA card.

---

# 7 · What's in the folder

```
Start.bat           Windows — double-click this (same as: make_video.py studio)
Start.command       macOS — double-click this   (same as: make_video.py studio)
setup.bat           Windows — one-time setup
setup.sh            macOS/Linux — one-time Python setup
install.sh          macOS — one-time system setup (Homebrew, ffmpeg)
tools/              shared install logic used by both setup scripts
config.json         your API keys
voices.json         chosen voice per language

studio.py           the control panel
make_video.py       command-line equivalent
lib/                the pipeline itself
lib/console.py      keeps output readable on Windows consoles
lib/ui.html         the whole interface, one file, no build step
tools/check_ui.mjs  renders every view headlessly to catch breakage
cli.py              the `faceless` command
pyproject.toml      declares that command (deliberately no dependencies)

sheets/             master sheets + translation files
voices_refs/        reference clips for voice cloning
music/              background tracks
cache/stock/        downloaded photos and clips (shared across languages)
cache/voice/        generated narration (one file per line)
cache/previews/     voice audition samples
work/               intermediate clips — safe to delete
out/                finished MP4s and .srt files
```

Deleting `work/` and `cache/` is always safe — it just means the next run redoes that
work. Deleting `.venv` is safe too; re-run `bash setup.sh`.

---

# 8 · Honest limits

- **Voicing is slow without a GPU.** Chatterbox runs on your own machine rather than a
  datacentre — fast on an NVIDIA card, slow on anything else. Run
  `benchtts` to see what that means in minutes before planning around it.
- **The reference clip is your responsibility.** Whatever you put in `voices_refs/`
  becomes the channel's voice, so it needs to be yours or clearly cleared.
- **Auto-picked stock is roughly 70% as good as hand-picked.** The review step exists to
  close that gap; the gold borders tell you where to spend attention.
- **Gemini-written sheets need spot-checking**, especially the `ALT / search:` queries.
  Check the first few closely, then skim.
- **Pexels allows 200 requests/hour.** A fresh 115-scene video uses ~115. Re-runs and
  other languages are free — everything is cached.
- **The studio is local only.** It listens on 127.0.0.1. Nothing is exposed to your
  network; nothing is uploaded anywhere except the stock searches and the text sent for
  voicing.
