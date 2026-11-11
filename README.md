# doc-to-audiobook

Turn a whole book or document into a natural-sounding audiobook (MP3) on your
own machine. Text PDFs, scanned PDFs (with OCR), EPUB, DOCX, Markdown, HTML and
plain text all go in; one command gives you a narrated, loudness-levelled MP3.

```bash
audiobook.cmd "book.pdf"                    # Windows
./audiobook.sh "book.pdf"                   # Linux / macOS
```

- **Any document.** It detects text vs. scanned PDFs on its own. Scans of an
  open book are split into single pages, straightened and OCR'd, and running
  headers, page numbers and scanner watermarks are removed. Footnotes are moved
  to the end so they don't break up sentences.
- **Two voice engines.** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
  is fast, English only and has 13 fixed voices.
  [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) speaks 600+ languages,
  can design a voice from a description, and can clone one from a short
  recording.
- **Books of any length.** Audio goes straight into ffmpeg as it is produced, so
  a 30-hour book never sits in memory and never runs into the 4 GB WAV limit.
  EPUBs can be split into one resumable MP3 per chapter.
- **Sentence-accurate timings.** A `timings.json` next to the MP3 records where
  each paragraph starts, so a player can show the text and jump to it on a tap.
- **One installer for the hard part.** It sets up CUDA, torch, PaddleOCR,
  Tesseract and four isolated Python environments on Windows and Linux.

There is a phone-friendly web player and Telegram bot built on top of this:
[**audiobook-web**](https://github.com/emdaitaj/audiobook-web).

---

## Install

Clone the repository, then run the installer for your system.

**Windows.** Double-click `install.cmd`, or from a terminal:

```powershell
git clone https://github.com/emdaitaj/doc-to-audiobook
cd doc-to-audiobook
.\install.cmd
```

**Linux / macOS:**

```bash
git clone https://github.com/emdaitaj/doc-to-audiobook
cd doc-to-audiobook
./install.sh
```

You don't need to install Python first, and you don't need admin rights for the
Python side.

### What the installer does

1. Gets [uv](https://docs.astral.sh/uv/), a fast Python and package manager,
   installed for your user only.
2. Looks for an NVIDIA GPU and reads which CUDA version your driver supports.
3. For each stage below, finds the **newest Python that has ready-made
   binary wheels for every package**. It does this by asking uv to resolve the
   packages for Python 3.14, 3.13 and so on down to 3.9, together with the
   newest CUDA build of PyTorch your driver can run. Packages that would need
   compiling (torch, numpy, paddle, OpenCV, spaCy and others) are never built
   from source, so a Python version without wheels for them counts as
   unsupported, not as a three-hour compile.
4. Downloads that Python if you don't have it, creates the env, installs, and
   **checks the env really works**: it imports everything and runs a small
   calculation on the GPU. If the check fails, it rebuilds with the next option
   (an older Python, or an older CUDA build) on its own.
5. Checks for Tesseract (and the Visual C++ runtime on Windows, and libGL on
   Linux) and offers to install whatever is missing (`winget`, `apt`, `dnf`,
   `pacman`, `zypper` or `brew`). Then it downloads Tesseract's best English
   model.
6. Writes the `audiobook.cmd` / `audiobook.sh` launchers.

The stages each get their own virtualenv under `envs/`, because their
dependencies conflict. PaddleOCR 2.x needs NumPy 1.x, while the voice engines
want NumPy 2 and a CUDA build of torch:

| env | what's in it | what it does |
| --- | --- | --- |
| `core` | PyMuPDF, OpenCV | runs the converter, reads PDFs, deskews scans |
| `kokoro` | Kokoro-82M, torch (CUDA), spaCy | fast English narration |
| `omnivoice` | OmniVoice, torch (CUDA), torchaudio | multilingual narration, voice design, cloning |
| `paddle` | PaddlePaddle 2.6 + PaddleOCR 2.x (CPU) | OCR for scanned PDFs |

For example, on a Windows 11 laptop with an RTX 3060 in November 2026, `core`,
`kokoro` and `omnivoice` resolved to Python 3.14 (the voice engines with
torch+cu132), and `paddle` resolved to Python 3.12, because PaddlePaddle 2.6
has no wheels for 3.13 or newer. Run `--dry-run` to see what your machine gets.

### Installer options

| option | does |
| --- | --- |
| `--only core,kokoro` | install just these envs |
| `--skip omnivoice` | everything except these (OmniVoice is the big one, ~6 GB with its model) |
| `--dry-run` | resolve and print the plan, install nothing |
| `--check` | test the envs that are already there |
| `--cuda cpu` | force a torch build: `cpu`, `cu118` … `cu132`; the default `auto` picks the newest one your driver supports |
| `--python 3.12` | force one Python version |
| `--recreate` | rebuild envs even if they already work (by default, working envs are kept) |
| `--models` | download the Kokoro and OmniVoice weights now instead of on first use |
| `--yes` | answer yes to every question (for unattended installs) |
| `--verbose` | show why each Python version was rejected |

On Windows pass them through the wrapper, e.g. `install.cmd --skip omnivoice --yes`.

**Disk space:** about 3 GB per torch env, plus 0.3 GB for the Kokoro model and
3.3 GB for the OmniVoice model. uv's cache can be emptied afterwards with
`uv cache clean`.

**No NVIDIA GPU?** Everything still works on the CPU. Kokoro is about 10× slower
on CPU and OmniVoice is much slower still.

---

## Usage

```bash
audiobook "book.pdf" --transcript-only    # 1. extract + clean the text, then stop
# read <book>_audiobook/transcript.txt and fix anything that would read badly
audiobook "book.pdf"                      # 2. narrate it (reuses the transcript)
```

(`audiobook` here means `audiobook.cmd` on Windows and `./audiobook.sh`
elsewhere. Both run `pdf_to_audiobook.py` with the `core` env.)

Cleaning the transcript is what makes the audiobook sound good. OCR mistakes
get read out as nonsense, so look at `transcript.txt` before a long run.

```bash
audiobook "book.epub" --list                         # sections with word counts
audiobook "book.epub" --split                        # one MP3 per chapter, resumable
audiobook "book.pdf" --pages 3-40 --voice af_heart --speed 1.1
audiobook "scan.pdf" --layout single --ocr tesseract
audiobook "notes.md" --read-code                     # also narrate code blocks
audiobook "book.pdf" --engine omnivoice --voice "female, elderly, moderate pitch, british accent"
audiobook "roman.pdf" --engine omnivoice --lang de
audiobook "book.pdf" --engine omnivoice --ref-audio me.wav --ref-text "exactly what the clip says"
audiobook --engine omnivoice --list-voices
```

### Options

| option | does |
| --- | --- |
| `--transcript-only` | write `transcript.txt` and stop |
| `--pages 3-40` | PDF page numbers, or EPUB section numbers from `--list` (0-based) |
| `--list` | EPUB: print the sections and exit |
| `--split` | EPUB: one MP3 per section; a rerun skips sections already done |
| `--engine` | `kokoro` (default) or `omnivoice` |
| `--voice` | Kokoro voice name, or OmniVoice attributes (see below) |
| `--speed` | Kokoro speed, `0.9` slower, `1.1` faster |
| `--lang` | OmniVoice language, a code or a name (`de`, `German`) |
| `--ref-audio` / `--ref-text` | OmniVoice: clone this voice (and its exact transcript) |
| `--wpm` | OmniVoice pace in words per minute, default 165 |
| `--steps` | OmniVoice diffusion steps, default 64 |
| `--seed` / `--anchor` | OmniVoice: keep the same narrator across runs |
| `--layout` | scanned PDF: `auto`, `single` or `side-by-side` |
| `--ocr` | `paddle` (default) or `tesseract` |
| `--force-scanned` | OCR a PDF even if it has a text layer |
| `--read-code` | Markdown: narrate fenced code blocks |
| `--outdir` | output folder, default `<file>_audiobook` |

If the file name starts with a dash, put the options first and use `--`:
`audiobook --pages 0-9 -- "-weird.pdf"`.

### What comes out

```
book_audiobook/
  book.mp3            96 kbps mono, loudness-normalised (about 40 MB per hour)
  transcript.txt      exactly what was read, one paragraph per blank line
  timings.json        {"sample_rate": 24000, "offsets": [...]}  start of each paragraph
  split/, ocr.json    scanned PDFs only: page images and raw OCR
  voice_anchor.wav    OmniVoice only: the narrator, listen to this first
  voice_prompt.omni   OmniVoice only: cached narrator, reused on reruns
```

While narration is running the MP3 is written as `book.mp3.part`. It is
constant bitrate, so it can be played and seeked while it grows.

---

## Supported formats

| format | how it is read |
| --- | --- |
| **PDF, text** | text layer, with hyphenation joined, page numbers removed, abbreviations expanded (`pp.` → "pages", `vs.` → "versus") |
| **PDF, scanned** | rendered at high DPI, two-page spreads split at the gutter, each page deskewed, OCR'd with PaddleOCR (Tesseract as fallback), headers, page numbers and watermarks dropped, footnotes moved to the end |
| **EPUB** | follows the spine (real reading order), skips the table of contents and `linear="no"` items such as covers |
| **Markdown** | headings, list items and table rows become their own paragraphs, links keep their text, URLs, images and front matter are dropped, code blocks are skipped unless `--read-code` |
| **HTML** | `script`, `style` and `nav` removed, headings kept as their own paragraph |
| **DOCX** | paragraph text from the document (footnotes and endnotes are not included) |
| **anything else** | read as plain text |

`.doc`, `.rtf` and `.odt` aren't supported. Re-save them as `.docx`, `.pdf` or `.txt`.

---

## Choosing a voice engine

| | Kokoro (default) | OmniVoice |
| --- | --- | --- |
| model | Kokoro-82M | k2-fsa OmniVoice 0.6B |
| languages | English | 600+ |
| voice | 13 fixed voices | designed from attributes, or cloned |
| speed on an RTX 3060 | ~24× realtime | ~2× realtime at 64 steps |
| VRAM / download | tiny / 0.3 GB | ~1.5–2 GB / 3.3 GB |
| model licence | Apache-2.0 | **CC-BY-NC-4.0, no commercial use** |

**Kokoro voices:** `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `am_michael`
(default), `am_adam`, `am_eric`, `am_liam`, `am_onyx`, `bf_emma`, `bf_isabella`,
`bm_george`, `bm_lewis`. In each name the first letter is the accent (a =
American, b = British) and the second is f/m.

### OmniVoice voices

`--voice` is **not free text**. It is a comma-separated list from a fixed
vocabulary, at most one item per category:

| category | options |
| --- | --- |
| gender | `female`, `male` |
| age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| accent | `american`, `australian`, `british`, `canadian`, `chinese`, `indian`, `japanese`, `korean`, `portuguese`, `russian` (written as `... accent`) |
| effect | `whisper` |

The default is `male, low pitch, middle-aged, american accent`. Anything outside
the list is rejected before the model loads, so a typo costs you a second, not a
minute.

**One narrator for the whole book.** OmniVoice is a diffusion model, so it
would design a slightly different voice for every paragraph. The converter
renders the description **once** to `voice_anchor.wav`, caches it as
`voice_prompt.omni`, and clones every paragraph from that clip. Delete the
`.omni` file to re-roll the voice, or point `--anchor` at a clip to use the same
narrator for another book. With `--ref-audio` your own clip is cloned instead.

**Tuning, from measurements:**

- `--steps 64` is the default. Below about 64 the delivery flattens: stresses
  land in the wrong place and word transitions get abrupt. Time grows linearly
  with steps (32 steps ≈ 3.9× realtime, 64 ≈ 2×, 96 ≈ 1.3×).
- Keep `--batch 1`. OmniVoice already uses the whole GPU on one paragraph:
  batch 4 measured 4% slower for 40% more VRAM.
- `--wpm 165` slows the model down from its natural ~225 wpm, which is podcast
  pace. Below ~170 it stops following the request, so treat it as a nudge.
- Numbers are spelled out with `num2words`. Full text normalisation
  (`omnivoice[tn]`) needs `pynini`, which does not build on Windows.

---

## Configuration

The converter finds its tools next to itself. Override any of them with
environment variables:

| variable | default |
| --- | --- |
| `AUDIOBOOK_ENVS` | `./envs` |
| `KOKORO_PY` | `envs/kokoro/…/python` |
| `OMNIVOICE_PY` | `envs/omnivoice/…/python` |
| `PADDLE_PY` | `envs/paddle/…/python` |
| `TESSERACT` | `tesseract` on PATH, or `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `TESSDATA_BEST` | `./tessdata_best` (if it holds `eng.traineddata`) |
| `DESKEW_PY` | `tools/book_deskew_split.py` |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` |

`tools/book_deskew_split.py` also works on its own for scanned books:

```bash
envs/core/bin/python tools/book_deskew_split.py scan.pdf -o clean.pdf --layout side-by-side --crop
```

### Without the installer

If you'd rather build the envs yourself, create these four virtualenvs and set
the variables above:

```bash
# core      - any Python 3.9+
pip install pymupdf opencv-python numpy pillow
# kokoro    - torch and torchvision from the SAME index, in ONE command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install kokoro soundfile imageio-ffmpeg num2words
python -m spacy download en_core_web_sm
# omnivoice
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install omnivoice soundfile imageio-ffmpeg num2words
# paddle    - Python 3.12 or older
pip install "paddlepaddle>=2.6,<3" "paddleocr>=2.6,<3" "numpy<2"
```

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| `operator torchvision::nms does not exist` | torch and torchvision don't match. Never `pip install torch` on its own in the kokoro env; rerun `install.py --only kokoro --recreate` |
| `OSError: [WinError 126] ... c10.dll` | the Visual C++ runtime is missing; the installer offers it, or get [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| narration says `Device cpu` on a GPU machine | `install.py --check`; if cuda shows False, update the NVIDIA driver or close other GPU programs |
| `module 'numpy' has no attribute 'int'` | PaddleOCR 2.x on new NumPy; the converter patches this, and the paddle env pins `numpy<2` |
| `libGL.so.1: cannot open shared object file` | Linux: `sudo apt install libgl1 libglib2.0-0` (the installer offers it) |
| a few OCR pages are garbled | usually pages near the binding. Fix them by hand in `transcript.txt` and rerun |

---

## License

Copyright © 2026 [emdaitaj](https://github.com/emdaitaj).

Licensed under the **GNU Affero General Public License v3.0** ([LICENSE](LICENSE)),
with additional attribution terms under section 7 ([NOTICE](NOTICE)). In short:

- **Credit the source.** Any copy, fork or derived work, including a website or
  service built on it, must visibly credit and link to
  <https://github.com/emdaitaj>, e.g.
  *"Based on doc-to-audiobook by emdaitaj — https://github.com/emdaitaj"*.
- **Share your changes.** If you distribute a modified version, or run one as a
  network service that others use, you must publish its full source code
  under the same licence.
- Modified versions must be marked as modified.

The models and tools this downloads have their own licences. In particular,
the **OmniVoice weights are CC-BY-NC-4.0 (non-commercial)**. Kokoro, PaddleOCR
and Tesseract are Apache-2.0.
