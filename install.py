#!/usr/bin/env python3
# Part of doc-to-audiobook - https://github.com/emdaitaj/doc-to-audiobook
# Copyright (C) 2026 emdaitaj. AGPL-3.0 with attribution terms (see NOTICE).
"""
install.py - set up every tool doc-to-audiobook needs, on Windows, Linux or macOS.

Normally started through install.cmd / install.ps1 (Windows) or install.sh
(Linux/macOS), which fetch `uv` first. Run directly it needs Python 3.8+ and uv.

The stages cannot share a single interpreter, so each gets its own virtualenv
under envs/:

    core       PyMuPDF + OpenCV       runs the converter, reads PDFs, deskews scans
    kokoro     Kokoro-82M + torch     fast English narration (GPU if available)
    omnivoice  OmniVoice + torch      600+ languages, voice design and cloning
    paddle     PaddleOCR 2.x (CPU)    OCR for scanned PDFs (needs NumPy 1.x)

For each one the script works out the newest Python that has binary wheels for
every package (asking uv to resolve against 3.14, 3.13, ... down to 3.9, and
the best CUDA build of torch your driver supports), downloads that Python if
you do not have it, builds the env, and checks it by importing everything. If
the check fails it tries the next Python down.

It also checks for Tesseract (backup OCR) and the Visual C++ runtime on
Windows, and offers to install them, and downloads Tesseract's best English model.

    python install.py                      everything
    python install.py --only core,kokoro   just these envs
    python install.py --skip omnivoice     everything but this
    python install.py --dry-run            show what would be installed, change nothing
    python install.py --check              test the envs that are already there
    python install.py --cuda cpu           force CPU torch (or cu118/cu124/cu126/cu128)
    python install.py --python 3.11        force one Python version for every env
    python install.py --recreate           rebuild envs even if they already work
    python install.py --models             also pre-download the voice models (~3.7 GB)
    python install.py --yes                answer yes to every question
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = os.environ.get("AUDIOBOOK_ENVS", os.path.join(HERE, "envs"))
TESSDATA_BEST = os.path.join(HERE, "tessdata_best")
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
PY_CANDIDATES = ["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"]

# Packages that must come as a wheel. Building any of them from source is
# what makes these installs miserable (no compiler, no CUDA toolkit, hours of
# build), so a Python version without wheels for them is treated as unsupported.
NO_BUILD = [
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "pillow",
    "paddlepaddle", "opencv-python", "opencv-contrib-python",
    "opencv-python-headless", "opencv-contrib-python-headless", "pymupdf",
    "shapely", "lmdb", "pyclipper", "scikit-image", "lxml", "numba",
    "llvmlite", "spacy", "thinc", "blis", "cymem", "preshed", "murmurhash",
    "srsly", "tokenizers", "safetensors", "sentencepiece", "pynini",
    "soundfile", "pydantic-core", "rapidfuzz", "regex",
]

SPACY_EN = ("en_core_web_sm @ https://github.com/explosion/spacy-models/releases/"
            "download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")


class TorchBuild:
    """One PyTorch wheel flavour (a CUDA version, or CPU).

    uv's --torch-backend sends torch/torchvision/torchaudio to the matching
    download.pytorch.org index and everything else to PyPI. Because the three
    are resolved together, and torchvision/torchaudio pin their exact torch,
    they always come as a matched set - installing torch on its own later is
    what causes "operator torchvision::nms does not exist".
    """

    def __init__(self, tag, min_cuda):
        self.tag, self.min_cuda = tag, min_cuda

    def index_args(self):
        return [] if self.tag == "pypi" else ["--torch-backend", self.tag]

    @property
    def gpu(self):
        return self.tag.startswith("cu")

    def __str__(self):
        return "torch (PyPI)" if self.tag == "pypi" else f"torch+{self.tag}"


# Newest first; the first one your driver supports that resolves is used.
TORCH_BUILDS = [
    TorchBuild("cu132", (13, 2)),
    TorchBuild("cu130", (13, 0)),
    TorchBuild("cu129", (12, 9)),
    TorchBuild("cu128", (12, 8)),
    TorchBuild("cu126", (12, 6)),
    TorchBuild("cu124", (12, 4)),
    TorchBuild("cu118", (11, 8)),
    TorchBuild("cpu", None),
]
TORCH_PYPI = TorchBuild("pypi", None)   # macOS: PyPI wheels already do MPS/CPU

# Runs inside a torch env. A real kernel on the GPU, not just is_available():
# a new CUDA build on an old card reports True and then fails on first use.
TORCH_CHECK = (
    "import torch\n"
    "ok = torch.cuda.is_available()\n"
    "if ok:\n"
    "    try:\n"
    "        assert (torch.ones(4, device='cuda') * 2).sum().item() == 8\n"
    "    except Exception as e:\n"
    "        ok = False; print('GPU kernel failed:', e)\n"
    "print('torch', torch.__version__, '| cuda', ok)\n"
)


COMPONENTS = {
    "core": dict(
        about="PyMuPDF + OpenCV: runs the converter, reads PDFs, deskews scans",
        reqs=["pymupdf", "numpy", "pillow",
              "opencv-python" if IS_WIN or IS_MAC else "opencv-python-headless"],
        torch=None,
        smoke=("import fitz, cv2, numpy, PIL; "
               "print('PyMuPDF', fitz.VersionBind, '| OpenCV', cv2.__version__)"),
    ),
    "kokoro": dict(
        about="Kokoro-82M: fast English narration",
        reqs=["kokoro>=0.9.4", "soundfile", "imageio-ffmpeg", "num2words", SPACY_EN],
        torch=["torch>=2.6", "torchvision"],
        smoke=("import kokoro, soundfile, imageio_ffmpeg, spacy\n"
               "spacy.load('en_core_web_sm')\n" + TORCH_CHECK),
    ),
    "omnivoice": dict(
        about="OmniVoice: 600+ languages, voice design and cloning",
        reqs=["omnivoice>=0.2.1", "soundfile", "imageio-ffmpeg", "num2words"],
        torch=["torch>=2.8", "torchaudio"],
        smoke=("import torchaudio, soundfile, imageio_ffmpeg\n"
               "from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt\n"
               + TORCH_CHECK),
    ),
    "paddle": dict(
        about="PaddleOCR 2.x on CPU: OCR for scanned PDFs",
        # The converter uses the PaddleOCR 2.x API; 3.x changed it completely.
        # Paddle 2.6 is built against NumPy 1.x.
        reqs=["paddlepaddle>=2.6,<3", "paddleocr>=2.6,<3", "numpy<2", "pillow",
              "setuptools"],
        torch=None,
        # Building the reader also downloads its small detection/recognition
        # models, so a passing check means the first real run will work.
        smoke=("import numpy as np; np.int=int; np.float=float; np.bool=bool; "
               "np.object=object\nimport paddle\nfrom paddleocr import PaddleOCR\n"
               "PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)\n"
               "print('paddle', paddle.__version__)"),
    ),
}

MODEL_PREFETCH = {
    "kokoro": "from huggingface_hub import snapshot_download; "
              "print(snapshot_download('hexgrad/Kokoro-82M'))",
    "omnivoice": "import os; from huggingface_hub import snapshot_download; "
                 "print(snapshot_download(os.environ.get('OMNIVOICE_MODEL', 'k2-fsa/OmniVoice')))",
}


# ---------------------------------------------------------------------------
#  small helpers
# ---------------------------------------------------------------------------
def say(msg=""):
    print(msg, flush=True)


def head(msg):
    say()
    say("== " + msg + " " + "=" * max(0, 70 - len(msg)))


def ask(question, args):
    if args.yes:
        say(f"{question} [Y/n] y  (--yes)")
        return True
    if not sys.stdin or not sys.stdin.isatty():
        say(f"{question} -> skipped (not interactive; rerun with --yes)")
        return False
    return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def run(cmd, check=True, capture=False, **kw):
    kw.setdefault("env", {**os.environ, "PYTHONIOENCODING": "utf-8"})
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                  encoding="utf-8", errors="replace")
    return subprocess.run(cmd, check=check, **kw)


def venv_python(env_dir):
    for rel in (("Scripts", "python.exe"), ("python.exe",), ("bin", "python")):
        p = os.path.join(env_dir, *rel)
        if os.path.exists(p):
            return p
    return os.path.join(env_dir, "Scripts" if IS_WIN else "bin",
                        "python.exe" if IS_WIN else "python")


def find_uv():
    for cand in (os.environ.get("UV"), shutil.which("uv"),
                 os.path.expanduser("~/.local/bin/uv" + (".exe" if IS_WIN else "")),
                 os.path.expanduser("~/.cargo/bin/uv" + (".exe" if IS_WIN else ""))):
        if cand and os.path.exists(cand):
            return cand
    return None


def is_link(path):
    isjunction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(isjunction and isjunction(path))


def remove_env(path):
    """Delete an env folder - but only unlink a symlink/junction, never follow it."""
    if not os.path.lexists(path):
        return
    if is_link(path):
        try:
            os.unlink(path)
        except OSError:
            os.rmdir(path)          # Windows junction
        return
    shutil.rmtree(path)


def last_error_line(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for l in lines:                       # uv's own one-line explanation if any
        if "because" in l or "requires" in l or "no wheels" in l.lower():
            return l[:200]
    return lines[-1][:200] if lines else "no details"


# ---------------------------------------------------------------------------
#  machine detection
# ---------------------------------------------------------------------------
def detect_gpu():
    """(gpu name, driver CUDA version tuple) or (None, None)."""
    cands = [shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi",
             r"C:\Windows\System32\nvidia-smi.exe"]
    smi = next((c for c in cands if c and os.path.exists(c)), None)
    if not smi:
        return None, None
    try:
        out = run([smi], capture=True, check=False, timeout=30).stdout or ""
        names = run([smi, "--query-gpu=name", "--format=csv,noheader"],
                    capture=True, check=False, timeout=30).stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer.
    m = re.search(r"CUDA(?: UMD| Driver)? Version:\s*(\d+)\.(\d+)", out)
    name = names.strip().splitlines()[0] if names.strip() else "NVIDIA GPU"
    return name, ((int(m.group(1)), int(m.group(2))) if m else None)


def torch_builds_for(cuda, forced):
    if IS_MAC:
        return [TORCH_PYPI]
    if forced and forced != "auto":
        match = [b for b in TORCH_BUILDS if b.tag == forced]
        if not match:
            sys.exit(f"--cuda must be one of: auto, " + ", ".join(b.tag for b in TORCH_BUILDS))
        return match
    usable = [b for b in TORCH_BUILDS if b.min_cuda and cuda and cuda >= b.min_cuda]
    return usable + [b for b in TORCH_BUILDS if b.tag == "cpu"]


def platform_args(args):
    return ["--python-platform", args.platform] if args.platform else []


# ---------------------------------------------------------------------------
#  resolve -> pick python + torch build
# ---------------------------------------------------------------------------
def requirements(comp, build):
    reqs = list(comp["reqs"])
    if comp.get("torch"):
        reqs = list(comp["torch"]) + reqs
    return reqs


def resolve(uv, reqs, py, build, args):
    """Ask uv whether `reqs` has a binary-only solution for Python `py`."""
    with tempfile.TemporaryDirectory() as tmp:
        req_file = os.path.join(tmp, "requirements.in")
        with open(req_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(reqs) + "\n")
        cmd = [uv, "pip", "compile", req_file, "--python-version", py,
               "--output-file", os.path.join(tmp, "out.txt"), "--quiet",
               "--no-header", *platform_args(args)]
        cmd += build.index_args() if build else []
        for pkg in NO_BUILD:
            cmd += ["--only-binary", pkg]
        r = run(cmd, capture=True, check=False)
        if r.returncode != 0 and NETWORK_ERROR.search(r.stderr or ""):
            sys.exit("\n!! Could not reach the package index (network error):\n   "
                     + last_error_line(r.stderr)
                     + "\n   Check your internet connection / proxy / VPN and rerun.")
        return r.returncode == 0, last_error_line(r.stderr)


# uv's wording when the index could not be reached, as opposed to "no solution".
NETWORK_ERROR = re.compile(r"Request failed after|Failed to fetch|error sending request|"
                           r"dns error|operation timed out|Connection refused", re.I)


def plan_component(uv, name, comp, builds, args, skip_builds=()):
    """Yield (python, torch build) pairs that resolve, best first.

    A GPU build beats a newer Python, so builds are the outer loop. Within a
    build the two newest Pythons that resolve are offered. Lazy, because each
    probe is a network round trip and usually the first answer is used.
    """
    pythons = [args.python] if args.python else PY_CANDIDATES
    for build in (builds if comp.get("torch") else [None]):
        offered = 0
        for py in pythons:
            if build in skip_builds:
                break
            ok, why = resolve(uv, requirements(comp, build), py, build, args)
            label = f"Python {py}" + (f" + {build}" if build else "")
            if ok:
                say(f"  [ok]   {label}")
                yield py, build
                offered += 1
                if offered == 2:
                    break
            elif args.verbose:
                say(f"  [no]   {label}: {why}")


# ---------------------------------------------------------------------------
#  build + verify
# ---------------------------------------------------------------------------
# A check that failed because a model download broke off, not because the env
# is wrong. Retried in place instead of rebuilding with another Python.
FLAKY_DOWNLOAD = re.compile(r"ChunkedEncodingError|IncompleteRead|ConnectionError|"
                            r"ConnectTimeout|ReadTimeout|Connection (?:broken|reset|aborted)|"
                            r"timed out|Temporary failure in name resolution", re.I)


def smoke(env_dir, comp, attempts=3):
    py = venv_python(env_dir)
    if not os.path.exists(py):
        return False, "no interpreter"
    for attempt in range(1, attempts + 1):
        r = run([py, "-c", comp["smoke"]], capture=True, check=False)
        if r.returncode == 0:
            lines = (r.stdout or "").strip().splitlines()
            return True, lines[-1] if lines else "ok"
        why = last_error_line(r.stderr)
        if attempt < attempts and FLAKY_DOWNLOAD.search(r.stderr or ""):
            say(f"  download interrupted ({why[:80]}); retrying the check ({attempt}/{attempts - 1})")
            continue
        return False, why
    return False, why


def build_env(uv, name, comp, py, build, args):
    env_dir = os.path.join(ENVS, name)
    remove_env(env_dir)
    say(f"  creating {name} env with Python {py} (uv downloads it if needed)")
    run([uv, "venv", env_dir, "--python", py, "--seed", "--quiet"])
    reqs = requirements(comp, build)
    with tempfile.TemporaryDirectory() as tmp:
        req_file = os.path.join(tmp, "requirements.txt")
        with open(req_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(reqs) + "\n")
        cmd = [uv, "pip", "install", "--python", venv_python(env_dir), "-r", req_file]
        cmd += build.index_args() if build else []
        for pkg in NO_BUILD:
            cmd += ["--only-binary", pkg]
        say(f"  installing: {', '.join(r.split(' @ ')[0] for r in reqs)}")
        run(cmd)
    return env_dir


def install_component(uv, name, builds, args, report):
    comp = COMPONENTS[name]
    head(f"{name}: {comp['about']}")
    env_dir = os.path.join(ENVS, name)
    if os.path.lexists(env_dir) and not args.recreate:
        ok, info = smoke(env_dir, comp)
        if ok:
            say(f"  already installed and working ({info}) - use --recreate to rebuild")
            report[name] = {"status": "ok", "info": info, "kept": True}
            return True
        say(f"  existing env is broken ({info}); rebuilding")

    say("  finding the newest Python with wheels for everything ...")
    bad_builds = set()            # CUDA builds whose GPU check failed
    tried = 0
    for py, build in plan_component(uv, name, comp, builds, args, bad_builds):
        tried += 1
        if args.dry_run:
            say(f"  would install: Python {py}" + (f" + {build}" if build else ""))
            report[name] = {"status": "planned", "python": py,
                            "torch": str(build) if build else None}
            return True
        try:
            build_env(uv, name, comp, py, build, args)
        except subprocess.CalledProcessError as e:
            say(f"  install failed on Python {py} ({e}); trying the next option")
            continue
        ok, info = smoke(env_dir, comp)
        if ok and build and build.gpu and "cuda False" in info:
            say(f"  {build} installed but the GPU check failed ({info}); "
                "trying an older CUDA build")
            bad_builds.add(build)
            continue
        if ok:
            say(f"  [works] {info}")
            report[name] = {"status": "ok", "python": py, "info": info,
                            "torch": str(build) if build else None}
            return True
        say(f"  check failed on Python {py}: {info}; trying the next option")
    if not tried:
        say(f"  !! no Python version from {PY_CANDIDATES[-1]} to {PY_CANDIDATES[0]} "
            f"has wheels for all of {name}'s packages on this platform.")
        say("     rerun with --verbose to see why each one failed.")
        report[name] = {"status": "unresolvable"}
        return False
    report[name] = {"status": "failed"}
    return False


# ---------------------------------------------------------------------------
#  system tools
# ---------------------------------------------------------------------------
def find_tesseract():
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(cand):
            return cand
    return None


def linux_package_manager():
    for pm, install, pkgs in (
        ("apt-get", ["apt-get", "install", "-y"], ["tesseract-ocr", "libgl1", "libglib2.0-0"]),
        ("dnf", ["dnf", "install", "-y"], ["tesseract", "mesa-libGL", "glib2"]),
        ("pacman", ["pacman", "-S", "--needed", "--noconfirm"],
         ["tesseract", "tesseract-data-eng", "libglvnd", "glib2"]),
        ("zypper", ["zypper", "--non-interactive", "install"],
         ["tesseract-ocr", "tesseract-ocr-traineddata-english", "Mesa-libGL1", "glib2"]),
    ):
        if shutil.which(pm):
            return pm, install, pkgs
    return None, None, None


def system_tools(args, report):
    head("system tools")
    if IS_WIN:
        sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        if all(os.path.exists(os.path.join(sysdir, d))
               for d in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")):
            say("  Visual C++ runtime: present")
        else:
            say("  Visual C++ runtime: MISSING - torch fails with 'c10.dll' errors without it")
            if shutil.which("winget") and ask("  Install it with winget?", args):
                run(["winget", "install", "-e", "--id", "Microsoft.VCRedist.2015+.x64",
                     "--accept-source-agreements", "--accept-package-agreements"], check=False)
            else:
                say("  get it from https://aka.ms/vs/17/release/vc_redist.x64.exe")

    tess = find_tesseract()
    if tess:
        say(f"  Tesseract: {tess}")
    elif IS_WIN:
        say("  Tesseract: not found (backup OCR engine; PaddleOCR is the main one)")
        if shutil.which("winget") and ask("  Install Tesseract with winget?", args):
            run(["winget", "install", "-e", "--id", "UB-Mannheim.TesseractOCR",
                 "--accept-source-agreements", "--accept-package-agreements"], check=False)
            tess = find_tesseract()
        else:
            say("  get it from https://github.com/UB-Mannheim/tesseract/wiki")
    elif IS_MAC:
        say("  Tesseract: not found")
        if shutil.which("brew") and ask("  Install it with Homebrew?", args):
            run(["brew", "install", "tesseract"], check=False)
            tess = find_tesseract()
    else:
        import ctypes.util
        missing_gl = ctypes.util.find_library("GL") is None
        pm, install, pkgs = linux_package_manager()
        if not tess or missing_gl:
            what = ", ".join(x for x, miss in (("Tesseract", not tess),
                                               ("libGL (OpenCV)", missing_gl)) if miss)
            say(f"  missing: {what}")
            if pm and ask(f"  Install {' '.join(pkgs)} with {pm} (needs sudo)?", args):
                sudo = [] if os.geteuid() == 0 else ["sudo"]
                if pm == "apt-get":
                    run(sudo + ["apt-get", "update"], check=False)
                run(sudo + install + pkgs, check=False)
                tess = find_tesseract()
            elif not pm:
                say("  install tesseract and libGL with your package manager")
        else:
            say("  libGL: present")
    report["tesseract"] = tess

    model = os.path.join(TESSDATA_BEST, "eng.traineddata")
    if os.path.exists(model):
        say(f"  Tesseract best English model: {model}")
    elif not args.dry_run:
        say("  downloading Tesseract's best English model (~15 MB) ...")
        os.makedirs(TESSDATA_BEST, exist_ok=True)
        try:
            urllib.request.urlretrieve(TESSDATA_URL, model + ".part")
            os.replace(model + ".part", model)
            say(f"  saved {model}")
        except OSError as e:
            say(f"  !! download failed ({e}); Tesseract will use its default model")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def write_launchers():
    """audiobook.cmd / audiobook.sh ship with the repo; make sure .sh is runnable
    (a zip download loses the executable bit)."""
    path = os.path.join(HERE, "audiobook.sh")
    if not IS_WIN and os.path.exists(path):
        os.chmod(path, 0o755)


def main():
    ap = argparse.ArgumentParser(description="Install doc-to-audiobook's toolchain.")
    ap.add_argument("--only", default="", help="comma list: " + ",".join(COMPONENTS))
    ap.add_argument("--skip", default="", help="comma list of envs to leave out")
    ap.add_argument("--cuda", default="auto",
                    help="auto (default), cpu, cu118, cu124, cu126 or cu128")
    ap.add_argument("--python", default="", help="force this Python version, e.g. 3.11")
    ap.add_argument("--recreate", action="store_true", help="rebuild working envs too")
    ap.add_argument("--models", action="store_true",
                    help="pre-download Kokoro and OmniVoice weights now")
    ap.add_argument("--dry-run", action="store_true", help="resolve only, install nothing")
    ap.add_argument("--check", action="store_true", help="just test the existing envs")
    ap.add_argument("--no-system", action="store_true",
                    help="do not look for or install Tesseract / VC++ / libGL")
    ap.add_argument("--yes", "-y", action="store_true", help="answer yes to every question")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="say why each Python version was rejected")
    ap.add_argument("--platform", default="", help=argparse.SUPPRESS)  # testing: resolve for another OS
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(COMPONENTS)
    names = [n for n in names if n not in {s.strip() for s in args.skip.split(",")}]
    bad = [n for n in names if n not in COMPONENTS]
    if bad:
        sys.exit(f"unknown env(s): {', '.join(bad)} - choose from {', '.join(COMPONENTS)}")

    say("doc-to-audiobook installer - https://github.com/emdaitaj/doc-to-audiobook")
    say(f"  system   {platform.system()} {platform.release()} ({platform.machine()})")
    say(f"  envs     {ENVS}")

    if args.check:
        failures = 0
        for name in names:
            ok, info = smoke(os.path.join(ENVS, name), COMPONENTS[name])
            failures += not ok
            say(f"  {name:10s} {'OK  ' if ok else 'FAIL'} {info}")
        tess = find_tesseract()
        say(f"  {'tesseract':10s} {'OK  ' if tess else 'none'} {tess or ''}")
        sys.exit(1 if failures else 0)

    uv = find_uv()
    if not uv:
        sys.exit("uv was not found. Run install.ps1 / install.sh (they fetch it), "
                 "or install it from https://docs.astral.sh/uv/ and rerun.")
    say(f"  uv       {run([uv, '--version'], capture=True).stdout.strip()}")

    gpu, cuda = (None, None) if IS_MAC else detect_gpu()
    if gpu:
        say(f"  GPU      {gpu}" + (f" (driver supports CUDA {cuda[0]}.{cuda[1]})" if cuda else ""))
    else:
        say("  GPU      none found - narration will run on the CPU (much slower)")
    builds = torch_builds_for(cuda, args.cuda)
    if any(COMPONENTS[n].get("torch") for n in names):
        say(f"  torch    trying {', '.join(b.tag for b in builds)} in that order")

    os.makedirs(ENVS, exist_ok=True)
    report = {}
    if not args.no_system:
        system_tools(args, report)

    results = {n: install_component(uv, n, builds, args, report) for n in names}

    if args.models and not args.dry_run:
        head("voice models")
        for name in ("kokoro", "omnivoice"):
            if results.get(name):
                say(f"  downloading {name} weights ...")
                run([venv_python(os.path.join(ENVS, name)), "-c", MODEL_PREFETCH[name]],
                    check=False)

    if not args.dry_run:
        write_launchers()
        old = {}
        summary = os.path.join(ENVS, "installed.json")
        if os.path.exists(summary):
            try:
                old = json.load(open(summary, encoding="utf-8"))
            except ValueError:
                old = {}
        old.update(report)
        with open(summary, "w", encoding="utf-8") as fh:
            json.dump(old, fh, indent=2)

    head("summary")
    for name in names:
        r = report.get(name, {})
        detail = r.get("info") or (f"Python {r.get('python')}"
                                   + (f" + {r['torch']}" if r.get("torch") else "")
                                   if r.get("python") else "")
        say(f"  {name:10s} {r.get('status', '?'):13s} {detail}")
    if not args.no_system:
        say(f"  {'tesseract':10s} {('found' if report.get('tesseract') else 'missing'):13s} "
            f"{report.get('tesseract') or ''}")
    if not args.dry_run and all(results.values()):
        say()
        say("Done. Try:")
        say('  audiobook.cmd "book.pdf" --transcript-only' if IS_WIN
            else '  ./audiobook.sh "book.pdf" --transcript-only')
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
