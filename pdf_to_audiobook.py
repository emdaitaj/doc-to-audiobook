#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
pdf_to_audiobook.py  -  One-shot document -> natural audiobook (MP3).

Handles the formats you actually have lying around:

    .pdf                  text-based OR scanned (auto-detected; OCR when needed)
    .epub                 unpacked in spine order, chapter by chapter
    .md / .markdown       markdown syntax stripped so it reads as prose
    .html / .htm / .xhtml tags stripped, headings kept as their own beat
    .docx                 paragraphs pulled straight out of the XML
    .txt and anything     treated as plain text
    else that is text

Because Kokoro, OmniVoice and PaddleOCR live in different virtualenvs, this
orchestrator runs under the "core" Python (which has PyMuPDF + OpenCV) and
shells out to each venv for its heavy stage.  install.py builds them all.  Only the PDF path needs
PyMuPDF; every other format is pure standard library.

Usage:
    python pdf_to_audiobook.py "book.pdf"
    python pdf_to_audiobook.py "book.epub" --list
    python pdf_to_audiobook.py "book.epub" --pages 3-40 --voice am_michael
    python pdf_to_audiobook.py "notes.md" --transcript-only
    python pdf_to_audiobook.py "scan.pdf" --layout single --ocr tesseract
    python pdf_to_audiobook.py "book.pdf" --engine omnivoice
    python pdf_to_audiobook.py "roman.pdf" --engine omnivoice --lang de
    python pdf_to_audiobook.py "book.pdf" --engine omnivoice --ref-audio me.wav
    python pdf_to_audiobook.py --engine omnivoice --list-voices

Run it with the core env that install.py creates:
    envs/core/Scripts/python.exe pdf_to_audiobook.py ...   (Windows)
    envs/core/bin/python pdf_to_audiobook.py ...           (Linux / macOS)
or use the audiobook.cmd / audiobook.sh wrappers.

Part of doc-to-audiobook - https://github.com/emdaitaj/doc-to-audiobook
Copyright (C) 2026 emdaitaj. AGPL-3.0 with attribution terms (see NOTICE).
"""
import argparse, difflib, html, json, os, posixpath, re, statistics, subprocess, sys, zipfile
NEWLINE = chr(10)
from html.parser import HTMLParser
from urllib.parse import unquote

# ---- tool locations ----------------------------------------------------------
# install.py builds one virtualenv per heavy stage under envs/ next to this
# file, because their dependencies cannot share one interpreter (PaddleOCR 2.x
# wants NumPy 1.x, Kokoro and OmniVoice want different torch builds).
# Every location can be overridden with an environment variable.
HERE = os.path.dirname(os.path.abspath(__file__))


def venv_python(env_dir):
    """The interpreter inside a virtualenv, on Windows or Linux/macOS."""
    for rel in (("Scripts", "python.exe"), ("python.exe",), ("bin", "python")):
        path = os.path.join(env_dir, *rel)
        if os.path.exists(path):
            return path
    return os.path.join(env_dir, "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")


def find_tesseract():
    import shutil
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(cand):
            return cand
    return "tesseract"


ENVS   = os.environ.get("AUDIOBOOK_ENVS", os.path.join(HERE, "envs"))
KVENV  = os.environ.get("KOKORO_PY",     venv_python(os.path.join(ENVS, "kokoro")))
PVENV  = os.environ.get("PADDLE_PY",     venv_python(os.path.join(ENVS, "paddle")))
OVENV  = os.environ.get("OMNIVOICE_PY",  venv_python(os.path.join(ENVS, "omnivoice")))
TESS   = os.environ.get("TESSERACT") or find_tesseract()
# Tesseract's "best" English model, downloaded by install.py. Without it the
# system tessdata is used.
TBEST  = os.environ.get("TESSDATA_BEST", os.path.join(HERE, "tessdata_best"))
DESKEW = os.environ.get("DESKEW_PY",     os.path.join(HERE, "tools", "book_deskew_split.py"))
OMODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

# ---- which extension goes down which path -----------------------------------
PDF_EXT  = {".pdf"}
EPUB_EXT = {".epub"}
MD_EXT   = {".md", ".markdown", ".mdown", ".mkd", ".mdx"}
HTML_EXT = {".html", ".htm", ".xhtml", ".xht"}
DOCX_EXT = {".docx", ".docm"}

# ---- small OCR-error dictionary for scanned pages ---------------------------
OCR_FIX = {
 "Amencan":"American","Amencans":"Americans","Amenca":"America","naton":"nation",
 "natonal":"national","berween":"between","penod":"period","polincal":"political",
 "polincs":"politics","idenuty":"identity","conunent":"continent","wnter":"writer",
 "wnters":"writers","wnting":"writing","atizen":"citizen","atizens":"citizens",
 "carly":"early","clite":"elite","socicty":"society","Bntish":"British","thc":"the",
 "Amcrican":"American","publshed":"published","Constirution":"Constitution",
 "itelf":"itself","Washingron":"Washington","carne":"came","settement":"settlement",
}
ABBREV = [(r"\bpp\.\s*", "pages "), (r"\bp\.\s*(?=\d)", "page "),
          (r"\bVol(?:ume)?\.\s*", "Volume "), (r"\bNo\.\s*(?=\d)", "Number "),
          (r"\bvs\.", "versus"), (r"\bMrs\.", "Missus"), (r"\bMr\.", "Mister"),
          (r"\s&\s", " and ")]


def sh(cmd, **kw):
    """Run a subprocess, raise on failure, return CompletedProcess."""
    return subprocess.run(cmd, check=True, **kw)


def parse_pages(spec, total):
    if not spec:
        return list(range(total))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(p for p in out if 0 <= p < total)


# ---------------------------------------------------------------------------
#  Shared text tidying (every format funnels through finalize())
# ---------------------------------------------------------------------------
def expand(text):
    for pat, rep in ABBREV:
        text = re.sub(pat, rep, text)
    text = re.sub(r"([A-Za-z])/([A-Za-z])", r"\1 \2", text)
    return text


def strip_urls(t):
    """Bare URLs read aloud are torture; drop them (link text is kept)."""
    t = re.sub(r"<(?:https?|mailto|ftp):[^>\s]*>", " ", t)
    t = re.sub(r"\b(?:https?://|www\.)\S+", " ", t)
    return t


def end_stop(t):
    """Give headings / list items a full stop so Kokoro doesn't clip them."""
    return t if re.search(r"""[.!?:;,\u2026]["')\]\u201d\u2019]?$""", t) else t + "."


def finalize(paras):
    """List of raw paragraphs -> the blank-line-separated transcript."""
    out = []
    for p in paras:
        p = p.replace("\u00a0", " ").replace("\ufeff", "").replace("\u200b", "")
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r" +([,.;:!?])", r"\1", p)          # space left by a dropped URL
        p = re.sub(r"\(\s*\)|\[\s*\]", "", p).strip()
        if not re.search(r"[^\W_]", p):                # nothing pronounceable (any script)
            continue
        out.append(expand(p))
    return "\n\n".join(out).strip()


def read_text_file(path):
    """Read a text file, guessing the encoding; refuse obvious binaries."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if b"\x00" in raw[:4096]:
        sys.exit(f"[error] {os.path.basename(path)} looks like a binary file, not text.\n"
                 "        Supported: .pdf .epub .md .html .docx .txt (and any plain-text file).")
    return decode(raw)


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
#  HTML / XHTML  (also used for every EPUB chapter)
# ---------------------------------------------------------------------------
SKIP_TAGS  = {"script", "style", "head", "title", "noscript", "svg", "nav",
              "form", "button", "select", "textarea", "iframe"}
HEAD_TAGS  = {"h1", "h2", "h3", "h4", "h5", "h6"}
ITEM_TAGS  = {"li", "dd", "dt", "td", "th", "caption", "figcaption"}
BLOCK_TAGS = {"p", "div", "hr", "tr", "pre", "section", "article", "aside",
              "header", "footer", "figure", "blockquote", "table", "ul", "ol",
              "dl", "main", "body", "address"} | HEAD_TAGS | ITEM_TAGS


class HTMLToText(HTMLParser):
    """Very forgiving tags -> paragraphs converter (stdlib only)."""

    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.paras, self.heads, self.buf = [], [], []
        self.skip = self.head = self.item = 0

    def flush(self):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if not t:
            return
        if self.head:
            t = end_stop(t)
            self.heads.append(t)
        elif self.item:
            t = end_stop(t)
        self.paras.append(t)

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "br":                       # soft line break, not a new paragraph
            self.buf.append(" ")
        elif tag in BLOCK_TAGS:
            self.flush()
            if tag in HEAD_TAGS:
                self.head += 1
            elif tag in ITEM_TAGS:
                self.item += 1

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag == "br":
            return
        if tag in BLOCK_TAGS:
            self.flush()
            if tag in HEAD_TAGS:
                self.head = max(0, self.head - 1)
            elif tag in ITEM_TAGS:
                self.item = max(0, self.item - 1)

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def html_to_paragraphs(raw):
    """-> (paragraphs, meta) where meta feeds the EPUB section labels."""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
    doc_title = re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else ""
    p = HTMLToText()
    try:
        p.feed(raw)
        p.close()
    except Exception as e:                       # malformed markup: keep what we got
        print(f"[html] parser stopped early ({e})")
    p.flush()
    paras = [strip_urls(t) for t in p.paras]
    head = p.heads[0] if p.heads else ""
    if len(p.heads) > 1 and len(re.sub(r"\W", "", head)) <= 4:
        head = head.rstrip(".") + " " + p.heads[1]          # "1" -> "1 The Envious Friend"
    return paras, {"head": head, "title": doc_title}


# ---------------------------------------------------------------------------
#  EPUB
# ---------------------------------------------------------------------------
def zip_read(zf, name):
    """Zip lookup that tolerates case / percent-encoding differences."""
    try:
        return zf.read(name)
    except KeyError:
        low = unquote(name).lower().lstrip("/")
        for n in zf.namelist():
            if n.lower() == low:
                return zf.read(n)
        raise


def epub_spine(zf):
    """-> list of hrefs (zip paths) of the content documents, in reading order."""
    from xml.etree import ElementTree as ET

    def local(tag):
        return tag.split("}")[-1]

    opf = ""
    try:
        cont = ET.fromstring(zip_read(zf, "META-INF/container.xml"))
        for el in cont.iter():
            if local(el.tag) == "rootfile" and el.get("full-path"):
                opf = el.get("full-path")
                break
    except Exception:
        pass
    if not opf:
        cand = [n for n in zf.namelist() if n.lower().endswith(".opf")]
        opf = cand[0] if cand else ""

    docs = []
    if opf:
        base = posixpath.dirname(opf)
        try:
            root = ET.fromstring(zip_read(zf, opf))
        except Exception as e:
            print(f"[epub] unreadable package file ({e}); falling back to file order")
            root = None
        if root is not None:
            manifest, spine = {}, []
            for el in root.iter():
                t = local(el.tag)
                if t == "item":
                    manifest[el.get("id")] = (el.get("href", ""),
                                              el.get("media-type", "") or "",
                                              el.get("properties", "") or "")
                elif t == "itemref":
                    spine.append((el.get("idref"), (el.get("linear") or "yes").lower()))
            order = spine or [(i, "yes") for i in manifest]
            for idref, linear in order:
                href, mtype, props = manifest.get(idref, ("", "", ""))
                if not href or linear == "no" or "nav" in props.split():
                    continue
                if "html" not in mtype and not re.search(r"\.x?html?$", href, re.I):
                    continue
                full = unquote(href.split("#")[0])
                docs.append(posixpath.normpath(posixpath.join(base, full)) if base else full)
    if not docs:
        docs = sorted(n for n in zf.namelist()
                      if re.search(r"\.x?html?$", n, re.I) and "META-INF" not in n)
    return docs


def epub_sections(path):
    """-> [(label, [paragraph, ...]), ...] in reading order."""
    docs = []
    with zipfile.ZipFile(path) as zf:
        for href in epub_spine(zf):
            try:
                raw = decode(zip_read(zf, href))
            except KeyError:
                print(f"[epub] missing from archive: {href}")
                continue
            paras, meta = html_to_paragraphs(raw)
            meta["href"] = href
            docs.append((meta, paras))
    # a <title> repeated on every chapter is just the book name - useless as a label
    use_title = len({m["title"] for m, _ in docs if m["title"]}) > 1
    out = []
    for meta, paras in docs:
        cands = [meta["head"]] if meta["head"] else []
        if use_title and meta["title"]:
            cands.append(meta["title"])
        cands += [p for p in paras[:4] if p not in cands]
        label = ""
        for c in cands:                     # keep adding until it says something
            label = (label + " " + c).strip()
            if len(re.sub(r"\W", "", label)) > 4:
                break
        label = label or os.path.splitext(posixpath.basename(meta["href"]))[0]
        out.append((re.sub(r"\s+", " ", label).strip(" .")[:58], paras))
    return out


def read_epub(path, spec):
    secs = epub_sections(path)
    if not secs:
        sys.exit("[error] no readable chapters found in that EPUB.")
    keep = parse_pages(spec, len(secs))
    print(f"[epub] {len(secs)} section(s), narrating {len(keep)}"
          f"{' (use --list to see them)' if not spec else ''}")
    paras = []
    for i in keep:
        paras.extend(secs[i][1])
    return finalize(paras)


def list_epub(path):
    secs = epub_sections(path)
    print(f"{len(secs)} section(s) in {os.path.basename(path)}:\n")
    for i, (label, paras) in enumerate(secs):
        words = sum(len(p.split()) for p in paras)
        print(f"  {i:>3}  {label[:52]:<52} {words:>7,} words")
    total = sum(sum(len(p.split()) for p in ps) for _, ps in secs)
    print(f"\n  total {total:,} words  (~{total / 155:.0f} min narrated)")
    print("  narrate a range with:  --pages 3-40")


# ---------------------------------------------------------------------------
#  Markdown
# ---------------------------------------------------------------------------
FENCE = re.compile(r"^[ \t]*(```+|~~~+)[^\n]*\n(.*?)(?:^[ \t]*\1[ \t]*$|\Z)", re.S | re.M)


def md_inline(t):
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                    # images -> gone
    t = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                 # link -> its text
    t = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", t)
    t = re.sub(r"\[\^[^\]]+\]", "", t)                             # footnote markers
    t = re.sub(r"`+([^`]*)`+", r"\1", t)                           # inline code
    t = re.sub(r"~~(.+?)~~", r"\1", t)                             # strikethrough
    t = re.sub(r"(\*\*\*|___|\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", r"\2", t)   # emphasis
    t = re.sub(r"</?[A-Za-z][^>]*>", " ", t)                       # inline html
    t = html.unescape(t)
    t = strip_urls(t)
    t = re.sub(r"\\([\\`*_{}\[\]()#+.!\-])", r"\1", t)             # backslash escapes
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def md_to_paragraphs(text, read_code=False):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\A(?:---|\+\+\+)\n.*?\n(?:---|\+\+\+)[ \t]*\n", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    dropped = len(FENCE.findall(text))
    text = FENCE.sub(lambda m: ("\n\n" + m.group(2) + "\n\n") if read_code else "\n\n", text)
    if dropped and not read_code:
        print(f"[md] skipped {dropped} code block(s) - pass --read-code to narrate them")
    # setext headings -> atx, so the block loop below sees them as headings
    text = re.sub(r"^([^\n\s][^\n]*)\n[ \t]*={2,}[ \t]*$", r"# \1", text, flags=re.M)
    text = re.sub(r"^([^\n\s][^\n]*)\n[ \t]*-{2,}[ \t]*$", r"## \1", text, flags=re.M)

    out = []

    def flush(lines, item=False):
        if not lines:
            return
        t = md_inline(" ".join(lines))
        if t:
            out.append(end_stop(t) if item else t)
        del lines[:]

    for block in re.split(r"\n[ \t]*\n", text):
        lines = [l.rstrip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        pipes = sum(1 for l in lines if l.lstrip().startswith("|"))
        if pipes >= 2 and pipes >= len(lines) // 2:                # a table
            for l in lines:
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    continue                                       # header rule
                row = md_inline(", ".join(c for c in cells if c))
                if row:
                    out.append(end_stop(row))
            continue
        cur, cur_item = [], False
        for l in lines:
            s = l.strip()
            if re.fullmatch(r"(?:[-*_][ \t]*){3,}", s):            # horizontal rule
                continue
            if re.match(r"^\[[^\]]+\]:\s*\S+", s):                 # link definition
                continue
            s = re.sub(r"^>+[ \t]?", "", s).strip()                # blockquote marker
            m = re.match(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*$", s)
            if m:
                flush(cur, cur_item); cur_item = False
                h = md_inline(m.group(2))
                if h:
                    out.append(end_stop(h))
                continue
            m = re.match(r"^(?:[-*+]|\d{1,3}[.)])[ \t]+(.*)$", s)
            if m:                                                  # new list item
                flush(cur, cur_item)
                cur, cur_item = [m.group(1)], True
                continue
            cur.append(s)
        flush(cur, cur_item)
    return out


# ---------------------------------------------------------------------------
#  DOCX
# ---------------------------------------------------------------------------
def docx_to_paragraphs(path):
    with zipfile.ZipFile(path) as zf:
        try:
            xml = decode(zf.read("word/document.xml"))
        except KeyError:
            sys.exit("[error] that .docx has no word/document.xml - is it really a Word file?")
    # tabs / line breaks are outside <w:t>, so smuggle them in as runs of their own
    xml = re.sub(r"<w:(?:tab|br|cr)\b[^>]*/?>", "<w:t> </w:t>", xml)
    out = []
    for chunk in re.split(r"</w:p>", xml):
        style = re.search(r'w:pStyle[^>]*w:val="([^"]+)"', chunk)
        txt = "".join(re.findall(r"<w:t\b[^>]*>(.*?)</w:t>", chunk, re.S))
        txt = re.sub(r"\s+", " ", html.unescape(txt)).strip()
        if not txt:
            continue
        if style and re.match(r"(?i)heading|title|subtitle", style.group(1)):
            txt = end_stop(txt)
        out.append(txt)
    return out


# ---------------------------------------------------------------------------
#  Plain text
# ---------------------------------------------------------------------------
def plain_to_paragraphs(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n")
    text = re.sub(r"([A-Za-z])-\n[ \t]*([a-z])", r"\1\2", text)    # de-hyphenate
    blocks = re.split(r"\n[ \t]*\n", text)
    if len(blocks) < 2 and text.count("\n") > 8:                   # line-per-entry file
        blocks = text.split("\n")
    out = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        lines = [l for l in lines
                 if not re.fullmatch(r"[\divxlcIVXLC.\-\u2013\u2014]{1,6}", l)]   # page numbers
        if not lines:
            continue
        para = ""
        for l in lines:
            if not para:
                para = l
            elif re.search(r"[A-Za-z]-$", para):
                para = para[:-1] + l
            else:
                para = para + " " + l
        out.append(para)
    return out


# ---------------------------------------------------------------------------
#  PDF: detection
# ---------------------------------------------------------------------------
def detect(pdf, pages):
    import fitz
    doc = fitz.open(pdf)
    pages = pages or list(range(doc.page_count))
    chars = sum(len(doc[i].get_text().strip()) for i in pages)
    per_page = chars / max(1, len(pages))
    aspects = [doc[i].rect.width / doc[i].rect.height for i in pages]
    aspect = statistics.median(aspects)
    doc.close()
    kind = "text" if per_page > 100 else "scanned"
    layout = "side-by-side" if aspect > 1.15 else "single"
    print(f"[detect] {len(pages)} page(s), ~{per_page:.0f} chars/page -> {kind.upper()}"
          f" | median aspect {aspect:.2f} -> layout '{layout}'")
    return kind, layout


# ---------------------------------------------------------------------------
#  PDF: TEXT-based path
# ---------------------------------------------------------------------------
def clean_text_pdf(pdf, pages):
    import fitz
    doc = fitz.open(pdf)
    raw = "\n\n".join(doc[i].get_text() for i in pages)
    doc.close()
    raw = raw.replace("\ufffd", "\u2014")             # replacement char -> em dash
    raw = re.sub(r"([A-Za-z])-\n[ \t]*([a-z])", r"\1\2", raw)   # de-hyphenate line breaks
    out = []
    for block in re.split(r"\n[ \t]*\n", raw):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        lines = [l for l in lines if not re.fullmatch(r"[\divxlcIVXLC.\-\u2013\u2014]{1,6}", l)]  # page nums
        if not lines:
            continue
        para = ""
        for l in lines:
            if not para:
                para = l
            elif re.search(r"[A-Za-z]-$", para):
                para = para[:-1] + l
            else:
                para = para + " " + l
        out.append(expand(re.sub(r"[ \t]{2,}", " ", para)))
    return "\n\n".join(out).strip()


# ---------------------------------------------------------------------------
#  PDF: SCANNED path
# ---------------------------------------------------------------------------
PADDLE_WORKER = r'''
import sys, glob, os, json
import numpy as np
np.int=int; np.float=float; np.bool=bool; np.object=object
from paddleocr import PaddleOCR
from PIL import Image
img_dir, out_json = sys.argv[1], sys.argv[2]
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
pages=[]
for img in sorted(glob.glob(os.path.join(img_dir,"*.png"))):
    W,H = Image.open(img).size
    res = ocr.ocr(img, cls=True)
    lines=[]
    for box,(txt,conf) in (res[0] or []):
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        lines.append({"text":txt,"left":min(xs),"right":max(xs),"top":min(ys),"bottom":max(ys)})
    pages.append({"name":os.path.basename(img),"W":W,"H":H,"lines":lines})
json.dump(pages, open(out_json,"w",encoding="utf-8"), ensure_ascii=False)
print("OCRPAGES", len(pages))
'''


def render_or_split(pdf, pages, layout, split_dir):
    """Produce one PNG per book-page in split_dir. Uses the deskew/split script
    when available (best for open-book scans); else renders pages directly."""
    os.makedirs(split_dir, exist_ok=True)
    spec = f"{min(pages)}-{max(pages)}" if pages else ""
    if os.path.exists(DESKEW):
        cmd = [sys.executable, DESKEW, pdf, "-o", os.path.join(split_dir, "_ds.pdf"),
               "--dpi", "800", "--layout", layout, "--gutter", "detect",
               "--mode", "gray", "--crop", "--debug-dir", split_dir]
        if spec:
            cmd += ["--pages", spec]
        print(f"[split] deskew+split ({layout}) ...")
        sh(cmd)
    else:
        print("[split] deskew script not found; rendering pages at 300 DPI")
        import fitz
        doc = fitz.open(pdf)
        for i in (pages or range(doc.page_count)):
            doc[i].get_pixmap(dpi=300).save(os.path.join(split_dir, f"p{i:03d}.png"))
        doc.close()
    return sorted(f for f in os.listdir(split_dir) if f.endswith(".png"))


def clean_scanned_page(pg):
    """PaddleOCR line boxes for one page -> (body_text, footnote_text)."""
    W, H = pg["W"], pg["H"]
    ls = sorted(pg["lines"], key=lambda l: l["top"])
    ls = [l for l in ls if "camscanner" not in l["text"].lower().replace(" ", "")
          and l["text"].strip().upper() != "CS"]
    ls = [l for l in ls if l["top"] > 0.055 * H]                       # drop running header
    def pageno(t):
        s = t.strip(); return s and len(s) <= 4 and re.fullmatch(r"[0-9ivxlcIVXLC]+", s)
    while ls and ls[-1]["top"] > 0.84 * H and pageno(ls[-1]["text"]):  # drop page number
        ls.pop()
    if not ls:
        return "", ""
    tops = [l["top"] for l in ls]
    sp = [tops[i + 1] - tops[i] for i in range(len(ls) - 1)]
    body, foot = ls, []
    if len(sp) >= 4:                                                   # split off footnote cluster
        top_sp = [sp[i] for i in range(len(sp)) if tops[i] < 0.55 * H] or sp
        base = statistics.median(top_sp) or 1
        cand = [(sp[i], i) for i in range(len(sp)) if tops[i + 1] > 0.58 * H]
        if cand:
            g, i = max(cand, key=lambda x: x[0])
            below = ls[i + 1:]
            below_sp = [tops[j + 1] - tops[j] for j in range(i + 1, len(ls) - 1)]
            tight = statistics.median(below_sp) < 0.82 * base if below_sp else False
            if g > 1.55 * base and len(below) >= 2 and tight:
                body, foot = ls[:i + 1], below
    thr = statistics.median(l["left"] for l in body) + 0.018 * W       # paragraph indents
    paras, cur = [], ""
    for k, l in enumerate(body):
        t = l["text"].strip()
        if cur and k > 0 and l["left"] > thr:
            paras.append(cur); cur = t
        elif not cur:
            cur = t
        elif re.search(r"[A-Za-z]-$", cur):
            cur = cur[:-1] + t
        else:
            cur = cur + " " + t
    if cur:
        paras.append(cur)
    body_text = "\n\n".join(clean_line(p) for p in paras)
    foot_text = clean_line(" ".join(l["text"] for l in foot)) if foot else ""
    return body_text, foot_text


def clean_line(t):
    t = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", t)          # missing space after comma
    t = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", t)          # missing space after period
    t = re.sub(r"[A-Za-z]+", lambda m: OCR_FIX.get(m.group(0), m.group(0)), t)
    return expand(re.sub(r"[ \t]{2,}", " ", t)).strip()


def ocr_tesseract(split_dir, names):
    """Fallback OCR: Tesseract best model, light text cleanup."""
    body_pages, foots = [], []
    for nm in names:
        cmd = [TESS, os.path.join(split_dir, nm), "stdout", "--psm", "6", "-l", "eng"]
        if os.path.exists(os.path.join(TBEST, "eng.traineddata")):
            cmd += ["--tessdata-dir", TBEST]
        r = sh(cmd, capture_output=True, text=True, encoding="utf-8")
        lines = [l.rstrip() for l in (r.stdout or "").splitlines()]
        lines = [l for l in lines if l.strip() and "camscanner" not in l.lower()]
        if lines and len(lines[0].strip()) < 45:            # likely running header
            lines = lines[1:]
        lines = [l for l in lines if not re.fullmatch(r"\s*[\divxlcIVXLC]{1,4}\s*", l)]
        txt = "\n".join(lines)
        txt = re.sub(r"([A-Za-z])-\n[ \t]*([a-z])", r"\1\2", txt)
        txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)
        body_pages.append(clean_line(txt))
    return body_pages, foots


def assemble(body_pages, foot_pages):
    """Join page bodies into flowing paragraphs; footnotes go to the end."""
    paras = []
    for body in body_pages:
        for block in re.split(r"\n\s*\n", body):
            block = re.sub(r"\s+", " ", block).strip()
            if not block:
                continue
            if paras and (not re.search(r'[.!?)"\u201d\u2019]$', paras[-1]) or block[:1].islower()):
                if re.search(r"[A-Za-z]-$", paras[-1]):
                    paras[-1] = paras[-1][:-1] + block
                else:
                    paras[-1] = paras[-1] + " " + block
            else:
                paras.append(block)
    text = "\n\n".join(paras)
    notes = [f for f in foot_pages if f and f.strip()]
    if notes:
        text += "\n\n\nNotes and footnotes.\n\n" + "\n\n".join(notes)
    return text


def run_scanned(pdf, pages, layout, workdir, ocr_engine):
    split_dir = os.path.join(workdir, "split")
    names = render_or_split(pdf, pages, layout, split_dir)
    print(f"[split] {len(names)} page image(s)")
    if ocr_engine == "paddle":
        try:
            out_json = os.path.join(workdir, "ocr.json")
            print("[ocr] PaddleOCR ...")
            sh([PVENV, "-", split_dir, out_json], input=PADDLE_WORKER,
               text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            pgs = json.load(open(out_json, encoding="utf-8"))
            bodies, foots = zip(*(clean_scanned_page(p) for p in pgs))
            return assemble(list(bodies), list(foots))
        except Exception as e:
            print(f"[ocr] PaddleOCR failed ({e}); falling back to Tesseract")
    bodies, foots = ocr_tesseract(split_dir, names)
    return assemble(bodies, foots)


def run_pdf(pdf, args):
    import fitz
    doc = fitz.open(pdf)
    total = doc.page_count
    doc.close()
    pages = parse_pages(args.pages, total)
    kind, layout = detect(pdf, pages)
    if args.force_scanned:
        kind = "scanned"
    if args.layout != "auto":
        layout = args.layout
    if kind == "text":
        return clean_text_pdf(pdf, pages)
    return run_scanned(pdf, pages, layout, args.outdir, args.ocr)


# ---------------------------------------------------------------------------
#  TTS + encode
#
#  Two engines live here.  Both get the same job manifest and both stream raw
#  float32 PCM into ffmpeg, so nothing downstream knows which one ran:
#
#    kokoro     Kokoro-82M, English only, fixed voice list, very fast.
#    omnivoice  k2-fsa OmniVoice 0.6B - 600+ languages, voice built from a
#               fixed attribute vocabulary, zero-shot cloning.
# ---------------------------------------------------------------------------
KOKORO_WORKER = r'''
import sys, os, re, json, time, numpy as np, torch, subprocess, imageio_ffmpeg
from kokoro import KPipeline
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
jobs, voice, speed = cfg["jobs"], cfg["voice"], float(cfg["speed"])
SR=24000
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("[tts] device", dev, flush=True)
pipe = KPipeline(lang_code="a", device=dev)
ff=imageio_ffmpeg.get_ffmpeg_exe()
total_words=sum(j["words"] for j in jobs) or 1
sil=lambda s: np.zeros(int(SR*s),dtype=np.float32)
grand=0; done_words=0; t0=time.time()
for k,job in enumerate(jobs,1):
    paras=[p.strip() for p in re.split(r"\n\s*\n", open(job["txt"],encoding="utf-8").read()) if p.strip()]
    # Stream PCM straight into ffmpeg: a long book is far too big to hold in RAM, and a
    # WAV over 4 GB is not representable at all. loudnorm replaces the old peak-normalise
    # pass, which needed the whole book in memory to find its maximum.
    part=job["mp3"]+".part"
    proc=subprocess.Popen([ff,"-y","-hide_banner","-loglevel","error",
                           "-f","f32le","-ar",str(SR),"-ac","1","-i","pipe:0",
                           "-af","loudnorm=I=-20:TP=-1.5:LRA=11",
                           "-codec:a","libmp3lame","-b:a","96k","-ac","1",
                           "-f","mp3",part],
                          stdin=subprocess.PIPE)
    def emit(a):
        proc.stdin.write(np.ascontiguousarray(a,dtype=np.float32).tobytes())
        return a.size
    n=0; ok=False; offsets=[]
    if len(jobs)>1:
        print(f"[tts] ({k}/{len(jobs)}) {job['label']} - {job['words']:,} words", flush=True)
    try:
        for i,para in enumerate(paras):
            if i: n+=emit(sil(0.6 if len(para)<70 else 0.45))
            offsets.append(int(n))   # sample offset where this paragraph's audio starts
            for r in pipe(para, voice=voice, speed=speed):
                a=r.audio
                a=a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else np.asarray(a)
                if a is not None and a.size: n+=emit(a.astype(np.float32).reshape(-1))
            if (i+1)%200==0 or i+1==len(paras):
                el=time.time()-t0
                frac=(done_words + job["words"]*(i+1)/len(paras))/total_words
                print(f"[tts]   {i+1:,}/{len(paras):,} paragraphs | {(grand+n)/SR/60:.0f} min audio"
                      f" | {el/60:.0f} min elapsed | ~{el/max(frac,1e-9)*(1-frac)/60:.0f} min left",
                      flush=True)
        ok=True
    finally:
        try: proc.stdin.close()
        except Exception: pass
        rc=proc.wait()
        if not ok:
            try: os.remove(part)
            except OSError: pass
    if rc: sys.exit(f"ffmpeg exited {rc} on {job['mp3']}")
    os.replace(part, job["mp3"])
    if job.get("timings"):
        json.dump({"sample_rate": SR, "offsets": offsets},
                  open(job["timings"], "w", encoding="utf-8"))
    grand+=n; done_words+=job["words"]
print("MINUTES", round(grand/SR/60,2))
'''


OMNIVOICE_WORKER = r"""
import sys, os, re, json, time, subprocess
import numpy as np, torch, soundfile as sf, imageio_ffmpeg
from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

cfg   = json.load(open(sys.argv[1], encoding="utf-8"))
jobs  = cfg["jobs"]
SR    = 24000
BATCH = max(1, int(cfg["batch"]))
lang  = cfg["lang"] or None
speed = float(cfg["speed"])
speed = None if abs(speed - 1.0) < 1e-6 else speed
# Left alone OmniVoice reads at ~225 wpm, which is podcast pace, not narration.
# Asking for an explicit duration pulls it back; it bottoms out near 170 wpm, so
# a lower --wpm than that just gets ignored by the model.
WPM = float(cfg["wpm"])
SHORT = 8          # headings are too short for duration control to behave


def want(texts):
    if not WPM:
        return None
    return [len(t.split()) / WPM * 60 if len(t.split()) >= SHORT else None
            for t in texts]

# A neutral, well-punctuated line: long enough to be a good clone prompt, and
# free of anything the designed voice might latch onto as "character".
ANCHOR_TEXT = ("The evening light fell slowly across the room, and for a long "
               "moment nobody spoke at all.")

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(cfg["seed"])
print(f"[tts] device {dev}, seed {cfg['seed']}", flush=True)
model = OmniVoice.from_pretrained(cfg["model"], device_map=dev,
                                  dtype=torch.float16 if dev == "cuda" else torch.float32)
model.eval()

gcfg = OmniVoiceGenerationConfig()
if cfg["steps"]:
    gcfg.num_step = int(cfg["steps"])

if lang:
    ok = set(model.supported_language_ids()) | set(model.supported_language_names())
    if lang not in ok:
        near = sorted(x for x in ok if lang.lower() in str(x).lower())[:12]
        sys.exit(f"[error] --lang {lang!r} is not one of this model's "
                 f"{len(ok)} languages."
                 + (f" Did you mean: {', '.join(near)}?" if near else
                    " Run --engine omnivoice --list-voices to see them."))

# ---- the narrator -----------------------------------------------------------
# Voice design re-rolls a new voice on every call, so a book narrated chunk by
# chunk would wander between speakers.  Render the described voice once, then
# clone from that clip for the whole book.  The prompt is cached on disk, so
# reruns and per-chapter --split jobs keep the same narrator.
pf = cfg["prompt_file"]
if os.path.exists(pf):
    prompt = VoiceClonePrompt.load(pf)
    print(f"[tts] reusing voice prompt {os.path.basename(pf)}", flush=True)
elif cfg["ref_audio"]:
    rt = cfg["ref_text"] or None
    if rt is None:
        print("[tts] no --ref-text: transcribing the reference with ASR", flush=True)
        model.load_asr_model()
    prompt = model.create_voice_clone_prompt(cfg["ref_audio"], rt)
    prompt.save(pf)
    print(f"[tts] cloned {os.path.basename(cfg['ref_audio'])}", flush=True)
else:
    print(f"[tts] designing voice: {cfg['voice']!r}", flush=True)
    # pace the anchor too: a clone inherits some of its reference's speaking rate
    a = model.generate(text=ANCHOR_TEXT, instruct=cfg["voice"], language=lang,
                       duration=(want([ANCHOR_TEXT]) or [None])[0],
                       generation_config=gcfg)[0]
    sf.write(cfg["anchor"], np.asarray(a, dtype=np.float32), SR)
    prompt = model.create_voice_clone_prompt(cfg["anchor"], ANCHOR_TEXT)
    prompt.save(pf)
    print(f"[tts] anchor -> {os.path.basename(cfg['anchor'])} "
          f"({len(a)/SR:.1f}s); the whole book is cloned from it", flush=True)

# Spoken numbers/dates need WeTextProcessing, whose pynini dependency has no
# Windows wheel.  Ask once rather than crashing on the first paragraph.
from omnivoice.utils.text import normalize_text as _norm
try:
    _norm("Chapter 12, 1987.", lang)
    NORMALIZE = True
except Exception:
    NORMALIZE = False
    print("[tts] no WeTextProcessing: numbers and dates are passed through as "
          "written (pip install WeTextProcessing to fix)", flush=True)

ff = imageio_ffmpeg.get_ffmpeg_exe()
sil = lambda s: np.zeros(int(SR * s), dtype=np.float32)


def say(texts):
    # one batched call, falling back to one-at-a-time if the GPU says no.
    # A batch takes one duration per item and has no way to say "you decide"
    # for just one of them, so a batch holding a heading goes item by item.
    kw = dict(voice_clone_prompt=prompt, language=lang, speed=speed,
              normalize_text=NORMALIZE, generation_config=gcfg)
    d = want(texts)
    try:
        if d is None or all(x is not None for x in d):
            out = model.generate(text=texts, duration=d, **kw)
        else:
            out = [model.generate(text=t, duration=di, **kw)[0]
                   for t, di in zip(texts, d)]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        out = [model.generate(text=t, duration=(d[i] if d else None), **kw)[0]
               for i, t in enumerate(texts)]
    got = [np.asarray(a, dtype=np.float32).reshape(-1) for a in out]
    if len(got) != len(texts):        # never silently lose a paragraph
        sys.exit(f"[error] asked for {len(texts)} clips, got {len(got)}")
    return got


total_words = sum(j["words"] for j in jobs) or 1
grand = 0; done_words = 0; t0 = time.time()

for k, job in enumerate(jobs, 1):
    paras = [p.strip() for p in re.split(r"\n\s*\n",
             open(job["txt"], encoding="utf-8").read()) if p.strip()]
    part = job["mp3"] + ".part"
    proc = subprocess.Popen([ff, "-y", "-hide_banner", "-loglevel", "error",
                             "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                             "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
                             "-codec:a", "libmp3lame", "-b:a", "96k", "-ac", "1",
                             "-f", "mp3", part],   # .part hides the extension ffmpeg
                            # would otherwise pick the muxer from
                            stdin=subprocess.PIPE)

    def emit(a):
        proc.stdin.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        return a.size

    n = 0; done_p = 0; ok = False; offsets = []
    if len(jobs) > 1:
        print(f"[tts] ({k}/{len(jobs)}) {job['label']} - {job['words']:,} words", flush=True)
    try:
        for s in range(0, len(paras), BATCH):
            batch = paras[s:s + BATCH]
            for i, a in enumerate(say(batch)):
                if s + i:
                    n += emit(sil(0.6 if len(batch[i]) < 70 else 0.45))
                offsets.append(int(n))   # sample offset where this paragraph's audio starts
                if a.size:
                    n += emit(a)
            done_p += len(batch)
            el = time.time() - t0
            frac = (done_words + job["words"] * done_p / len(paras)) / total_words
            print(f"[tts]   {done_p:,}/{len(paras):,} paragraphs | {(grand+n)/SR/60:.0f} min audio"
                  f" | {el/60:.0f} min elapsed | ~{el/max(frac,1e-9)*(1-frac)/60:.0f} min left",
                  flush=True)
        ok = True
    finally:
        try: proc.stdin.close()
        except Exception: pass
        rc = proc.wait()
        if not ok:
            # half a book is not a finished chapter: leave nothing a rerun
            # would mistake for completed work
            try: os.remove(part)
            except OSError: pass
    if rc:
        sys.exit(f"ffmpeg exited {rc} on {job['mp3']}")
    os.replace(part, job["mp3"])
    if job.get("timings"):
        json.dump({"sample_rate": SR, "offsets": offsets},
                  open(job["timings"], "w", encoding="utf-8"))
    grand += n; done_words += job["words"]

print("MINUTES", round(grand / SR / 60, 2))
"""


# OmniVoice does not take a free-form voice description: --voice is a comma+space
# list drawn from this fixed vocabulary, at most one item per category.
OMNIVOICE_INSTRUCT = {
    "gender": ["female", "male"],
    "age":    ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch":  ["very low pitch", "low pitch", "moderate pitch",
               "high pitch", "very high pitch"],
    "accent": ["american accent", "australian accent", "british accent",
               "canadian accent", "chinese accent", "indian accent",
               "japanese accent", "korean accent", "portuguese accent",
               "russian accent"],
    "effect": ["whisper"],
}


def check_instruct(voice):
    """Catch a bad --voice here, before we spend a minute loading the model."""
    valid = {v: cat for cat, vs in OMNIVOICE_INSTRUCT.items() for v in vs}
    seen, bad = {}, []
    for item in [i.strip().lower() for i in voice.split(",") if i.strip()]:
        cat = valid.get(item)
        if cat is None:
            near = difflib.get_close_matches(item, valid, n=1, cutoff=0.6)
            bad.append(f"  {item!r} is not a voice attribute"
                       + (f" - did you mean {near[0]!r}?" if near else ""))
        elif cat in seen:
            bad.append(f"  {item!r} and {seen[cat]!r} are both {cat}; pick one")
        else:
            seen[cat] = item
    if bad:
        lines = ["[error] --engine omnivoice --voice takes a comma-separated list "
                 "of these attributes:"] + bad + [""]
        for cat, vs in OMNIVOICE_INSTRUCT.items():
            lines.append(f"  {cat:<7} {', '.join(vs)}")
        lines += ["", "  at most one per category, e.g. "
                      "--voice \"male, low pitch, middle-aged, american accent\"",
                  "  to use a real person's voice instead: --ref-audio clip.wav"]
        sys.exit(NEWLINE.join(lines))


KOKORO_VOICES = ["am_michael", "am_adam", "am_eric", "am_liam", "am_onyx",
                 "af_heart", "af_bella", "af_nicole", "af_sarah",
                 "bm_george", "bm_lewis", "bf_emma", "bf_isabella"]


def list_voices(args):
    """Print whatever voices the chosen engine offers, then exit."""
    if args.engine == "kokoro":
        print("Kokoro voices (lang_code 'a'):")
        for v in KOKORO_VOICES:
            print("   ", v)
        return
    if args.engine == "omnivoice":
        print("OmniVoice builds a voice out of these attributes.  Pass them to")
        print("--voice as a comma-separated list, at most one per category:")
        print()
        for cat, vs in OMNIVOICE_INSTRUCT.items():
            print(f"  {cat:<7} {', '.join(vs)}")
        print()
        print("  default:  male, low pitch, middle-aged, american accent")
        print("  example:  --voice \"female, elderly, moderate pitch, british accent\"")
        print()
        print("To use a real voice instead, clone it:")
        print("  --ref-audio clip.wav --ref-text \"exactly what the clip says\"")
        print("  (--ref-text is optional; OmniVoice will transcribe the clip itself)")


def safe_name(s):
    """Make a label usable as a Windows filename."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s)
    return re.sub(r"\s+", " ", s).strip(" .")[:60] or "section"


def narrate(jobs, args):
    """jobs: [{'txt':..., 'mp3':..., 'label':..., 'words':...}] - all narrated by a
    single worker process so the model only loads once."""
    todo = []
    for j in jobs:
        if os.path.exists(j["mp3"]) and os.path.getsize(j["mp3"]) > 1024:
            print(f"[skip] already done: {os.path.basename(j['mp3'])}")
        else:
            todo.append(j)
    if not todo:
        print("[tts] every file already exists - nothing to do")
        return

    # Each job writes a paragraph -> sample-offset map alongside its output.
    # Single audiobook: <outdir>/timings.json (beside transcript.txt); EPUB --split:
    # one <mp3-stem>.timings.json per section so they never collide.
    single = len(jobs) == 1
    for j in todo:
        j["timings"] = (os.path.join(args.outdir, "timings.json") if single
                        else os.path.splitext(j["mp3"])[0] + ".timings.json")

    cfg = {"jobs": todo, "voice": args.voice, "speed": args.speed}
    if args.engine == "kokoro":
        py, worker = KVENV, KOKORO_WORKER
        print(f"[tts] Kokoro voice '{args.voice}' -> {len(todo)} file(s) "
              f"(this is the slow step)")
    else:
        py, worker = OVENV, OMNIVOICE_WORKER
        cfg.update(model=OMODEL, lang=args.lang, seed=args.seed, steps=args.steps,
                   batch=args.batch, wpm=args.wpm,
                   ref_audio=os.path.abspath(args.ref_audio) if args.ref_audio else "",
                   ref_text=args.ref_text,
                   anchor=args.anchor or os.path.join(args.outdir, "voice_anchor.wav"),
                   prompt_file=os.path.join(args.outdir, "voice_prompt.omni"))
        how = (f"cloning {os.path.basename(cfg['ref_audio'])}" if cfg["ref_audio"]
               else f"voice {args.voice!r}")
        print(f"[tts] OmniVoice, {how} -> {len(todo)} file(s) "
              f"(this is the slow step)")
    mf = os.path.join(args.outdir, "_jobs.json")
    with open(mf, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    # Hand the worker UTF-8 bytes, not text: text=True would encode through the
    # console locale (cp1252 here) and any non-ASCII in the worker source comes
    # back as a SyntaxError inside the child.
    sh([py, "-", mf], input=worker.encode("utf-8"))
    os.remove(mf)


# ---------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):     # console is cp1252 here: don't die on
        try:                                    # a Persian / CJK / Greek chapter title
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Convert a document (PDF, EPUB, Markdown, HTML, DOCX, text) "
                    "into an MP3 audiobook.")
    ap.add_argument("src", metavar="FILE", nargs="?",
                    help="pdf / epub / md / html / docx / txt")
    ap.add_argument("--pages", default="",
                    help="subset: PDF page numbers, or EPUB section numbers "
                         "(0-based), e.g. '0-9' or '0,2,5' (default all)")
    ap.add_argument("--engine", default="kokoro",
                    choices=["kokoro", "omnivoice"],
                    help="TTS engine: 'kokoro' (fast, English, fixed voices) or "
                         "'omnivoice' (k2-fsa OmniVoice: 600+ languages, voice "
                         "design from a description, zero-shot cloning)")
    ap.add_argument("--voice", default="",
                    help="kokoro: am_michael, af_heart, bm_george, bf_emma, ... ; "
                         "omnivoice: comma-separated voice attributes, e.g. "
                         "'male, low pitch, middle-aged, american accent'. "
                         "--list-voices prints the set for the chosen engine")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lang", default="en",
                    help="omnivoice: speech language, a name or code from its 600+ "
                         "('en', 'English', 'de', ...)")
    ap.add_argument("--ref-audio", default="",
                    help="omnivoice: WAV/MP3 of a voice to clone instead of "
                         "designing one from --voice")
    ap.add_argument("--ref-text", default="",
                    help="omnivoice: exact transcript of --ref-audio; omit it and "
                         "OmniVoice will transcribe the clip itself")
    ap.add_argument("--seed", type=int, default=1234,
                    help="omnivoice: fixed sampling seed - keeps the designed voice "
                         "identical across every chunk of the book")
    ap.add_argument("--steps", type=int, default=64,
                    help="omnivoice: diffusion steps (0 = the model's own default of "
                         "32). 64 is the default here: below it the read flattens out "
                         "- stresses land in the wrong place and word-to-word "
                         "transitions get abrupt. Time scales linearly, so 64 costs "
                         "about twice 32")
    ap.add_argument("--wpm", type=float, default=165,
                    help="omnivoice: target narration pace in words per minute "
                         "(default 165; left alone the model reads at ~225, which "
                         "is too fast for a book). 0 turns pacing off. Below about "
                         "170 the model stops obeying, so treat it as a nudge")
    ap.add_argument("--batch", type=int, default=1,
                    help="omnivoice: paragraphs per GPU call. 1 is both the fastest "
                         "and the leanest here - measured whole-job, batch 4 ran 4%% "
                         "slower than batch 1 for 40%% more VRAM. Diffusion already "
                         "parallelises across the utterance, so a batch only adds "
                         "padding waste: every item runs as long as the longest")
    ap.add_argument("--anchor", default="",
                    help="omnivoice: reuse/write this WAV as the voice anchor - the "
                         "designed voice is rendered once, then cloned for the whole "
                         "book so the narrator never drifts "
                         "(default: <outdir>/voice_anchor.wav)")
    ap.add_argument("--list-voices", action="store_true",
                    help="print the engine's voice/style list and exit")
    ap.add_argument("--layout", default="auto", choices=["auto", "single", "side-by-side"],
                    help="scanned-PDF open-book layout (auto detects by page aspect)")
    ap.add_argument("--ocr", default="paddle", choices=["paddle", "tesseract"])
    ap.add_argument("--outdir", default="", help="output folder (default <file>_audiobook)")
    ap.add_argument("--force-scanned", action="store_true", help="PDF only: skip detection, always OCR")
    ap.add_argument("--read-code", action="store_true",
                    help="markdown only: narrate fenced code blocks instead of skipping them")
    ap.add_argument("--list", action="store_true", help="EPUB only: list sections and exit")
    ap.add_argument("--split", action="store_true",
                    help="EPUB only: one MP3 per section instead of a single long file "
                         "(rerunning skips sections already encoded)")
    ap.add_argument("--transcript-only", action="store_true",
                    help="write transcript.txt and stop (check it before the slow TTS step)")
    args = ap.parse_args()
    if not args.voice:
        args.voice = {"kokoro": "am_michael",
                      "omnivoice": "male, low pitch, middle-aged, american accent",
                      }[args.engine]
    if args.ref_text and not args.ref_audio:
        print("[note] --ref-text does nothing without --ref-audio; ignoring")
    if args.engine == "omnivoice" and not args.ref_audio:
        check_instruct(args.voice)
    if args.list_voices:
        return list_voices(args)
    if args.ref_audio and args.engine == "kokoro":
        print("[note] --ref-audio needs --engine omnivoice; ignoring")
    if args.engine == "kokoro" and args.lang != "en":
        print("[note] --lang does not apply to Kokoro; ignoring")

    if not args.src:
        ap.error("a FILE is required (except with --list-voices)")
    src = os.path.abspath(args.src)
    if not os.path.exists(src):
        sys.exit(f"not found: {src}")
    ext = os.path.splitext(src)[1].lower()

    if args.list:
        if ext not in EPUB_EXT:
            sys.exit("--list only applies to .epub files")
        list_epub(src)
        return

    stem = os.path.splitext(os.path.basename(src))[0].strip("-_ ") or "book"
    args.outdir = os.path.abspath(args.outdir or f"{stem}_audiobook")
    os.makedirs(args.outdir, exist_ok=True)

    if ext not in PDF_EXT and (args.force_scanned or args.ocr != "paddle" or args.layout != "auto"):
        print("[note] --layout / --ocr / --force-scanned apply to PDFs only; ignoring")
    if args.split and ext not in EPUB_EXT:
        print("[note] --split needs an EPUB (sections come from the spine); writing one file")
        args.split = False

    if args.split:
        return run_split_epub(src, args)

    if ext in PDF_EXT:
        transcript = run_pdf(src, args)
    elif ext in EPUB_EXT:
        transcript = read_epub(src, args.pages)
    elif ext in MD_EXT:
        transcript = finalize(md_to_paragraphs(read_text_file(src), args.read_code))
    elif ext in HTML_EXT:
        paras, _ = html_to_paragraphs(read_text_file(src))
        transcript = finalize(paras)
    elif ext in DOCX_EXT:
        transcript = finalize(docx_to_paragraphs(src))
    elif ext in (".doc", ".rtf", ".odt", ".pages"):
        sys.exit(f"[error] {ext} isn't supported - re-save it as .docx, .pdf, .md or .txt first.")
    else:
        if ext not in (".txt", ".text", ".log", ".rst", ".org", ""):
            print(f"[note] unknown extension '{ext}'; reading it as plain text")
        transcript = finalize(plain_to_paragraphs(read_text_file(src)))

    tp = os.path.join(args.outdir, "transcript.txt")
    open(tp, "w", encoding="utf-8").write(transcript + "\n")
    words = len(transcript.split())
    print(f"[transcript] {words:,} words (~{words / 155:.0f} min narrated) -> {tp}")
    letters = re.findall(r"[^\W\d_]", transcript)
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.5:
        if args.engine == "kokoro":
            print("[note] this text is mostly non-English - the Kokoro voices wired up "
                  "here are English (lang_code 'a') and will mispronounce it. "
                  "--engine omnivoice --lang <code> speaks 600+ languages.")
        elif args.engine == "omnivoice" and args.lang == "en":
            print("[note] this text is mostly non-English but --lang is 'en'; "
                  "pass the right --lang or it will be mispronounced")
    if words < 20:
        sys.exit("[error] transcript is nearly empty - empty source, wrong --pages range,"
                 " or (scanned PDFs) the wrong --layout / failed OCR.")
    if args.transcript_only:
        print("\nDONE (transcript only) -> " + tp)
        return

    out_mp3 = os.path.join(args.outdir, f"{stem}.mp3")
    narrate([{"txt": tp, "mp3": out_mp3, "label": stem, "words": words}], args)
    print(f"\nDONE -> {out_mp3}")
    print(f"      transcript: {tp}")


def run_split_epub(src, args):
    """One MP3 per EPUB section, resumable across runs."""
    secs = epub_sections(src)
    if not secs:
        sys.exit("[error] no readable chapters found in that EPUB.")
    keep = parse_pages(args.pages, len(secs))
    tdir = os.path.join(args.outdir, "transcripts")
    os.makedirs(tdir, exist_ok=True)

    jobs, skipped, whole = [], [], []
    for i in keep:
        label, paras = secs[i]
        text = finalize(paras)
        words = len(text.split())
        whole.append(text)
        if words < 20:                      # covers, title pages, blank sections
            skipped.append(f"{i} {label}")
            continue
        base = f"{i:02d} - {safe_name(label)}"
        tp = os.path.join(tdir, base + ".txt")
        with open(tp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        jobs.append({"txt": tp, "mp3": os.path.join(args.outdir, base + ".mp3"),
                     "label": label, "words": words})

    full = os.path.join(args.outdir, "transcript.txt")
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(t for t in whole if t) + "\n")
    total = sum(j["words"] for j in jobs)
    print(f"[epub] {len(secs)} section(s), {len(jobs)} to narrate, {total:,} words "
          f"(~{total / 155 / 60:.1f} h) -> {args.outdir}")
    if skipped:
        print(f"[epub] too short to narrate, skipped: {', '.join(skipped)}")
    if not jobs:
        sys.exit("[error] nothing substantial to narrate - check --pages.")
    if args.transcript_only:
        print(f"\nDONE (transcript only) -> {tdir}")
        return

    narrate(jobs, args)
    print(f"\nDONE -> {len(jobs)} MP3(s) in {args.outdir}")


if __name__ == "__main__":
    main()
