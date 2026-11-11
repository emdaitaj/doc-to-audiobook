#!/usr/bin/env python3
"""
book_deskew_split.py
====================

Process a PDF that is a scan of an open book where every PDF page contains TWO
book pages side by side (a left page and a right page) with a gutter in the
middle and some rotation/skew introduced by the scanning process.

For every input page the tool:

  1. Renders the page to a high-resolution raster (respecting any /Rotate flag
     so the content comes out upright).
  2. Finds the vertical "cutting line" that separates the left page from the
     right page (the gutter) and splits the image into a left half and a right
     half.  A straight top-to-bottom line is used, placed on the whitest
     central column (falls back to the exact middle).
  3. Detects the skew angle of the text in each half independently and rotates
     the half so the text becomes horizontal/straight.
  4. (optionally) trims black scanner borders / excess white margin.

The processed single pages are written to a new PDF (one book page per output
page), in reading order: left page first, then right page.

Usage
-----
    python book_deskew_split.py "C:/path/book.pdf"
    python book_deskew_split.py "book.pdf" -o "book_clean.pdf" --dpi 250
    python book_deskew_split.py "book.pdf" --pages 0-4 --debug-dir ./debug

Dependencies: PyMuPDF (fitz), opencv-python, numpy, Pillow
    pip install PyMuPDF opencv-python numpy Pillow

Part of doc-to-audiobook - https://github.com/emdaitaj/doc-to-audiobook
Copyright (C) 2026 emdaitaj. AGPL-3.0 with attribution terms (see NOTICE).
"""

import io
import os
import sys
import argparse
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import fitz  # PyMuPDF
    import cv2
    from PIL import Image
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "Missing dependency. Install with:\n"
        "    pip install PyMuPDF opencv-python numpy Pillow\n"
        f"Import error: {e}\n"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("book_deskew_split")


# Default location of the sample book used while developing / debugging.


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def native_zoom(page: "fitz.Page", cap: float = 8.0) -> float:
    """
    Zoom factor that renders the page at the native resolution of its embedded
    scan, so we never downsample.

    Uses an area-based pixel density (native_pixels / displayed_points) which is
    invariant to page /Rotate and to scans stored as multiple horizontal stripes
    (this book stores each scan as 17 stripes).  Returns the max density over all
    placed images, capped so a stray tiny high-res glyph can't explode the size.
    """
    best = 0.0
    for im in page.get_image_info(xrefs=True):
        bbox = fitz.Rect(im["bbox"])
        bw, bh = bbox.width, bbox.height
        if bw > 1 and bh > 1:
            best = max(best, (im["width"] * im["height"] / (bw * bh)) ** 0.5)+150
    if best <= 0:
        best = 200.0 / 72.0  # no raster found -> sensible default
    return min(best, cap)


def page_zoom(page: "fitz.Page", dpi) -> float:
    """Resolve the render zoom: native (dpi='auto') or a fixed dpi number."""
    if dpi == "auto":
        return native_zoom(page)
    return float(dpi) / 72.0


def render_page_gray(page: "fitz.Page", zoom: float) -> np.ndarray:
    """Render a PDF page to a grayscale numpy array (H, W), upright."""
    # alpha=False -> no alpha channel; PyMuPDF honours /Rotate by default so the
    # rendered pixmap is already in the visually-upright orientation.
    pm = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    return img[:, :, 0].copy()


# --------------------------------------------------------------------------- #
#  Gutter detection / vertical split
# --------------------------------------------------------------------------- #
def find_gutter_x(gray: np.ndarray, search_frac: float = 0.18,
                  center_bias: float = 0.5) -> int:
    """
    Find the x coordinate of the vertical cutting line between the two pages.

    The gutter is a low-ink vertical band near the centre of the scan.  The
    catch: an index/table-of-contents page can have an *internal* whitespace
    column (between entry text and its page-number column) that is just as empty
    as the true gutter, so a plain "lowest-ink column" search can cut inside a
    page and move part of one page onto the other.

    To avoid that we score each column by its ink *plus* a gentle penalty for
    distance from centre.  Among similarly-empty valleys (true gutter, index gap,
    outer margins) this prefers the one nearest the centre - the real gutter -
    while a genuinely deeper valley can still win if a page is scanned off-centre.
    Falls back to the exact centre when no convincing gutter exists (e.g. a
    full-bleed cover).
    """
    h, w = gray.shape
    # Binarise: ink (dark) -> 1
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Ignore the top/bottom 8% so headers/footers/page numbers don't dominate.
    y0, y1 = int(h * 0.08), int(h * 0.92)
    col_ink = ink[y0:y1, :].sum(axis=0).astype(np.float64)

    # Smooth the profile so we pick a wide whitespace valley, not a 1-px fluke.
    k = max(3, int(w * 0.01) | 1)  # odd kernel ~1% of width
    kernel = np.ones(k, dtype=np.float64) / k
    col_ink = np.convolve(col_ink, kernel, mode="same")

    centre = w // 2
    half_win = int(w * search_frac)
    lo, hi = max(1, centre - half_win), min(w - 1, centre + half_win)
    window = col_ink[lo:hi]
    if window.size == 0:
        return centre

    page_median = float(np.median(col_ink[col_ink > 0])) if np.any(col_ink > 0) else 0.0

    # Add a centre-distance penalty (scaled by the page's typical ink) so ties
    # between equally-empty columns resolve to the most central one.
    xs = np.arange(lo, hi)
    penalty = center_bias * page_median * (np.abs(xs - centre) / (w / 2.0))
    cut = lo + int(np.argmin(window + penalty))

    # Sanity: the chosen column should genuinely be low-ink relative to the page.
    if page_median > 0 and col_ink[cut] > 0.6 * page_median:
        # No clear gutter (e.g. a full-bleed image/cover) -> use the middle.
        return centre
    return cut


# --------------------------------------------------------------------------- #
#  Skew detection (projection-profile variance)
# --------------------------------------------------------------------------- #
def _projection_variance(binary: np.ndarray, angle: float) -> float:
    """Rotate a binary image by `angle` and return variance of the row sums."""
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    proj = rot.sum(axis=1, dtype=np.float64)
    return float(np.var(proj))


def _weighted_median(values, weights) -> float:
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cum = np.cumsum(weights)
    cutoff = cum[-1] / 2.0
    return float(values[np.searchsorted(cum, cutoff)])


def estimate_text_line_angle(gray: np.ndarray, max_skew: float):
    """
    Robustly estimate the skew angle from genuine *text lines* only.

    Words are merged into line blobs with a horizontal dilation, then only
    components that look like a line of text (clearly elongated horizontally,
    a meaningful fraction of the page wide, and not too tall) are kept.  Each
    surviving line contributes its principal-axis angle, weighted by length.

    This deliberately ignores the steep diagonal trend-lines and large picture
    blobs found on figure pages, which would otherwise fool a plain
    projection-profile estimator.

    Returns (coarse_angle, reliable).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = binary.shape
    scale = 1600.0 / max(h, w)
    if scale < 1.0:
        binary = cv2.resize(binary, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST)
        h, w = binary.shape

    if binary.mean() < 0.5:                      # essentially blank
        return 0.0, False

    # Merge characters/words within a line into a single horizontal blob.
    kx = max(15, int(w * 0.03))
    merged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.10 * w:        # too short to trust as a text line
            continue
        if bh > 0.05 * h:        # too tall -> a figure / block, not a line
            continue
        if bw < 4 * bh:          # not clearly horizontal-elongated
            continue
        vx, vy, _, _ = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        ang = np.degrees(np.arctan2(vy, vx))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        if abs(ang) <= max_skew:     # ignore steep diagram lines
            angles.append(ang)
            weights.append(float(bw))

    if len(angles) < 3:
        return 0.0, False
    return _weighted_median(angles, weights), True


def detect_skew_angle(gray: np.ndarray, max_skew: float = 6.0) -> float:
    """
    Detect the skew of text in a grayscale image.

    Two stages: (1) a text-line component estimate that is robust on figure
    pages and decides whether a reliable text orientation exists, and (2) a fine
    projection-profile variance sweep around that estimate for sub-degree
    accuracy.  Returns the CCW-positive angle to apply to straighten the text;
    0.0 when no reliable text orientation is found (blank / figure-only pages).
    """
    coarse, reliable = estimate_text_line_angle(gray, max_skew)
    if not reliable:
        return 0.0

    # Fine refinement via projection variance, on a downscaled binary.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = binary.shape
    scale = 1200.0 / max(h, w)
    if scale < 1.0:
        binary = cv2.resize(binary, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST)
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))

    fine = np.arange(coarse - 1.0, coarse + 1.0 + 0.01, 0.05)
    best = max(((_projection_variance(binary, a), a) for a in fine),
               key=lambda s: s[0])[1]
    # Never let the refinement run away from the robust coarse estimate.
    if abs(best - coarse) > 1.0:
        best = coarse
    return float(best)


# --------------------------------------------------------------------------- #
#  Rotation / cropping
# --------------------------------------------------------------------------- #
def rotate_keep_white(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate `gray` by `angle` (CCW deg), expanding the canvas, white fill."""
    if abs(angle) < 0.02:
        return gray
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += (nw / 2.0) - cx
    m[1, 2] += (nh / 2.0) - cy
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def trim_borders(gray: np.ndarray, pad: int = 12) -> np.ndarray:
    """Trim surrounding black scanner edge and excess white margin."""
    # First, blank out near-black scanner frame by flooding from the edges is
    # overkill; instead crop to the bounding box of the *content* (ink), which
    # naturally drops both black frames and wide white margins.
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Remove tiny speckles before measuring the content box.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    ys, xs = np.where(ink > 0)
    if xs.size == 0 or ys.size == 0:
        return gray
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = gray.shape
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    return gray[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
#  Per-page processing
# --------------------------------------------------------------------------- #
def process_page(gray: np.ndarray, layout: str, gutter_mode: str,
                 deskew: bool, crop: bool, max_skew: float):
    """Split one rendered scan into its two book pages and straighten each.

    Returns a list of (image, info_dict) in reading order.
    """
    h, w = gray.shape
    if layout == "side-by-side":
        if gutter_mode == "detect":
            cut = find_gutter_x(gray)
        else:
            cut = w // 2
        halves = [("L", gray[:, :cut]), ("R", gray[:, cut:])]
    elif layout == "stacked":
        cut = h // 2  # top/bottom split
        halves = [("T", gray[:cut, :]), ("B", gray[cut:, :])]
    else:  # single page, just deskew
        halves = [("S", gray)]

    out = []
    for tag, half in halves:
        angle = detect_skew_angle(half, max_skew) if deskew else 0.0
        straight = rotate_keep_white(half, angle) if deskew else half
        if crop:
            straight = trim_borders(straight)
        out.append((straight, {"side": tag, "angle": angle,
                               "size": (straight.shape[1], straight.shape[0])}))
    return out


# --------------------------------------------------------------------------- #
#  PDF assembly
# --------------------------------------------------------------------------- #
def encode_page(gray: np.ndarray, mode: str, jpeg_quality: int):
    """Encode one page image to compressed bytes. Runs inside worker threads so
    the (GIL-releasing) compression is parallelised, not serialised in the
    writer.

    mode 'bw'   -> 1-bit black/white PNG (tiny files, matches the bilevel scan).
    mode 'gray' -> 8-bit grayscale JPEG (preserves anti-aliased edges).

    Returns (img_bytes, width, height).
    """
    if mode == "bw":
        bw = (gray > 128).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(bw).convert("1").save(buf, format="PNG")
        img_bytes = buf.getvalue()
    else:
        ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        img_bytes = enc.tobytes()
    h, w = gray.shape
    return img_bytes, w, h


def insert_encoded(doc_out: "fitz.Document", img_bytes: bytes,
                   w: int, h: int, out_dpi: float):
    """Append a pre-encoded image as a new page sized to print at `out_dpi`."""
    pw, ph = w * 72.0 / out_dpi, h * 72.0 / out_dpi
    page = doc_out.new_page(width=pw, height=ph)
    page.insert_image(fitz.Rect(0, 0, pw, ph), stream=img_bytes)


# --------------------------------------------------------------------------- #
#  Parallel worker
# --------------------------------------------------------------------------- #
# PyMuPDF Documents are not safe to share across threads, so every worker thread
# opens (and reuses) its own handle to the source file.
_thread_local = threading.local()
_open_docs = []
_open_docs_lock = threading.Lock()


def _thread_doc(path: str) -> "fitz.Document":
    doc = getattr(_thread_local, "doc", None)
    if doc is None:
        doc = fitz.open(path)
        _thread_local.doc = doc
        with _open_docs_lock:
            _open_docs.append(doc)
    return doc


def render_and_process(path: str, idx: int, args, debug_dir):
    """Thread task: render one source page at native res, split+deskew each
    half, and encode the results.  Doing render/deskew/encode here keeps the
    main writer thread doing only the cheap, ordered PDF insertion.

    Returns (idx, [(img_bytes, w, h, info), ...], out_dpi).
    """
    page = _thread_doc(path)[idx]
    zoom = page_zoom(page, args.dpi)
    gray = render_page_gray(page, zoom)
    results = process_page(
        gray, args.layout, args.gutter,
        deskew=not args.no_deskew, crop=args.crop, max_skew=args.max_skew,
    )
    encoded = []
    for img, info in results:
        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"p{idx:03d}_{info['side']}.png"), img)
        data, w, h = encode_page(img, args.mode, args.jpeg_quality)
        encoded.append((data, w, h, info))
    return idx, encoded, zoom * 72.0


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def parse_pages(spec: str, total: int):
    if not spec:
        return list(range(total))
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        elif part:
            pages.add(int(part))
    return sorted(p for p in pages if 0 <= p < total)


def run(args):
    in_path = Path(args.input)
    if not in_path.exists():
        log.error("Input not found: %s", in_path)
        return 1

    out_path = Path(args.output) if args.output else \
        in_path.with_name(f"{in_path.stem}_split_deskew.pdf")

    src = fitz.open(str(in_path))
    total = src.page_count
    src.close()  # workers each open their own handle
    page_indices = parse_pages(args.pages, total)

    workers = max(1, args.workers)
    log.info("Input: %s (%d pages) -> processing %d page(s)",
             in_path.name, total, len(page_indices))
    log.info("dpi=%s layout=%s gutter=%s deskew=%s crop=%s mode=%s workers=%d",
             args.dpi, args.layout, args.gutter, not args.no_deskew,
             args.crop, args.mode, workers)

    out = fitz.open()
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_out = 0
    n_in = len(page_indices)
    window = workers * 2  # bound in-flight pages so native-res rasters don't pile up

    # Render+deskew pages concurrently in worker threads, but write them to the
    # output PDF strictly in order and with a bounded look-ahead window.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        submit_pos = 0

        def submit(pos):
            futures[pos] = ex.submit(
                render_and_process, str(in_path), page_indices[pos], args, debug_dir)

        while submit_pos < min(window, n_in):
            submit(submit_pos); submit_pos += 1

        for write_pos in range(n_in):
            idx, encoded, out_dpi = futures.pop(write_pos).result()
            angles = ", ".join(f"{e[3]['side']}={e[3]['angle']:+.2f}deg" for e in encoded)
            log.info("page %d -> %d part(s)  [%s]", idx, len(encoded), angles)

            for data, w, h, info in encoded:
                insert_encoded(out, data, w, h, out_dpi)
                n_out += 1

            if submit_pos < n_in:
                submit(submit_pos); submit_pos += 1

            n_done += 1
            if n_done % 10 == 0:
                log.info("  ...%d/%d input pages done", n_done, n_in)

    out.save(str(out_path), deflate=True, garbage=4)
    out.close()
    with _open_docs_lock:
        for d in _open_docs:
            d.close()
        _open_docs.clear()
    log.info("Done. Wrote %d output page(s) -> %s", n_out, out_path)
    if debug_dir:
        log.info("Debug images in: %s", debug_dir)
    return 0


def dpi_arg(v):
    """argparse type: accept 'auto' or a positive integer DPI."""
    if str(v).lower() == "auto":
        return "auto"
    iv = int(v)
    if iv <= 0:
        raise argparse.ArgumentTypeError("dpi must be 'auto' or a positive integer")
    return iv


def build_parser():
    p = argparse.ArgumentParser(
        description="Split a side-by-side scanned-book PDF into single pages "
                    "and deskew (straighten) each page.")
    p.add_argument("input", help="Source PDF.")
    p.add_argument("-o", "--output", default=None, help="Output PDF path.")
    p.add_argument("--dpi", type=dpi_arg, default="auto",
                   help="Render resolution: 'auto' = native scan resolution "
                        "(no downsampling, default), or a number e.g. 300.")
    p.add_argument("-j", "--workers", type=int,
                   default=min(8, (os.cpu_count() or 4)),
                   help="Parallel worker threads (default min(8, CPUs)).")
    p.add_argument("--layout", choices=["side-by-side", "stacked", "single"],
                   default="side-by-side",
                   help="Scan layout (default side-by-side = left/right).")
    p.add_argument("--gutter", choices=["detect", "center"], default="detect",
                   help="Where to place the vertical cut (default detect).")
    p.add_argument("--max-skew", type=float, default=6.0,
                   help="Max skew searched, in degrees (default 6).")
    p.add_argument("--no-deskew", action="store_true",
                   help="Split only, do not straighten.")
    p.add_argument("--crop", action="store_true",
                   help="Trim black scanner borders / white margins.")
    p.add_argument("--mode", choices=["bw", "gray"], default="bw",
                   help="Output colour: 'bw' 1-bit (small, matches scan) or "
                        "'gray' JPEG (default bw).")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="Output JPEG quality 1-100, gray mode only (default 85).")
    p.add_argument("--pages", default="",
                   help="Subset, e.g. '0-4' or '0,2,5' (default all).")
    p.add_argument("--debug-dir", default=None,
                   help="Also write per-page PNGs here for inspection.")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
