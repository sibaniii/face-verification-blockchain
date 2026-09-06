"""
Step 3 of the pipeline: independent face-match verification.

ARCHITECTURE NOTE (important distinction):
    Step 2 (search_stage.py) performs DISCOVERY -- a web/visual search
    engine says "this URL looks visually similar." That is not proof
    of identity.

    This module performs VERIFICATION -- it independently downloads
    the images actually found on that candidate page and compares
    their face embeddings against the query image's own embedding,
    using the same face_recognition/dlib approach as Step 1. Only this
    module's result may be described as a "face match." A page ranked
    highly by the search engine (e.g. an Instagram result) is treated
    exactly the same as any other candidate: it must pass this
    independent check, or it is reported as NO MATCH.

Pipeline:
    query image
        -> process_face() [reused from face_stage.py, Step 1]
        -> query face encoding
    candidate_url (read from data/output/search_result.json)
        -> fetch webpage (requests)
        -> extract image URLs (BeautifulSoup)
        -> download each image
        -> detect ALL faces in each image (0, 1, or many)
        -> compare every detected face against the query encoding
        -> keep the smallest distance seen across all images
        -> match_found = (best distance <= FACE_DISTANCE_THRESHOLD)

No raw face embeddings, and no API keys, are ever printed or saved.
"""

import os
import re
import sys
import io
import json
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import face_recognition

from src.face_stage import process_face, FaceStageError

# A face_recognition euclidean distance below this is considered a match.
# 0.6 is the conservative default commonly used with this library.
FACE_DISTANCE_THRESHOLD = 0.6

# Don't try to verify against an unbounded number of images on one page.
# Deliberately generous: it's far better to extract 20-50 candidates and
# let face detection eliminate irrelevant ones than to risk a tight
# filter accidentally excluding the real photo (see extract_image_urls).
MAX_CANDIDATE_IMAGES = 30

# Matches url(...) inside a CSS background-image declaration, whether it
# appears in an inline style="" attribute or inside a <style> block.
BACKGROUND_IMAGE_PATTERN = re.compile(r'url\(\s*[\'"]?([^\'")]+)[\'"]?\s*\)', re.IGNORECASE)

# Light heuristic fallback: image-looking URLs embedded in <script>
# blocks (e.g. a carousel's JSON config or page data). Not a substitute
# for the structural parsing above -- just an extra net.
SCRIPT_IMAGE_URL_PATTERN = re.compile(
    r'https?://[^\s\'"<>]+?\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s\'"<>]*)?',
    re.IGNORECASE,
)

# Don't download unbounded content for a single image.
MAX_IMAGE_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
REQUEST_TIMEOUT_SECONDS = 15

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (hackathon-face-verify-bot)"}

DEFAULT_SEARCH_RESULT_PATH = os.path.join("data", "output", "search_result.json")
DEFAULT_VERIFICATION_OUTPUT_PATH = os.path.join("data", "output", "verification_result.json")
DEFAULT_VERIFICATION_IMAGES_DIR = os.path.join("data", "output", "verification_images")


class VerifyMatchError(Exception):
    """Base exception for all verify_match errors."""


class SearchResultMissingError(VerifyMatchError):
    """Raised when data/output/search_result.json is missing or unreadable."""


class CandidateMissingError(VerifyMatchError):
    """Raised when search_result.json has no usable selected_candidate.url."""


class WebpageFetchError(VerifyMatchError):
    """Raised when the candidate webpage itself cannot be fetched."""


# --------------------------------------------------------------------------
# Step 2 output loading
# --------------------------------------------------------------------------

def load_search_result(path=DEFAULT_SEARCH_RESULT_PATH):
    """Read the candidate URL that Step 2 discovered. Never invented,
    never asked for interactively -- it must come from search_stage's
    saved output."""
    if not os.path.isfile(path):
        raise SearchResultMissingError(
            f"{path} not found. Run Step 2 (python -m src.search_stage <image>) first."
        )

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchResultMissingError(f"Could not read {path}: {exc}") from exc

    candidate = data.get("selected_candidate") if isinstance(data, dict) else None
    if not isinstance(candidate, dict) or not candidate.get("url"):
        raise CandidateMissingError(
            f"{path} does not contain a selected_candidate with a url. "
            "Re-run Step 2 and select a candidate."
        )

    return candidate["url"]


# --------------------------------------------------------------------------
# Webpage fetching and image-URL extraction
# --------------------------------------------------------------------------

def fetch_webpage(url, timeout=REQUEST_TIMEOUT_SECONDS):
    try:
        response = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
    except requests.exceptions.RequestException as exc:
        raise WebpageFetchError(
            f"Could not fetch candidate webpage ({type(exc).__name__})."
        ) from exc

    if response.status_code != 200:
        raise WebpageFetchError(
            f"Candidate webpage returned HTTP {response.status_code}."
        )

    return response.text


def _parse_srcset(value):
    """Split a srcset/data-srcset attribute ('url1 1x, url2 2x, ...')
    into its individual candidate URLs."""
    urls = []
    if not value:
        return urls
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if tokens:
            urls.append(tokens[0])
    return urls


def extract_image_urls(html, base_url, max_images=MAX_CANDIDATE_IMAGES):
    """
    Pull candidate image URLs out of a webpage, covering the range of
    ways modern sites (lazy-loading libraries, carousels, WordPress
    page builders) represent images -- not just plain <img src>:

        - og:image / twitter:image meta tags
        - <img src>
        - common lazy-load attributes: data-src, data-lazy-src,
          data-original
        - srcset / data-srcset on <img> and <source> (all candidate
          URLs in the set, not just the first)
        - <picture><source srcset=...></picture>
        - CSS background-image: url(...) in inline style="" attributes
          and in <style> blocks
        - common page-builder background-image data attributes:
          data-bg, data-background, data-background-image
        - a light fallback sweep of image-looking URLs (.jpg/.png/etc.)
          inside <script> blocks, for content loaded from embedded
          JSON/config rather than markup

    Design choice: filtering here is intentionally light. We reject
    non-http(s) schemes (data:, javascript:, file:) and .svg files
    (a UI icon/logo format that is essentially never a person's
    photo), but we do NOT filter by filename patterns like "logo" or
    "icon" -- a real photo can have a generic filename, and it is far
    safer to over-collect candidates and let face detection eliminate
    irrelevant ones than to risk silently excluding the real photo.

    Resolves relative URLs against base_url and deduplicates while
    preserving order.

    IMPORTANT LIMITATION: this only sees what a plain HTTP GET
    receives. If a page injects its images via client-side JavaScript
    *after* the initial page load (a genuine single-page-app/AJAX
    carousel, as opposed to a lazy-load library that still ships real
    <img>/data-* attributes in the server-rendered HTML), those images
    are not present in `html` at all and cannot be found by any static
    parser, including this one -- see known_limitations.md.
    """
    soup = BeautifulSoup(html, "html.parser")
    found = []

    def add(raw_url):
        if not raw_url:
            return
        raw_url = raw_url.strip()
        if not raw_url:
            return
        if raw_url.startswith(("data:", "javascript:", "file:")):
            return
        absolute = urljoin(base_url, raw_url)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            return
        if parsed.path.lower().endswith(".svg"):
            return
        found.append(absolute)

    # 1. Open Graph / Twitter card meta images.
    og_image = soup.find("meta", property="og:image")
    if og_image:
        add(og_image.get("content"))

    twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
    if twitter_image:
        add(twitter_image.get("content"))

    # 2. <img> tags: normal src, common lazy-load attributes, srcset.
    for img in soup.find_all("img"):
        add(img.get("src"))
        add(img.get("data-src"))
        add(img.get("data-lazy-src"))
        add(img.get("data-original"))
        for u in _parse_srcset(img.get("srcset")):
            add(u)
        for u in _parse_srcset(img.get("data-srcset")):
            add(u)

    # 3. <picture><source srcset=...></picture> (and standalone <source>).
    for source in soup.find_all("source"):
        for u in _parse_srcset(source.get("srcset")):
            add(u)
        for u in _parse_srcset(source.get("data-srcset")):
            add(u)

    # 4. CSS background-image: inline style="" attributes anywhere...
    for tag in soup.find_all(style=True):
        for match in BACKGROUND_IMAGE_PATTERN.findall(tag["style"]):
            add(match)
    # ...and embedded <style> blocks.
    for style_tag in soup.find_all("style"):
        for match in BACKGROUND_IMAGE_PATTERN.findall(style_tag.get_text()):
            add(match)

    # 5. Common page-builder / lazy-load background-image data attributes.
    for attr in ("data-bg", "data-background", "data-background-image"):
        for tag in soup.find_all(attrs={attr: True}):
            add(tag.get(attr))

    # 6. Fallback sweep: image-looking URLs inside <script> blocks.
    for script_tag in soup.find_all("script"):
        script_text = script_tag.get_text() or ""
        for match in SCRIPT_IMAGE_URL_PATTERN.findall(script_text):
            add(match)

    deduped = []
    seen = set()
    for url in found:
        if url not in seen:
            seen.add(url)
            deduped.append(url)

    return deduped[:max_images]


# --------------------------------------------------------------------------
# Image download
# --------------------------------------------------------------------------

def download_image_bytes(
    url, timeout=REQUEST_TIMEOUT_SECONDS, max_bytes=MAX_IMAGE_DOWNLOAD_BYTES
):
    """Download image bytes, or return None on any failure (network
    error, non-200 status, or oversized response). Never raises --
    callers treat None as 'skip this candidate'."""
    try:
        response = requests.get(
            url, timeout=timeout, headers=REQUEST_HEADERS, stream=True
        )
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    content = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > max_bytes:
                return None
    except requests.exceptions.RequestException:
        return None

    return bytes(content)


def save_candidate_image(image_bytes, index, output_dir=DEFAULT_VERIFICATION_IMAGES_DIR):
    """Save a downloaded candidate image under a safe, generated
    filename (never a filename taken directly from the URL)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"candidate_{index:03d}.jpg")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


# --------------------------------------------------------------------------
# Face detection/comparison on candidate images
# (uses the same face_recognition/dlib approach as face_stage.py, but
#  without Step 1's "exactly one face" constraint, since a webpage
#  photo may legitimately contain zero, one, or many people)
# --------------------------------------------------------------------------

def encode_all_faces(image_array):
    """Detect every face in an already-decoded image array and return
    a list of (location, encoding) pairs. Returns [] if no faces."""
    locations = face_recognition.face_locations(image_array)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image_array, known_face_locations=locations)
    return list(zip(locations, encodings))


def best_face_distance(query_encoding, candidate_encodings):
    """Given the query encoding and a list of encodings found in one
    candidate image, return the smallest distance (a group photo's
    closest face), or None if candidate_encodings is empty."""
    if not candidate_encodings:
        return None
    distances = face_recognition.face_distance(candidate_encodings, query_encoding)
    return float(min(distances))


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def _print_banner(query_image_path, candidate_url):
    print("=" * 40)
    print("STEP 3: FACE MATCH VERIFICATION")
    print("=" * 40)
    print()
    print("Query image:")
    print(query_image_path)
    print()
    print("Candidate source:")
    print(candidate_url)
    print()
    print("Fetching candidate webpage...")
    print()


def _print_debug_urls(image_urls):
    print("Extracted image URLs:")
    for i, url in enumerate(image_urls, start=1):
        print(f"{i}. {url}")
    print()


def _print_image_result(index, total, url, face_count, distance, threshold, outcome):
    print(f"[{index}/{total}] {url}")
    print(f"Faces detected: {face_count}")
    if distance is not None:
        print(f"Best face distance: {distance:.2f}")
    print(f"Result: {outcome}")
    print()


def _print_final_result(result, threshold):
    print("=" * 40)
    print("VERIFICATION RESULT")
    print("=" * 40)
    print()
    print("Candidate source:")
    print(result["candidate_url"])
    print()

    if result["best_match"] is not None:
        print("Matching image:")
        print(result["best_match"]["image_url"])
        print()
        print("Face distance:")
        print(f"{result['best_match']['face_distance']:.2f}")
        print()
        print("Threshold:")
        print(f"{threshold:.2f}")
        print()
    else:
        print("No face was detected in any candidate image.")
        print()

    print(f"FACE MATCH: {'YES' if result['match_found'] else 'NO'}")
    print()
    print(f"Candidate discovered: {'VERIFIED' if result['match_found'] else 'NOT VERIFIED'}")
    print("=" * 40)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_verification_result(result, output_path=DEFAULT_VERIFICATION_OUTPUT_PATH):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return output_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def verify_match(
    query_image_path,
    search_result_path=DEFAULT_SEARCH_RESULT_PATH,
    output_path=DEFAULT_VERIFICATION_OUTPUT_PATH,
    images_dir=DEFAULT_VERIFICATION_IMAGES_DIR,
    threshold=FACE_DISTANCE_THRESHOLD,
    process_face_fn=process_face,
    fetch_webpage_fn=fetch_webpage,
    download_image_fn=download_image_bytes,
    debug=False,
):
    """
    Runs Step 3 end to end. process_face_fn / fetch_webpage_fn /
    download_image_fn are injectable purely so tests can run fully
    offline with mocked network and face-recognition calls.

    debug=True prints the full extracted image-URL list before
    downloading anything, to help diagnose extraction on a new page.

    Returns the saved verification result dict.
    """
    # 1. Query face (reuses Step 1's exactly-one-face logic as-is).
    query_result = process_face_fn(query_image_path)
    query_encoding = query_result["face_encoding"]

    # 2. Candidate URL (must come from Step 2's saved output).
    candidate_url = load_search_result(search_result_path)

    _print_banner(query_image_path, candidate_url)

    # 3-4. Fetch page, extract image URLs.
    html = fetch_webpage_fn(candidate_url)
    image_urls = extract_image_urls(html, candidate_url)

    print(f"Images discovered: {len(image_urls)}")
    print(f"Images selected for verification: {len(image_urls)}")
    print()

    if debug:
        _print_debug_urls(image_urls)

    print("Analyzing candidate images...")
    print()

    best = None  # {"image_url":..., "local_path":..., "face_distance":...}
    analyzed_count = 0
    total = len(image_urls)

    # 5-8. Download, detect, compare -- one image at a time, tolerating
    # any single image failing without aborting the whole verification.
    for i, image_url in enumerate(image_urls, start=1):
        image_bytes = download_image_fn(image_url)
        if image_bytes is None:
            _print_image_result(i, total, image_url, 0, None, threshold, "SKIPPED (download failed)")
            continue

        try:
            image_array = face_recognition.load_image_file(io.BytesIO(image_bytes))
        except Exception:
            _print_image_result(i, total, image_url, 0, None, threshold, "SKIPPED (invalid image)")
            continue

        analyzed_count += 1
        faces = encode_all_faces(image_array)
        face_count = len(faces)

        if face_count == 0:
            _print_image_result(i, total, image_url, 0, None, threshold, "NO FACE")
            continue

        candidate_encodings = [encoding for _, encoding in faces]
        distance = best_face_distance(query_encoding, candidate_encodings)
        is_match = distance <= threshold
        _print_image_result(
            i, total, image_url, face_count, distance, threshold,
            "MATCH" if is_match else "NO MATCH",
        )

        if best is None or distance < best["face_distance"]:
            saved_path = save_candidate_image(image_bytes, i, images_dir)
            best = {"image_url": image_url, "local_path": saved_path, "face_distance": distance}

    # 9. Best match + overall decision.
    match_found = best is not None and best["face_distance"] <= threshold

    result = {
        "query_image": query_image_path,
        "candidate_url": candidate_url,
        "verification_completed": True,
        "match_found": match_found,
        "best_match": (
            {
                "image_url": best["image_url"],
                "local_image_path": best["local_path"],
                "face_distance": round(best["face_distance"], 4),
                "threshold": threshold,
            }
            if best is not None
            else None
        ),
        "images_discovered": len(image_urls),
        "images_analyzed": analyzed_count,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # 10. Save.
    save_verification_result(result, output_path)
    _print_final_result(result, threshold)

    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    debug = "--debug" in argv
    positional = [a for a in argv if a != "--debug"]

    if len(positional) != 1:
        print("Usage: python -m src.verify_match <query_image_path> [--debug]")
        return 2

    query_image_path = positional[0]

    try:
        result = verify_match(query_image_path, debug=debug)
    except (FaceStageError, VerifyMatchError) as exc:
        print("=" * 40)
        print(f"ERROR: {exc}")
        print("Status: FAILED")
        print("=" * 40)
        return 1

    return 0 if result["match_found"] else 1


if __name__ == "__main__":
    sys.exit(main())
