"""
Step 3 of the pipeline: independent face-match verification.

ARCHITECTURE NOTE:
    Step 2 performs DISCOVERY using a genuine visual web search.
    Step 3 independently verifies the discovered candidate.

Pipeline:
    query image
        -> process_face() from Step 1
        -> query face encoding
    candidate_url from Step 2
        -> fetch webpage
        -> extract images from normal HTML
        -> render webpage with Playwright for JavaScript/lazy-loaded images
        -> download candidate images
        -> detect ALL faces
        -> compare faces against query encoding
        -> keep smallest distance
        -> determine match

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
from playwright.sync_api import sync_playwright
import face_recognition

from src.face_stage import process_face, FaceStageError


# A face_recognition euclidean distance below this is considered a match.
# 0.6 is the standard threshold used by this project.
FACE_DISTANCE_THRESHOLD = 0.45


# Don't verify against an unbounded number of images on one page.
MAX_CANDIDATE_IMAGES = 30


# Matches url(...) inside CSS background-image declarations.
BACKGROUND_IMAGE_PATTERN = re.compile(
    r'url\(\s*[\'"]?([^\'")]+)[\'"]?\s*\)',
    re.IGNORECASE,
)


# Image-looking URLs embedded inside script blocks.
SCRIPT_IMAGE_URL_PATTERN = re.compile(
    r'https?://[^\s\'"<>]+?\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s\'"<>]*)?',
    re.IGNORECASE,
)


# Don't download unbounded content for one image.
MAX_IMAGE_DOWNLOAD_BYTES = 5 * 1024 * 1024

REQUEST_TIMEOUT_SECONDS = 15

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (hackathon-face-verify-bot)"
}


DEFAULT_SEARCH_RESULT_PATH = os.path.join(
    "data",
    "output",
    "search_result.json",
)

DEFAULT_VERIFICATION_OUTPUT_PATH = os.path.join(
    "data",
    "output",
    "verification_result.json",
)

DEFAULT_VERIFICATION_IMAGES_DIR = os.path.join(
    "data",
    "output",
    "verification_images",
)


class VerifyMatchError(Exception):
    """Base exception for all verify_match errors."""


class SearchResultMissingError(VerifyMatchError):
    """Raised when search_result.json is missing or unreadable."""


class CandidateMissingError(VerifyMatchError):
    """Raised when no usable candidate URL exists."""


class WebpageFetchError(VerifyMatchError):
    """Raised when the candidate webpage cannot be fetched."""


# --------------------------------------------------------------------------
# Step 2 output loading
# --------------------------------------------------------------------------

def load_search_result(path=DEFAULT_SEARCH_RESULT_PATH):
    """
    Read the candidate URL discovered by Step 2.

    The URL must come from search_stage's saved output.
    It is never invented or hardcoded.
    """

    if not os.path.isfile(path):
        raise SearchResultMissingError(
            f"{path} not found. "
            "Run Step 2 first."
        )

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    except (OSError, json.JSONDecodeError) as exc:
        raise SearchResultMissingError(
            f"Could not read {path}: {exc}"
        ) from exc

    candidate = (
        data.get("selected_candidate")
        if isinstance(data, dict)
        else None
    )

    if not isinstance(candidate, dict) or not candidate.get("url"):
        raise CandidateMissingError(
            f"{path} does not contain a selected_candidate with a url. "
            "Re-run Step 2 and select a candidate."
        )

    return candidate["url"]


# --------------------------------------------------------------------------
# Webpage fetching
# --------------------------------------------------------------------------

def fetch_webpage(url, timeout=REQUEST_TIMEOUT_SECONDS):
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers=REQUEST_HEADERS,
        )

    except requests.exceptions.RequestException as exc:
        raise WebpageFetchError(
            f"Could not fetch candidate webpage "
            f"({type(exc).__name__})."
        ) from exc

    if response.status_code != 200:
        raise WebpageFetchError(
            f"Candidate webpage returned HTTP "
            f"{response.status_code}."
        )

    return response.text


# --------------------------------------------------------------------------
# Static image extraction
# --------------------------------------------------------------------------

def _parse_srcset(value):
    """
    Split a srcset/data-srcset attribute into individual URLs.
    """

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


def extract_image_urls(
    html,
    base_url,
    max_images=MAX_CANDIDATE_IMAGES,
):
    """
    Extract image URLs from the normal HTML response.

    Handles:
        - og:image
        - twitter:image
        - img src
        - lazy-loading attributes
        - srcset
        - picture/source
        - CSS background images
        - page-builder background attributes
        - image URLs inside script blocks
    """

    soup = BeautifulSoup(html, "html.parser")

    found = []

    def add(raw_url):
        if not raw_url:
            return

        raw_url = raw_url.strip()

        if not raw_url:
            return

        if raw_url.startswith(
            ("data:", "javascript:", "file:")
        ):
            return

        absolute = urljoin(base_url, raw_url)

        parsed = urlparse(absolute)

        if parsed.scheme not in ("http", "https"):
            return

        if parsed.path.lower().endswith(".svg"):
            return

        found.append(absolute)

    # 1. Open Graph image.
    og_image = soup.find(
        "meta",
        property="og:image",
    )

    if og_image:
        add(og_image.get("content"))

    # 2. Twitter card image.
    twitter_image = soup.find(
        "meta",
        attrs={"name": "twitter:image"},
    )

    if twitter_image:
        add(twitter_image.get("content"))

    # 3. <img> tags.
    for img in soup.find_all("img"):

        add(img.get("src"))
        add(img.get("data-src"))
        add(img.get("data-lazy-src"))
        add(img.get("data-original"))

        for url in _parse_srcset(
            img.get("srcset")
        ):
            add(url)

        for url in _parse_srcset(
            img.get("data-srcset")
        ):
            add(url)

    # 4. <source> tags.
    for source in soup.find_all("source"):

        for url in _parse_srcset(
            source.get("srcset")
        ):
            add(url)

        for url in _parse_srcset(
            source.get("data-srcset")
        ):
            add(url)

    # 5. Inline CSS background images.
    for tag in soup.find_all(style=True):

        for match in BACKGROUND_IMAGE_PATTERN.findall(
            tag["style"]
        ):
            add(match)

    # 6. <style> blocks.
    for style_tag in soup.find_all("style"):

        for match in BACKGROUND_IMAGE_PATTERN.findall(
            style_tag.get_text()
        ):
            add(match)

    # 7. Common background-image attributes.
    for attr in (
        "data-bg",
        "data-background",
        "data-background-image",
    ):

        for tag in soup.find_all(
            attrs={attr: True}
        ):
            add(tag.get(attr))

    # 8. Image URLs embedded inside scripts.
    for script_tag in soup.find_all("script"):

        script_text = (
            script_tag.get_text() or ""
        )

        for match in SCRIPT_IMAGE_URL_PATTERN.findall(
            script_text
        ):
            add(match)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()

    for url in found:

        if url not in seen:
            seen.add(url)
            deduped.append(url)

    return deduped[:max_images]


# --------------------------------------------------------------------------
# Browser-rendered image extraction
# --------------------------------------------------------------------------

def extract_rendered_image_urls(
    url,
    max_images=MAX_CANDIDATE_IMAGES,
):
    """
    Render the candidate webpage in Chromium and collect images that may
    only appear after JavaScript or lazy loading executes.

    IMPORTANT:
        The URL always comes from Step 2.
        No website, person, platform, or URL is hardcoded.
    """

    found = []

    try:

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            page = browser.new_page(
                user_agent=REQUEST_HEADERS["User-Agent"]
            )

            try:

                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

            except Exception:
                # Some websites continue loading indefinitely.
                # The already-rendered page can still contain useful images.
                pass

            # Allow JavaScript and lazy loading to execute.
            page.wait_for_timeout(3000)

            # Scroll through the page to trigger lazy-loaded images.
            for _ in range(5):

                page.mouse.wheel(
                    0,
                    1200,
                )

                page.wait_for_timeout(500)

            # ----------------------------------------------------------
            # Collect <img> URLs.
            # ----------------------------------------------------------

            image_urls = page.locator(
                "img"
            ).evaluate_all(
                """
                imgs => imgs.flatMap(img => {
                    const values = [];

                    if (img.currentSrc) {
                        values.push(img.currentSrc);
                    }

                    if (img.src) {
                        values.push(img.src);
                    }

                    for (const attr of [
                        "data-src",
                        "data-lazy-src",
                        "data-original",
                        "data-image",
                        "data-url"
                    ]) {
                        const value = img.getAttribute(attr);

                        if (value) {
                            values.push(value);
                        }
                    }

                    for (const attr of [
                        "srcset",
                        "data-srcset"
                    ]) {
                        const value = img.getAttribute(attr);

                        if (value) {

                            for (
                                const part of value.split(",")
                            ) {

                                const candidate =
                                    part.trim()
                                        .split(/\\s+/)[0];

                                if (candidate) {
                                    values.push(candidate);
                                }
                            }
                        }
                    }

                    return values;
                })
                """
            )

            found.extend(image_urls)

            # ----------------------------------------------------------
            # Collect rendered CSS background images.
            # ----------------------------------------------------------

            background_urls = page.locator(
                "*"
            ).evaluate_all(
                """
                elements => elements.flatMap(el => {

                    const value =
                        getComputedStyle(el)
                        .getPropertyValue(
                            "background-image"
                        );

                    if (
                        !value ||
                        value === "none"
                    ) {
                        return [];
                    }

                    const matches = [];

                    const regex =
                        /url\\(["']?([^"')]+)["']?\\)/g;

                    let match;

                    while (
                        (match = regex.exec(value)) !== null
                    ) {
                        matches.push(match[1]);
                    }

                    return matches;
                })
                """
            )

            found.extend(background_urls)

            # ----------------------------------------------------------
            # Collect image resources loaded by the browser.
            # ----------------------------------------------------------

            resource_urls = page.evaluate(
                """
                () => performance
                    .getEntriesByType("resource")
                    .map(entry => entry.name)
                    .filter(url =>
                        /\\.(jpg|jpeg|png|webp|gif)(\\?|$)/i
                            .test(url)
                    )
                """
            )

            found.extend(resource_urls)

            browser.close()

    except Exception as exc:

        print(
            "Browser image extraction skipped: "
            f"{type(exc).__name__}"
        )

    # Resolve relative URLs and deduplicate.
    deduped = []
    seen = set()

    for raw_url in found:

        if not raw_url:
            continue

        raw_url = str(raw_url).strip()

        if not raw_url:
            continue

        if raw_url.startswith(
            ("data:", "javascript:", "file:")
        ):
            continue

        absolute = urljoin(
            url,
            raw_url,
        )

        parsed = urlparse(absolute)

        if parsed.scheme not in (
            "http",
            "https",
        ):
            continue

        if parsed.path.lower().endswith(".svg"):
            continue

        if absolute not in seen:

            seen.add(absolute)
            deduped.append(absolute)

    return deduped[:max_images]


# --------------------------------------------------------------------------
# Image download
# --------------------------------------------------------------------------

def download_image_bytes(
    url,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_bytes=MAX_IMAGE_DOWNLOAD_BYTES,
):
    """
    Download image bytes.

    Returns None if the download fails.
    """

    try:

        response = requests.get(
            url,
            timeout=timeout,
            headers=REQUEST_HEADERS,
            stream=True,
        )

    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    content = bytearray()

    try:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            if not chunk:
                continue

            content.extend(chunk)

            if len(content) > max_bytes:
                return None

    except requests.exceptions.RequestException:
        return None

    return bytes(content)


def save_candidate_image(
    image_bytes,
    index,
    output_dir=DEFAULT_VERIFICATION_IMAGES_DIR,
):
    """
    Save a downloaded candidate image.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        f"candidate_{index:03d}.jpg",
    )

    with open(path, "wb") as f:
        f.write(image_bytes)

    return path


# --------------------------------------------------------------------------
# Face detection and comparison
# --------------------------------------------------------------------------

def encode_all_faces(image_array):
    """
    Detect every face in an image.

    Returns:
        [(location, encoding), ...]
    """

    locations = face_recognition.face_locations(
        image_array
    )

    if not locations:
        return []

    encodings = face_recognition.face_encodings(
        image_array,
        known_face_locations=locations,
    )

    return list(
        zip(
            locations,
            encodings,
        )
    )


def best_face_distance(
    query_encoding,
    candidate_encodings,
):
    """
    Return the smallest face distance in a candidate image.
    """

    if not candidate_encodings:
        return None

    distances = face_recognition.face_distance(
        candidate_encodings,
        query_encoding,
    )

    return float(
        min(distances)
    )


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def _print_banner(
    query_image_path,
    candidate_url,
):
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

    for i, url in enumerate(
        image_urls,
        start=1,
    ):
        print(
            f"{i}. {url}"
        )

    print()


def _print_image_result(
    index,
    total,
    url,
    face_count,
    distance,
    threshold,
    outcome,
):
    print(
        f"[{index}/{total}] {url}"
    )

    print(
        f"Faces detected: {face_count}"
    )

    if distance is not None:

        print(
            f"Best face distance: "
            f"{distance:.2f}"
        )

    print(
        f"Result: {outcome}"
    )

    print()


def _print_final_result(
    result,
    threshold,
):
    print("=" * 40)
    print("VERIFICATION RESULT")
    print("=" * 40)
    print()

    print("Candidate source:")
    print(result["candidate_url"])
    print()

    if result["best_match"] is not None:

        print("Matching image:")
        print(
            result["best_match"]["image_url"]
        )
        print()

        print("Face distance:")
        print(
            f"{result['best_match']['face_distance']:.2f}"
        )
        print()

        print("Threshold:")
        print(
            f"{threshold:.2f}"
        )
        print()

    else:

        print(
            "No face was detected in any "
            "candidate image."
        )
        print()

    print(
        f"FACE MATCH: "
        f"{'YES' if result['match_found'] else 'NO'}"
    )

    print()

    print(
        "Candidate discovered: "
        f"{'VERIFIED' if result['match_found'] else 'NOT VERIFIED'}"
    )

    print("=" * 40)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_verification_result(
    result,
    output_path=DEFAULT_VERIFICATION_OUTPUT_PATH,
):
    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    return output_path


# --------------------------------------------------------------------------
# Main verification pipeline
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
    Runs Step 3 end to end.

    The candidate URL always comes from Step 2.

    Static HTML extraction and browser-rendered extraction are combined
    before face verification.
    """

    # ----------------------------------------------------------
    # 1. Query face from Step 1.
    # ----------------------------------------------------------

    query_result = process_face_fn(
        query_image_path
    )

    query_encoding = query_result[
        "face_encoding"
    ]

    # ----------------------------------------------------------
    # 2. Candidate URL from Step 2.
    # ----------------------------------------------------------

    candidate_url = load_search_result(
        search_result_path
    )

    _print_banner(
        query_image_path,
        candidate_url,
    )

    # ----------------------------------------------------------
    # 3. Fetch normal HTML.
    # ----------------------------------------------------------

    html = fetch_webpage_fn(
        candidate_url
    )

    # ----------------------------------------------------------
    # 4. Extract images from normal HTML.
    # ----------------------------------------------------------

    static_image_urls = extract_image_urls(
        html,
        candidate_url,
    )

    # ----------------------------------------------------------
    # 5. Render the same discovered URL in Chromium.
    # ----------------------------------------------------------

    rendered_image_urls = (
        extract_rendered_image_urls(
            candidate_url
        )
    )

    # ----------------------------------------------------------
    # 6. Combine both image sources.
    # ----------------------------------------------------------

    image_urls = []

    seen_image_urls = set()

    for image_url in (
        static_image_urls +
        rendered_image_urls
    ):

        if image_url not in seen_image_urls:

            seen_image_urls.add(
                image_url
            )

            image_urls.append(
                image_url
            )

    image_urls = image_urls[
        :MAX_CANDIDATE_IMAGES
    ]

    print(
        f"Images discovered: "
        f"{len(image_urls)}"
    )

    print(
        "Images selected for verification: "
        f"{len(image_urls)}"
    )

    print()

    if debug:
        _print_debug_urls(
            image_urls
        )

    print(
        "Analyzing candidate images..."
    )

    print()

    # ----------------------------------------------------------
    # 7. Download, detect and compare.
    # ----------------------------------------------------------

    best = None

    analyzed_count = 0

    total = len(image_urls)

    for i, image_url in enumerate(
        image_urls,
        start=1,
    ):

        image_bytes = download_image_fn(
            image_url
        )

        if image_bytes is None:

            _print_image_result(
                i,
                total,
                image_url,
                0,
                None,
                threshold,
                "SKIPPED (download failed)",
            )

            continue

        try:

            image_array = (
                face_recognition.load_image_file(
                    io.BytesIO(image_bytes)
                )
            )

        except Exception:

            _print_image_result(
                i,
                total,
                image_url,
                0,
                None,
                threshold,
                "SKIPPED (invalid image)",
            )

            continue

        analyzed_count += 1

        faces = encode_all_faces(
            image_array
        )

        face_count = len(faces)

        if face_count == 0:

            _print_image_result(
                i,
                total,
                image_url,
                0,
                None,
                threshold,
                "NO FACE",
            )

            continue

        candidate_encodings = [
            encoding
            for _, encoding in faces
        ]

        distance = best_face_distance(
            query_encoding,
            candidate_encodings,
        )

        is_match = (
            distance <= threshold
        )

        _print_image_result(
            i,
            total,
            image_url,
            face_count,
            distance,
            threshold,
            "MATCH"
            if is_match
            else "NO MATCH",
        )

        if (
            best is None
            or distance <
            best["face_distance"]
        ):

            saved_path = (
                save_candidate_image(
                    image_bytes,
                    i,
                    images_dir,
                )
            )

            best = {
                "image_url": image_url,
                "local_path": saved_path,
                "face_distance": distance,
            }

    # ----------------------------------------------------------
    # 8. Overall decision.
    # ----------------------------------------------------------

    match_found = (
        best is not None
        and best["face_distance"] <= threshold
    )

    result = {
        "query_image": query_image_path,

        "candidate_url": candidate_url,

        "verification_completed": True,

        "match_found": match_found,

        "best_match": (
            {
                "image_url": best["image_url"],
                "local_image_path": best["local_path"],
                "face_distance": round(
                    best["face_distance"],
                    4,
                ),
                "threshold": threshold,
            }
            if best is not None
            else None
        ),

        "images_discovered": len(
            image_urls
        ),

        "images_analyzed": analyzed_count,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # ----------------------------------------------------------
    # 9. Save result.
    # ----------------------------------------------------------

    save_verification_result(
        result,
        output_path,
    )

    _print_final_result(
        result,
        threshold,
    )

    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):

    argv = (
        sys.argv[1:]
        if argv is None
        else argv
    )

    debug = "--debug" in argv

    positional = [
        a
        for a in argv
        if a != "--debug"
    ]

    if len(positional) != 1:

        print(
            "Usage: "
            "python -m src.verify_match "
            "<query_image_path> [--debug]"
        )

        return 2

    query_image_path = positional[0]

    try:

        result = verify_match(
            query_image_path,
            debug=debug,
        )

    except (
        FaceStageError,
        VerifyMatchError,
    ) as exc:

        print("=" * 40)
        print(
            f"ERROR: {exc}"
        )
        print(
            "Status: FAILED"
        )
        print("=" * 40)

        return 1

    return (
        0
        if result["match_found"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())