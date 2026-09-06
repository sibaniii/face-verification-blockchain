"""
Step 1 of the pipeline: face detection and encoding.

Exposes process_face(image_path), which:
    - validates the image exists and can be loaded
    - detects faces in it
    - requires exactly ONE face (MVP constraint)
    - generates a 128-d face encoding for that face
    - returns a structured dict for later pipeline stages

This module does not perform any web search, hashing, or blockchain
work -- those are separate stages built later.
"""

import os
import sys

import face_recognition


class FaceStageError(Exception):
    """Base exception for all face_stage errors."""


class ImageNotFoundError(FaceStageError):
    """Raised when the given image path does not exist."""


class ImageLoadError(FaceStageError):
    """Raised when the image exists but cannot be read/decoded."""


class NoFaceDetectedError(FaceStageError):
    """Raised when zero faces are found in the image."""


class MultipleFacesDetectedError(FaceStageError):
    """Raised when more than one face is found in the image."""

    def __init__(self, count):
        self.count = count
        super().__init__(
            f"Multiple faces detected ({count}). "
            "Please provide an image containing exactly one face."
        )


def process_face(image_path):
    """
    Detect and encode the single face present in image_path.

    Returns:
        dict with keys:
            image_path:     str, the path passed in
            face_location:  tuple (top, right, bottom, left)
            face_encoding:  numpy.ndarray, shape (128,)
            face_count:     int, always 1 on success

    Raises:
        ImageNotFoundError, ImageLoadError,
        NoFaceDetectedError, MultipleFacesDetectedError
    """
    if not os.path.isfile(image_path):
        raise ImageNotFoundError(f"Image not found: {image_path}")

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as exc:
        raise ImageLoadError(
            f"Could not load image '{image_path}': {exc}"
        ) from exc

    face_locations = face_recognition.face_locations(image)
    face_count = len(face_locations)

    if face_count == 0:
        raise NoFaceDetectedError("No face detected.")
    if face_count > 1:
        raise MultipleFacesDetectedError(face_count)

    face_encodings = face_recognition.face_encodings(
        image, known_face_locations=face_locations
    )
    face_encoding = face_encodings[0]
    face_location = face_locations[0]

    return {
        "image_path": image_path,
        "face_location": face_location,
        "face_encoding": face_encoding,
        "face_count": face_count,
    }


def _print_success(result):
    print("=" * 40)
    print("STEP 1: FACE DETECTION & ENCODING")
    print(f"Image: {result['image_path']}")
    print(f"Faces detected: {result['face_count']}")
    print(f"Face location: {result['face_location']}")
    print("Encoding generated: YES")
    print(f"Encoding dimensions: {len(result['face_encoding'])}")
    print("Status: SUCCESS")
    print("=" * 40)


def _print_failure(message):
    print("=" * 40)
    print("STEP 1: FACE DETECTION & ENCODING")
    print(f"ERROR: {message}")
    print("Status: FAILED")
    print("=" * 40)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) != 1:
        print("Usage: python -m src.face_stage <image_path>")
        return 2

    image_path = argv[0]

    try:
        result = process_face(image_path)
    except (
        ImageNotFoundError,
        ImageLoadError,
        NoFaceDetectedError,
        MultipleFacesDetectedError,
    ) as exc:
        _print_failure(str(exc))
        return 1

    _print_success(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
