"""
Tests for src.face_stage.

Some tests need real sample images placed in data/input/ (see README
instructions). Those tests are automatically skipped if the relevant
image file is not present, so `pytest` still runs cleanly on a fresh
checkout before you've added sample images -- only test_nonexistent_
file_raises will actually execute at that point.
"""

import os

import pytest

from src.face_stage import (
    process_face,
    ImageNotFoundError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input")

SINGLE_FACE_IMAGE = os.path.join(DATA_DIR, "single_face.jpg")
NO_FACE_IMAGE = os.path.join(DATA_DIR, "no_face.jpg")
MULTI_FACE_IMAGE = os.path.join(DATA_DIR, "multi_face.jpg")


def test_nonexistent_file_raises():
    with pytest.raises(ImageNotFoundError):
        process_face(os.path.join(DATA_DIR, "does_not_exist.jpg"))


@pytest.mark.skipif(
    not os.path.isfile(SINGLE_FACE_IMAGE),
    reason="place a single-face sample image at data/input/single_face.jpg",
)
def test_single_face_success():
    result = process_face(SINGLE_FACE_IMAGE)

    assert result["face_count"] == 1
    assert result["image_path"] == SINGLE_FACE_IMAGE
    assert len(result["face_location"]) == 4
    assert len(result["face_encoding"]) == 128


@pytest.mark.skipif(
    not os.path.isfile(NO_FACE_IMAGE),
    reason="place a no-face sample image at data/input/no_face.jpg",
)
def test_zero_faces_raises():
    with pytest.raises(NoFaceDetectedError):
        process_face(NO_FACE_IMAGE)


@pytest.mark.skipif(
    not os.path.isfile(MULTI_FACE_IMAGE),
    reason="place a multi-face sample image at data/input/multi_face.jpg",
)
def test_multiple_faces_raises():
    with pytest.raises(MultipleFacesDetectedError):
        process_face(MULTI_FACE_IMAGE)
