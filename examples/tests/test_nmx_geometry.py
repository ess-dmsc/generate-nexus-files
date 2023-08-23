import json
from pathlib import PurePosixPath
from itertools import zip_longest
from math import isclose
import os

import pytest

from examples.nmx.nx_detector import BoxNXDetector
from examples.nmx.main import render


TOLERANCE = 0.00001  # relative tolerance (e.g. 0.00001 = 0.001%)


def assert_all_are_close(iterable_a, iterable_b):
    """Using an assert loop instead of all() just so test errors show the values that differ"""
    iterable_a, iterable_b = list(iterable_a), list(iterable_b)
    for a, b in zip_longest(iterable_a, iterable_b):
        if a is None or b is None:
            assert (
                a == b
            ), f"Iterables have different lengths: {list(iterable_a)} vs. {list(iterable_b)}"
        try:
            iter(a)
            iter(b)
            # items are iterables, call recursively
            assert_all_are_close(a, b)
        except TypeError:
            # items are not iterables, compare directly
            assert isclose(a, b, rel_tol=TOLERANCE)


@pytest.mark.parametrize("number_of_pixels_x", [4, 1280])
@pytest.mark.parametrize("number_of_pixels_y", [5, 1280])
@pytest.mark.parametrize("first_pixel_id", [0, 1, 1024])
def test_detector_boundaries(number_of_pixels_x, number_of_pixels_y, first_pixel_id):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_y=number_of_pixels_y,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=1.0,
        first_pixel_id=first_pixel_id,
    )

    # Test first pixel ID
    assert detector.is_valid_pixel_id(
        first_pixel_id
    ), f"First pixel ID {first_pixel_id} is not valid."
    # Test previous pixel ID
    with pytest.raises(AssertionError):
        assert detector.is_valid_pixel_id(
            first_pixel_id - 1
        ), f"Pixel ID {first_pixel_id - 1} should not be valid."

    last_pixel_id = first_pixel_id + detector.number_of_pixels - 1
    # Test last pixel ID
    assert detector.is_valid_pixel_id(
        last_pixel_id
    ), f"Last pixel ID {last_pixel_id} is not valid."
    # Test next pixel ID
    with pytest.raises(AssertionError):
        assert detector.is_valid_pixel_id(
            last_pixel_id + 1
        ), f"Pixel ID {last_pixel_id + 1} should not be valid."


@pytest.mark.parametrize("number_of_pixels_x", [4, 180])
@pytest.mark.parametrize("number_of_pixels_y", [5, 180])
@pytest.mark.parametrize("first_pixel_id", [0, 1, 1024])
def test_pixel_ids(number_of_pixels_x, number_of_pixels_y, first_pixel_id):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_y=number_of_pixels_y,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=1.0,
        first_pixel_id=first_pixel_id,
    )
    pixel_ids = list(detector.pixel_ids)
    assert len(pixel_ids) == number_of_pixels_x * number_of_pixels_y
    assert pixel_ids[0] == detector.first_pixel_id
    assert pixel_ids[-1] == detector.last_pixel_id
    for idx, pixel_id in enumerate(pixel_ids[1:]):
        assert pixel_id - 1 == pixel_ids[idx]


@pytest.mark.parametrize("first_pixel_id", [0, 1, 1024])
@pytest.mark.parametrize("channel_pitch_x", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("channel_pitch_y", [1.0, 2.0, 3.0])
@pytest.mark.parametrize(
    "x_length, y_length, gap_every_x_pixels, gap_every_y_pixels, gap_width_x, gap_width_y",
    [(4, 4, 0, 0, 0, 0), (4, 8, 1, 1, 0.5, 0.5), (20, 5, 4, 8, 0.2, 0.8)],
)
def test_detector_size(
    first_pixel_id,
    channel_pitch_x,
    channel_pitch_y,
    x_length,
    y_length,
    gap_every_x_pixels,
    gap_every_y_pixels,
    gap_width_x,
    gap_width_y,
):
    detector = BoxNXDetector(
        PurePosixPath("/entry/instrument"),
        "test",
        "nmx",
        x_length,
        y_length,
        10.0,
        channel_pitch_x,
        channel_pitch_y,
        first_pixel_id=first_pixel_id,
        gap_every_x_pixels=gap_every_x_pixels,
        gap_every_y_pixels=gap_every_y_pixels,
        gap_width_x=gap_width_x,
        gap_width_y=gap_width_y,
    )

    expected_size_x = (
        x_length * channel_pitch_x
        + ((x_length - 1) // gap_every_x_pixels) * gap_width_x
        if gap_every_x_pixels > 0
        else x_length * channel_pitch_x
    )
    assert detector.size_x == expected_size_x
    expected_size_y = (
        y_length * channel_pitch_y
        + ((y_length - 1) // gap_every_y_pixels) * gap_width_y
        if gap_every_y_pixels > 0
        else y_length * channel_pitch_y
    )
    assert detector.size_y == expected_size_y


def test_detector_size_simple_with_no_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector1",
        instrument_name="nmx",
        number_of_pixels_x=4,
        number_of_pixels_y=4,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        gap_every_x_pixels=0,
        gap_every_y_pixels=0,
        gap_width_x=0.0,
        gap_width_y=0.0,
        first_pixel_id=1,
    )
    assert isclose(detector.size_x, 4.0, rel_tol=TOLERANCE)
    assert isclose(detector.size_y, 8.4, rel_tol=TOLERANCE)


def test_detector_size_simple_with_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector2",
        instrument_name="nmx",
        number_of_pixels_x=4,
        number_of_pixels_y=4,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=1.0,
        gap_every_x_pixels=1,
        gap_every_y_pixels=2,
        gap_width_x=0.7,
        gap_width_y=0.5,
        first_pixel_id=1,
    )
    assert isclose(detector.size_x, 6.1, rel_tol=TOLERANCE)
    assert isclose(detector.size_y, 4.5, rel_tol=TOLERANCE)


def test_get_pixel_coordinates_simple_2x2_example_no_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector1",
        instrument_name="nmx",
        number_of_pixels_x=2,
        number_of_pixels_y=2,
        size_z=10.0,
        channel_pitch_x=128,
        channel_pitch_y=128,
        gap_every_x_pixels=0,
        gap_every_y_pixels=0,
        gap_width_x=0.0,
        gap_width_y=0.0,
        first_pixel_id=1,
    )
    pixel = 1
    expected_coords = [-64, 64, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 2
    expected_coords = [64, 64, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 3
    expected_coords = [-64, -64, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 4
    expected_coords = [64, -64, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)


def test_get_pixel_coordinates_simple_2x2_example_with_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector1",
        instrument_name="nmx",
        number_of_pixels_x=2,
        number_of_pixels_y=2,
        size_z=10.0,
        channel_pitch_x=128,
        channel_pitch_y=128,
        gap_every_x_pixels=1,
        gap_every_y_pixels=1,
        gap_width_x=0.7,
        gap_width_y=1.1,
        first_pixel_id=1,
    )
    pixel = 1
    expected_coords = [-64.35, 64.55, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 2
    expected_coords = [64.35, 64.55, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 3
    expected_coords = [-64.35, -64.55, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)
    pixel = 4
    expected_coords = [64.35, -64.55, 0]
    assert_all_are_close(detector.get_pixel_coordinates(pixel), expected_coords)


def test_get_pixel_coordinates_simple_4x4_no_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=4,
        number_of_pixels_y=4,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        gap_every_x_pixels=0,
        gap_every_y_pixels=0,
        gap_width_x=0.7,
        gap_width_y=0.5,
        first_pixel_id=1,
    )
    expected_coords_first_pixel = [-1.5, 3.15, 0]
    expected_coords_last_pixel = [1.5, -3.15, 0]
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.first_pixel_id),
        expected_coords_first_pixel,
    )
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.last_pixel_id),
        expected_coords_last_pixel,
    )


def test_get_pixel_coordinates_simple_4x4_with_gaps():
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=4,
        number_of_pixels_y=4,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        gap_every_x_pixels=1,
        gap_every_y_pixels=2,
        gap_width_x=0.7,
        gap_width_y=0.5,
        first_pixel_id=1,
    )
    expected_coords_first_pixel = [-1.5 - 1.5 * 0.7, 3.15 + 0.5 * 0.5, 0]
    expected_coords_last_pixel = [1.5 + 1.5 * 0.7, -3.15 - 0.5 * 0.5, 0]
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.first_pixel_id),
        expected_coords_first_pixel,
    )
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.last_pixel_id),
        expected_coords_last_pixel,
    )


@pytest.mark.parametrize("first_pixel_id", [0, 1, 1024])
def test_first_pixel_id_does_not_affect_pixel_coordinates(first_pixel_id):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=4,
        number_of_pixels_y=4,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        gap_every_x_pixels=1,
        gap_every_y_pixels=2,
        gap_width_x=0.7,
        gap_width_y=0.5,
        first_pixel_id=first_pixel_id,
    )
    expected_coords_first_pixel = [-1.5 - 1.5 * 0.7, 3.15 + 0.5 * 0.5, 0]
    expected_coords_last_pixel = [1.5 + 1.5 * 0.7, -3.15 - 0.5 * 0.5, 0]
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.first_pixel_id),
        expected_coords_first_pixel,
    )
    assert_all_are_close(
        detector.get_pixel_coordinates(detector.last_pixel_id),
        expected_coords_last_pixel,
    )


@pytest.mark.parametrize("first_pixel_id", [3])
@pytest.mark.parametrize("number_of_pixels_x", [2])
@pytest.mark.parametrize("number_of_pixels_y", [2])
def test_get_detector_numbers_simple_2x2_example(
    first_pixel_id, number_of_pixels_x, number_of_pixels_y
):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_y=number_of_pixels_y,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        first_pixel_id=first_pixel_id,
    )

    pixel_grid = [[3, 4], [5, 6]]
    assert_all_are_close(pixel_grid, detector.get_detector_numbers())


@pytest.mark.parametrize("first_pixel_id", [0, 1, 1024])
@pytest.mark.parametrize("number_of_pixels_x", [4, 180])
@pytest.mark.parametrize("number_of_pixels_y", [5, 180])
def test_get_detector_numbers(first_pixel_id, number_of_pixels_x, number_of_pixels_y):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_y=number_of_pixels_y,
        size_z=1.0,
        channel_pitch_x=1.0,
        channel_pitch_y=2.1,
        first_pixel_id=first_pixel_id,
    )

    pixels = [
        p
        for p in range(
            first_pixel_id, first_pixel_id + (number_of_pixels_x * number_of_pixels_y)
        )
    ]
    pixel_grid = [
        pixels[i: i + number_of_pixels_x]
        for i in range(0, len(pixels), number_of_pixels_x)
    ]
    assert_all_are_close(pixel_grid, detector.get_detector_numbers())


@pytest.mark.parametrize("first_pixel_id", [3])
@pytest.mark.parametrize("number_of_pixels_x", [2])
@pytest.mark.parametrize("number_of_pixels_y", [4])
def test_pixel_offsets_simple_4x2_example_with_gaps(
    first_pixel_id, number_of_pixels_x, number_of_pixels_y
):
    detector = BoxNXDetector(
        parent=PurePosixPath("/entry/instrument"),
        name="Detector",
        instrument_name="nmx",
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_y=number_of_pixels_y,
        size_z=1.0,
        channel_pitch_x=128,
        channel_pitch_y=128,
        first_pixel_id=first_pixel_id,
        gap_every_x_pixels=1,
        gap_every_y_pixels=2,
        gap_width_x=0.7,
        gap_width_y=0.5,
    )
    expected_x_offsets = [-64.35, 64.35]
    expected_y_offsets = [192.25, 64.25, -64.25, -192.25]
    expected_z_offsets = [0]

    assert_all_are_close(expected_x_offsets, detector.get_x_pixel_offsets())
    assert_all_are_close(expected_y_offsets, detector.get_y_pixel_offsets())
    assert_all_are_close(expected_z_offsets, detector.get_z_pixel_offsets())


# Test uses a file generated with FACTOR=80, so it is skipped by default
# Change the factor in main.py if you want to test this.
@pytest.mark.skip
def test_render_3_detectors():
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    json1 = load_json("examples/tests/json/nmx_render_3_detectors_FACTOR_80.json")
    template_dir, template_file_name = os.path.split(
        "examples/nmx/template_nmx_v3.0_baseline.json.j2"
    )
    json2 = json.loads(render(template_dir, template_file_name))
    assert json1 == json2
