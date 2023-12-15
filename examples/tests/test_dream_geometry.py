import json
from os import path
from typing import Tuple

import numpy as np
import pytest
from examples.utils.detector_geometry_from_json import (
    DETECTOR_NUMBER, X_PIXEL_OFFSET, Y_PIXEL_OFFSET, Z_PIXEL_OFFSET,
    BaseDetectorGeometry, retrieve_data_from_json)

GLOBAL_FORWARD_ENDCAP_ID_OFFSET = 0
GLOBAL_BACKWARD_ENDCAP_ID_OFFSET = 71680
GLOBAL_MANTLE_ID_OFFSET = 229376
GLOBAL_HR_ID_OFFSET = 1122304

ENDCAP_X_OFFSET_PER_SECTOR = 56
ENDCAP_Y_OFFSET_PER_STRIP = 16

MANTLE_Y_OFFSET_PER_MODULE = 12
MANTLE_NUM_STRIPS = 256

CUBOID_Y_OFFSET_PER_STRIP = 112

# The geometries position is in [mm], define lengths for clarity
m = 1e3
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m

udeg = 1e-6

precision = 20 * nm
cuboid_precision = 1 * um
endcap_rotation_precision = 200 * udeg


"""
Functions for calculating the mantle pixel ID. Expects inputs in ICD form, NOT GEANT4 form.
"""


def calc_mantle_x(strip: np.ndarray) -> np.ndarray:
    """
    Calculate the mantle's x-coordinate.

    Args:
    strip (np.ndarray): NumPy array of strip numbers.

    Returns:
    np.ndarray: NumPy array of calculated x-coordinates.
    """
    if not isinstance(strip, np.ndarray) or strip.dtype.kind != "i":
        raise TypeError("strip must be a NumPy array of integers")

    return strip


def calc_mantle_y(
    mod: np.ndarray, cass: np.ndarray, ctr: np.ndarray, wire: np.ndarray, num_mods: int
) -> np.ndarray:
    """
    Calculate the mantle's y-coordinate.

    Args:
    mod (np.ndarray): NumPy array of module numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    ctr (np.ndarray): NumPy array of counter numbers.
    wire (np.ndarray): NumPy array of wire numbers.
    num_mods (int): The total number of modules.

    Returns:
    np.ndarray: NumPy array of calculated y-coordinates.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [mod, cass, ctr, wire]
    ) or not isinstance(num_mods, int):
        raise TypeError(
            "All arguments except num_mods must be NumPy arrays of integers, and num_mods must be an integer"
        )

    return (
        num_mods * MANTLE_Y_OFFSET_PER_MODULE * wire
        + MANTLE_Y_OFFSET_PER_MODULE * mod
        + 2 * cass
        + ctr
    )


def calc_mantle_pixel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the mantle pixel ID.

    Args:
    x (np.ndarray): NumPy array of x-coordinates.
    y (np.ndarray): NumPy array of y-coordinates.

    Returns:
    np.ndarray: NumPy array of calculated mantle pixel IDs.
    """
    if not all(isinstance(arg, np.ndarray) and arg.dtype.kind == "i" for arg in [x, y]):
        raise TypeError("x and y must be NumPy arrays of integers")

    return MANTLE_NUM_STRIPS * y + x + 1 + GLOBAL_MANTLE_ID_OFFSET


def get_mantle_pixels(
    strip: np.ndarray,
    module: np.ndarray,
    cass: np.ndarray,
    counter: np.ndarray,
    wire: np.ndarray,
    num_mods: int,
) -> np.ndarray:
    """
    Get the mantle pixels based on various parameters.

    Args:
    strip (np.ndarray): NumPy array of strip numbers.
    module (np.ndarray): NumPy array of module numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    counter (np.ndarray): NumPy array of counter numbers.
    wire (np.ndarray): NumPy array of wire numbers.
    num_mods (int): The number of modules.

    Returns:
    np.ndarray: NumPy array of calculated mantle pixels.
    """
    x = calc_mantle_x(strip)
    y = calc_mantle_y(module, cass, counter, wire, num_mods)
    pixel = calc_mantle_pixel(x, y)

    return pixel


"""
Functions for calculating the HR pixel ID. Expects inputs in ICD form, NOT GEANT4 form.
The OFFSETS_HR and ROTATE_HR lookup tables are taken from the ICD drawing but enumrated based on sector-segment and module.
"""


OFFSETS_HR = [
                        (32,  0), (48,  0), (64,  0),
              (16, 16), (32, 16), (48, 16), (64, 16), (80, 16),
     (0, 32), (16, 32), (32, 32), (48, 32), (64, 32), (80, 32), (96, 32),
     (0, 48), (16, 48), (32, 48),           (64, 48), (80, 48), (96, 48),
     (0, 64), (16, 64), (32, 64),           (64, 64), (80, 64), (96, 64),
              (16, 80), (32, 80),           (64, 80), (80, 80),
                        (32, 96),           (64, 96)
]

ROTATE_HR = [
                    (0, 5), (0, 2), (3, 8),
            (0, 7), (0, 4), (0, 1), (3, 6), (3, 7),
    (0, 8), (0, 6), (0, 3), (0, 0), (3, 3), (3, 4), (3, 5),
    (1, 2), (1, 1), (1, 0),         (3, 0), (3, 1), (3, 2),
    (1, 5), (1, 4), (1, 3),         (2, 3), (2, 6), (2, 8),
            (1, 7), (1, 6),         (2, 4), (2, 7),
                    (1, 8),         (2, 5)
]

ICD_TO_GEANT_MAP = {i: ROTATE_HR[i] for i in range(len(ROTATE_HR))}


def calc_cuboid_x_local(cass: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    """
    Calculate the local x-coordinate for a cuboid.

    Args:
    cass (np.ndarray): NumPy array of cassette numbers.
    ctr (np.ndarray): NumPy array of counter numbers.

    Returns:
    np.ndarray: NumPy array of calculated local x-coordinates.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i" for arg in [cass, ctr]
    ):
        raise TypeError("cass and ctr must be NumPy arrays of integers")

    return 2 * cass + ctr


def calc_cuboid_y_local(wire: np.ndarray) -> np.ndarray:
    """
    Calculate the local y-coordinate for a cuboid.

    Args:
    wire (np.ndarray): NumPy array of wire numbers.

    Returns:
    np.ndarray: NumPy array of calculated local y-coordinates.
    """
    if not isinstance(wire, np.ndarray) or wire.dtype.kind != "i":
        raise TypeError("wire must be a NumPy array of integers")

    return 15 - wire


def rotate_cuboid(
    x: np.ndarray, y: np.ndarray, sect_seg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate the cuboid coordinates based on the sector segment.

    This function applies a transformation to the coordinates based on the sector segment.
    There are four types of transformations depending on the sector segment (0 to 3):
    - Segment 1: Reflect both coordinates across the line x = 15 and y = 15.
    - Segment 2: Rotate 90 degrees clockwise.
    - Segment 3: No change.
    - Segment 0: Rotate 90 degrees counterclockwise.

    Args:
    x (np.ndarray): NumPy array of x-coordinates.
    y (np.ndarray): NumPy array of y-coordinates.
    sect_seg (np.ndarray): NumPy array of sector segment numbers.

    Returns:
    tuple[np.ndarray, np.ndarray]: Tuple of NumPy arrays representing the new x and y coordinates after rotation.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [x, y, sect_seg]
    ):
        raise TypeError("x, y, and sect_seg must be NumPy arrays of integers")

    x_new, y_new = np.empty_like(x), np.empty_like(y)
    transformations = {1: (15 - x, 15 - y), 2: (15 - y, x), 3: (x, y), 0: (y, 15 - x)}
    for seg, (trans_x, trans_y) in transformations.items():
        mask = sect_seg == seg
        x_new[mask], y_new[mask] = trans_x[mask], trans_y[mask]

    return x_new, y_new


def calc_cuboid_y_global(strip: np.ndarray) -> np.ndarray:
    """
    Calculate the global y-coordinate for a cuboid.

    Args:
    strip (np.ndarray): NumPy array of strip numbers.

    Returns:
    np.ndarray: NumPy array of calculated global y-coordinates.
    """
    if not isinstance(strip, np.ndarray) or strip.dtype.kind != "i":
        raise TypeError("strip must be a NumPy array of integers")

    return CUBOID_Y_OFFSET_PER_STRIP * strip


def calc_cuboid_offset(
    sect_seg: np.ndarray, module: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the offset for cuboid coordinates based on sector segment and module.

    Args:
    sect_seg (np.ndarray): NumPy array of sector segment numbers.
    module (np.ndarray): NumPy array of module numbers.

    Returns:
    tuple[np.ndarray, np.ndarray]: Tuple of NumPy arrays representing x and y offsets.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [sect_seg, module]
    ):
        raise TypeError("sect_seg and module must be NumPy arrays of integers")

    x = []
    y = []
    for ss, m in zip(sect_seg, module):
        if (ss, m) not in ROTATE_HR:
            print(f"sector-segment {ss} module {m} not found in lookup table")
            continue
        idx = ROTATE_HR.index((ss, m))
        x_o, y_o = OFFSETS_HR[idx]
        x.append(x_o)
        y.append(y_o)

    return np.array(x), np.array(y)


def calc_cuboid_x(x_local: np.ndarray, x_offset: np.ndarray) -> np.ndarray:
    """
    Calculate the global x-coordinate for a cuboid.

    Args:
    x_local (np.ndarray): NumPy array of local x-coordinates.
    x_offset (np.ndarray): NumPy array of x offsets.

    Returns:
    np.ndarray: NumPy array of calculated global x-coordinates.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [x_local, x_offset]
    ):
        raise TypeError("x_local and x_offset must be NumPy arrays of integers")

    return x_local + x_offset


def calc_cuboid_y(
    y_global: np.ndarray, y_local: np.ndarray, y_offset: np.ndarray
) -> np.ndarray:
    """
    Calculate the final y-coordinate for a cuboid.

    Args:
    y_global (np.ndarray): NumPy array of global y-coordinates.
    y_local (np.ndarray): NumPy array of local y-coordinates.
    y_offset (np.ndarray): NumPy array of y offsets.

    Returns:
    np.ndarray: NumPy array of calculated final y-coordinates.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [y_global, y_local, y_offset]
    ):
        raise TypeError(
            "y_global, y_local, and y_offset must be NumPy arrays of integers"
        )

    return y_global + y_local + y_offset


def calc_cuboid_pixel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the pixel ID for a cuboid.

    Args:
    x (np.ndarray): NumPy array of x-coordinates.
    y (np.ndarray): NumPy array of y-coordinates.

    Returns:
    np.ndarray: NumPy array of calculated pixel IDs.
    """
    if not all(isinstance(arg, np.ndarray) and arg.dtype.kind == "i" for arg in [x, y]):
        raise TypeError("x and y must be NumPy arrays of integers")

    return CUBOID_Y_OFFSET_PER_STRIP * y + x + 1 + GLOBAL_HR_ID_OFFSET


def get_cuboid_pixels(
    sect_seg: np.ndarray,
    module: np.ndarray,
    cass: np.ndarray,
    wire: np.ndarray,
    strip: np.ndarray,
    counter: np.ndarray,
) -> np.ndarray:
    """
    Get the pixel IDs for cuboids based on various parameters.

    Args:
    sect_seg (np.ndarray): NumPy array of sector segment numbers.
    module (np.ndarray): NumPy array of module numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    wire (np.ndarray): NumPy array of wire numbers.
    strip (np.ndarray): NumPy array of strip numbers.
    counter (np.ndarray): NumPy array of counter numbers.

    Returns:
    np.ndarray: NumPy array of calculated pixel IDs.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [sect_seg, module, cass, wire, strip, counter]
    ):
        raise TypeError("All arguments must be NumPy arrays of integers")

    x_local = calc_cuboid_x_local(cass, counter)
    y_local = calc_cuboid_y_local(wire)
    y_global = calc_cuboid_y_global(strip)
    x_offsets, y_offsets = calc_cuboid_offset(sect_seg, module)
    x_local, y_local = rotate_cuboid(x_local, y_local, sect_seg)
    x = calc_cuboid_x(x_local, x_offsets)
    y = calc_cuboid_y(y_global, y_local, y_offsets)
    pixel = calc_cuboid_pixel(x, y)

    return pixel


"""
Functions for calculating the endcaps pixel ID. Expects inputs in ICD form, NOT GEANT4 form.
"""


def calc_endcap_sumo(sumo: np.ndarray) -> np.ndarray:
    """
    Calculate the endcap sumo-related array.

    Args:
    sumo (np.ndarray): NumPy array of sumo numbers.

    Returns:
    np.ndarray: NumPy array representing the calculated sumo-related values.
    """
    if not isinstance(sumo, np.ndarray) or sumo.dtype.kind != "i":
        raise TypeError("sumo must be a NumPy array of integers")

    arr = np.zeros_like(sumo)
    arr[sumo == 6] = 0
    arr[sumo == 5] = 20
    arr[sumo == 4] = 36
    arr[sumo == 3] = 48
    return arr


def calc_endcap_sect(sumo: np.ndarray, cass: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    """
    Calculate the endcap sector value.

    Args:
    sumo (np.ndarray): NumPy array of sumo numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    ctr (np.ndarray): NumPy array of counter numbers.

    Returns:
    np.ndarray: NumPy array representing the calculated sector values.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [sumo, cass, ctr]
    ):
        raise TypeError("sumo, cass, and ctr must be NumPy arrays of integers")

    return calc_endcap_sumo(sumo) + 2 * cass + ctr


def calc_endcap_offset(sect: np.ndarray) -> np.ndarray:
    """
    Calculate the endcap offset.

    Args:
    sect (np.ndarray): NumPy array of sector numbers.

    Returns:
    np.ndarray: NumPy array representing the calculated endcap offsets.
    """
    if not isinstance(sect, np.ndarray) or sect.dtype.kind != "i":
        raise TypeError("sect must be a NumPy array of integers")

    return ENDCAP_X_OFFSET_PER_SECTOR * sect


def calc_endcap_x(
    sect: np.ndarray, sumo: np.ndarray, cass: np.ndarray, ctr: np.ndarray
) -> np.ndarray:
    """
    Calculate the endcap x-coordinate.

    Args:
    sect (np.ndarray): NumPy array of sector numbers.
    sumo (np.ndarray): NumPy array of sumo numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    ctr (np.ndarray): NumPy array of counter numbers.

    Returns:
    np.ndarray: NumPy array representing the calculated x-coordinates.
    """
    return calc_endcap_offset(sect) + calc_endcap_sect(sumo, cass, ctr)


def calc_endcap_y(strip: np.ndarray, wire: np.ndarray) -> np.ndarray:
    """
    Calculate the endcap y-coordinate.

    Args:
    strip (np.ndarray): NumPy array of strip numbers.
    wire (np.ndarray): NumPy array of wire numbers.

    Returns:
    np.ndarray: NumPy array representing the calculated y-coordinates.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i" for arg in [strip, wire]
    ):
        raise TypeError("strip and wire must be NumPy arrays of integers")

    return ENDCAP_Y_OFFSET_PER_STRIP * strip + 15 - wire


def calc_endcap_pixel(
    x: np.ndarray, y: np.ndarray, num_sectors: int, is_backward: bool
) -> np.ndarray:
    """
    Calculate the endcap pixel ID.

    Args:
    x (np.ndarray): NumPy array of x-coordinates.
    y (np.ndarray): NumPy array of y-coordinates.
    num_sectors (int): Number of sectors.
    is_backward (bool): Flag indicating whether the calculation is for backward direction.

    Returns:
    np.ndarray: NumPy array representing the calculated pixel IDs.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i" for arg in [x, y]
    ) or not isinstance(num_sectors, int):
        raise TypeError(
            "x and y must be NumPy arrays of integers, and num_sectors must be an integer"
        )

    offset = (
        GLOBAL_BACKWARD_ENDCAP_ID_OFFSET
        if is_backward
        else GLOBAL_FORWARD_ENDCAP_ID_OFFSET
    )
    return ENDCAP_X_OFFSET_PER_SECTOR * num_sectors * y + x + 1 + offset


def get_endcap_pixels(
    sumo: np.ndarray,
    module: np.ndarray,
    cass: np.ndarray,
    ctr: np.ndarray,
    strip: np.ndarray,
    wire: np.ndarray,
    num_sectors: int = 1,
    is_backward: bool = True,
) -> np.ndarray:
    """
    Get the pixel IDs for endcap components based on various parameters.

    Args:
    sumo (np.ndarray): NumPy array of sumo numbers.
    module (np.ndarray): NumPy array of module numbers.
    cass (np.ndarray): NumPy array of cassette numbers.
    ctr (np.ndarray): NumPy array of counter numbers.
    strip (np.ndarray): NumPy array of strip numbers.
    wire (np.ndarray): NumPy array of wire numbers.
    num_sectors (int): Number of sectors (default 1).
    is_backward (bool): Flag indicating backward direction (default True).

    Returns:
    np.ndarray: NumPy array representing the calculated pixel IDs.
    """
    if not all(
        isinstance(arg, np.ndarray) and arg.dtype.kind == "i"
        for arg in [sumo, module, cass, ctr, strip, wire]
    ):
        raise TypeError(
            "All arguments except num_sectors and is_backward must be NumPy arrays of integers"
        )

    x = calc_endcap_x(module, sumo, cass, ctr)
    y = calc_endcap_y(strip, wire)
    pixel = calc_endcap_pixel(x, y, num_sectors, is_backward)

    return pixel


class DreamDetectorGeometry(BaseDetectorGeometry):
    def __init__(self, file_name, debug=False, fatal=True):
        self.debug = debug  # enable debug print
        self.fatal = fatal  # terminate on first error
        data: dict
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
        x, y, z, pixel_ids, vertices, list_pixel_ids = [], [], [], [], [], []
        retrieve_data_from_json(data, DETECTOR_NUMBER, pixel_ids)
        retrieve_data_from_json(data, X_PIXEL_OFFSET, x)
        retrieve_data_from_json(data, Y_PIXEL_OFFSET, y)
        retrieve_data_from_json(data, Z_PIXEL_OFFSET, z)

        self.id_dict = dict(zip(pixel_ids, zip(x, y, z)))


@pytest.fixture(scope="session")
def dream_geometry():
    json_file_name = "/home/jonas/code/dream_data/dream-with-mantle_detector.json"
    file_dir = path.dirname(path.abspath(__file__))
    script_dir = path.join(file_dir, "..", "dream")
    json_file_path = path.join(script_dir, json_file_name)
    return DreamDetectorGeometry(json_file_path)


# Testing that corner pixels in same layer are at right angles for mantle
def test_mantle_points(dream_geometry):
    modules = np.array([0, 0, 4, 4, 0, 0, 4, 4])
    cassettes = np.array([0, 0, 5, 5, 0, 0, 5, 5])
    ctr = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    wires = np.array([0, 0, 0, 0, 31, 31, 31, 31])
    strips = np.array([0, 255, 0, 255, 0, 255, 0, 255])
    num_mods = 5

    pixels = get_mantle_pixels(strips, modules, cassettes, ctr, wires, num_mods)

    front_upper_left = pixels[0]
    front_upper_right = pixels[1]
    front_lower_left = pixels[2]
    front_lower_right = pixels[3]

    back_upper_left = pixels[4]
    back_upper_right = pixels[5]
    back_lower_left = pixels[6]
    back_lower_right = pixels[7]

    angle_rad = np.deg2rad(90.0)

    a = dream_geometry.pix2angle(
        front_upper_left, front_upper_right, front_upper_left, front_lower_left
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        front_upper_right, front_upper_left, front_upper_right, front_lower_right
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        front_lower_left, front_lower_right, front_lower_left, front_upper_left
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        front_lower_right, front_lower_left, front_lower_right, front_upper_right
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        back_upper_left, back_upper_right, back_upper_left, back_lower_left
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        back_upper_right, back_upper_left, back_upper_right, back_lower_right
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        back_lower_left, back_lower_right, back_lower_left, back_upper_left
    )
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(
        back_lower_right, back_lower_left, back_lower_right, back_upper_right
    )
    assert dream_geometry.expect(a, angle_rad, precision)


# Test that the z position of the mantle pixels increases as the strips increase
def test_mantle_strip_increases_in_z(dream_geometry):
    modules = np.array([0 for i in range(256)])
    cassettes = np.array([0 for i in range(256)])
    ctr = np.array([0 for i in range(256)])
    wires = np.array([0 for i in range(256)])
    strips = np.array([i for i in range(256)])
    num_mods = 5

    pixels = get_mantle_pixels(strips, modules, cassettes, ctr, wires, num_mods)
    coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

    # Check that z increases in instrument coordinate system
    assert np.all(coords[1:, 2] > coords[:-1, 2])

    # It should be relatively flat in x
    assert dream_geometry.expect(np.max(np.abs(np.diff(coords[:, 0]))), 0, precision)

    # It should be relatively flat in y
    assert dream_geometry.expect(np.max(np.abs(np.diff(coords[:, 1]))), 0, precision)


# Test that the x position of the mantle pixels increases as the wires increase
def test_mantle_wire_increases_in_x(dream_geometry):
    modules = np.array([0 for i in range(32)])
    cassettes = np.array([0 for i in range(32)])
    ctr = np.array([0 for i in range(32)])
    wires = np.array([i for i in range(32)])
    strips = np.array([0 for i in range(32)])
    num_mods = 5

    pixels = get_mantle_pixels(strips, modules, cassettes, ctr, wires, num_mods)
    coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

    # Check that x increases in instrument coordinate system
    assert np.all(coords[1:, 0] > coords[:-1, 0])


# Test that the y position of the mantle pixels decreases as the modules increase
def test_mantle_cass_ctr_decreases_in_y(dream_geometry):
    modules = np.sort(np.array([i for i in range(5)] * 12))
    cassettes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 5)
    ctr = np.array([0, 1] * 30)
    wires = np.array([0 for i in range(60)])
    strips = np.array([0 for i in range(60)])
    num_mods = 5

    pixels = get_mantle_pixels(strips, modules, cassettes, ctr, wires, num_mods)
    coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

    # Check that y decreases in instrument coordinate system
    assert np.all(coords[1:, 1] < coords[:-1, 1])


# Testing that corner pixels of a cuboid are at right angles for HR
def test_first_cuboid_points(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector_segment = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    modules = np.array([5, 5, 5, 5, 5, 5, 5, 5])
    cassettes = np.array([7, 7, 0, 0, 7, 7, 0, 0])
    ctr = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    wires = np.array([15, 0, 15, 0, 15, 0, 15, 0])
    strips = np.array([0, 0, 0, 0, 31, 31, 31, 31])

    pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)
    assert pixels[0] == 1122305 + 32
    assert pixels[1] == 1122305 + 47

    front_upper_left = pixels[0]
    front_upper_right = pixels[1]
    front_lower_left = pixels[2]
    front_lower_right = pixels[3]

    back_upper_left = pixels[4]
    back_upper_right = pixels[5]
    back_lower_left = pixels[6]
    back_lower_right = pixels[7]

    angle_rad = np.deg2rad(90.0)

    a = dream_geometry.pix2angle(
        front_upper_left, front_upper_right, front_upper_left, front_lower_left
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        front_upper_right, front_upper_left, front_upper_right, front_lower_right
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        front_lower_left, front_lower_right, front_lower_left, front_upper_left
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        front_lower_right, front_lower_left, front_lower_right, front_upper_right
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        back_upper_left, back_upper_right, back_upper_left, back_lower_left
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        back_upper_right, back_upper_left, back_upper_right, back_lower_right
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        back_lower_left, back_lower_right, back_lower_left, back_upper_left
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(
        back_lower_right, back_lower_left, back_lower_right, back_upper_right
    )
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)


# Test that the x position of the cuboid pixels increases as the wires increase
def test_first_cuboid_wires_increase_in_x(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector_segment = np.array([0 for i in range(16)])
    modules = np.array([5 for i in range(16)])
    cassettes = np.array([7 for i in range(16)])
    ctr = np.array([1 for i in range(16)])
    wires = np.array([15 - i for i in range(16)])
    strips = np.array([0 for i in range(16)])

    pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)

    expected_pixels = np.array([1122305 + 32 + i for i in range(16)])
    coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

    assert np.all(pixels == expected_pixels)

    # Check that x decreases in instrument coordinate system (increases in detector coordinate system)
    assert np.all(coords[1:, 0] < coords[:-1, 0])

    # It will drift slightly in y
    assert np.max(np.abs(np.diff(coords[:, 1]))) < 0.02
    # It will drift slightly more in z
    assert np.max(np.abs(np.diff(coords[:, 2]))) < 0.5


def test_all_cuboids_sector_0_wires_increase_in_x(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector = 0

    for module in range(9):
        if (sector, module) not in ROTATE_HR:
            print(
                f"Skipping sector {sector} module {module} because it does not exist in the lookup table"
            )
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7 for i in range(16)])
        ctr = np.array([1 for i in range(16)])
        wires = np.array([15 - i for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(
            sector_segment, modules, cassettes, wires, strips, ctr
        )
        coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

        # Check that x decreases in instrument coordinate system (increases in detector coordinate system)
        assert np.all(coords[1:, 0] < coords[:-1, 0])

        # It will drift slightly in y
        assert np.max(np.abs(np.diff(coords[:, 1]))) < 0.02
        # It will drift slightly more in z
        assert np.max(np.abs(np.diff(coords[:, 2]))) < 0.5


def test_all_cuboids_sector_1_cass_ctr_increase_in_x(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector = 1

    for module in range(9):
        if (sector, module) not in ROTATE_HR:
            print(
                f"Skipping sector {sector} module {module} because it does not exist in the lookup table"
            )
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0])
        ctr = np.array([1, 0] * 8)
        wires = np.array([0 for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(
            sector_segment, modules, cassettes, wires, strips, ctr
        )
        coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

        # Check that x decreases in instrument coordinate system (increases in detector coordinate system)
        assert np.all(coords[1:, 0] < coords[:-1, 0])

        # It will drift slightly in y
        assert np.max(np.abs(np.diff(coords[:, 1]))) < 0.02
        # It will drift slightly more in z
        assert np.max(np.abs(np.diff(coords[:, 2]))) < 0.5


def test_all_cuboids_sector_2_wires_increase_in_x(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector = 2

    for module in range(9):
        if (sector, module) not in ROTATE_HR:
            print(
                f"Skipping sector {sector} module {module} because it does not exist in the lookup table"
            )
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7 for i in range(16)])
        ctr = np.array([1 for i in range(16)])
        wires = np.array([i for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(
            sector_segment, modules, cassettes, wires, strips, ctr
        )
        coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

        # Check that x decreases in instrument coordinate system (increases in detector coordinate system)
        assert np.all(coords[1:, 0] < coords[:-1, 0])

        # It will drift slightly in y
        assert np.max(np.abs(np.diff(coords[:, 1]))) < 0.02
        # It will drift slightly more in z
        assert np.max(np.abs(np.diff(coords[:, 2]))) < 0.5


def test_all_cuboids_sector_3_cass_ctr_increase_in_x(dream_geometry):
    # Points taken from ICD drawing and assuming that the first cuboid is rotated 90 degrees counter clockwise
    sector = 3

    for module in range(9):
        if (sector, module) not in ROTATE_HR:
            print(
                f"Skipping sector {sector} module {module} because it does not exist in the lookup table"
            )
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        ctr = np.array([0, 1] * 8)
        wires = np.array([15 for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(
            sector_segment, modules, cassettes, wires, strips, ctr
        )
        coords = np.array([dream_geometry.id_dict[pixel] for pixel in pixels])

        # Check that x decreases in instrument coordinate system (increases in detector coordinate system)
        assert np.all(coords[1:, 0] < coords[:-1, 0])

        # It will drift slightly in y
        assert np.max(np.abs(np.diff(coords[:, 1]))) < 0.02
        # It will drift slightly more in z
        assert np.max(np.abs(np.diff(coords[:, 2]))) < 0.5


# Not sure about this test. I'd like a separate CAD drawing or something in order to test for exact expected distances.
# If we take the distances from the generation script, of course it will match.
def test_forward_endcap_pixel_distance(dream_geometry):
    sumo = np.array([6, 6])
    modules = np.array([0, 0])
    cassettes = np.array([0, 0])
    ctr = np.array([0, 1])
    wires = np.array([15, 15])
    strips = np.array([0, 0])

    expected_pixels = np.array([1, 2])
    expected_distance = 7.45  # Where would we get a good distance from?

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False
    )

    distance = dream_geometry.dist(pixels[0], pixels[1])

    assert np.all(pixels == expected_pixels)
    assert dream_geometry.expect(distance, expected_distance, 1)


# Testing that the radius decreases for the sumos. Sumo 6 should be the furthest out and sumo 3 should be the closest.
def test_forward_endcap_pixel_decreasing_radius_per_sumo(dream_geometry):
    sumo = np.array([6, 5, 4, 3])
    modules = np.array([0, 0, 0, 0])
    cassettes = np.array([0, 0, 0, 0])
    ctr = np.array([0, 0, 0, 0])
    wires = np.array([15, 15, 15, 15])
    strips = np.array([0, 0, 0, 0])

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False
    )
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

    # Make sure that the radius is smaller for each sumo
    assert np.all(R[1:] < R[:-1])


# We are expecting that the forward endcap rotates clockwise along the z axis
def test_forward_endcap_pixel_rotates_clockwise_per_sector(dream_geometry):
    sumo = np.array([6 for i in range(5)])
    modules = np.array([i for i in range(5)])
    cassettes = np.array([0 for i in range(5)])
    ctr = np.array([0 for i in range(5)])
    wires = np.array([15 for i in range(5)])
    strips = np.array([0 for i in range(5)])

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False
    )
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    phi = np.rad2deg(np.arctan2(coords[:, 1], coords[:, 0]))

    # Compute angle deltas
    angle_deltas = np.diff(phi)

    # Adjust for angle wrapping
    angle_deltas = np.where(angle_deltas < -180, angle_deltas + 360, angle_deltas)
    angle_deltas = np.where(angle_deltas > 180, angle_deltas - 360, angle_deltas)

    # Check if deltas are positive (clockwise)
    clockwise_movement = all(delta >= 0 for delta in angle_deltas)

    assert clockwise_movement
    assert np.allclose(angle_deltas, 12, atol=endcap_rotation_precision)


# We are expecting that the forward endcap rotates clockwise along the z axis
def test_forward_endcap_first_and_last_pixel_clockwise(dream_geometry):
    sumo = np.array([6, 3])
    modules = np.array([0, 4])
    cassettes = np.array([0, 3])
    ctr = np.array([0, 1])
    wires = np.array([15, 0])
    strips = np.array([0, 15])

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False
    )
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    phi = np.rad2deg(np.arctan2(coords[:, 1], coords[:, 0]))

    # Compute angle deltas
    angle_deltas = np.diff(phi)

    # Adjust for angle wrapping
    angle_deltas = np.where(angle_deltas < -180, angle_deltas + 360, angle_deltas)
    angle_deltas = np.where(angle_deltas > 180, angle_deltas - 360, angle_deltas)

    # Check if deltas are positive (clockwise)
    clockwise_movement = all(delta >= 0 for delta in angle_deltas)

    # The rotation is clockwise in the coordinate system of the instrument
    assert clockwise_movement

    # the first y coordinates should be much lower than the last y coordinates
    assert coords[0, 1] < coords[1, 1]


# We are expecting that the backwards endcap rotates anti-clockwise along the z axis
# But still clockwise when standing at the sample position an looking at the detector
# The first pixel should be at a lower y coordinate than the last pixel for this number of sectors
def test_backward_endcap_pixel_rotates_clockwise_per_sector(dream_geometry):
    sumo = np.array([6 for i in range(11)])
    modules = np.array([i for i in range(11)])
    cassettes = np.array([0 for i in range(11)])
    ctr = np.array([0 for i in range(11)])
    wires = np.array([15 for i in range(11)])
    strips = np.array([0 for i in range(11)])

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=11, is_backward=True
    )
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    phi = np.rad2deg(np.arctan2(coords[:, 1], coords[:, 0]))

    # Compute angle deltas
    angle_deltas = np.diff(phi)

    # Adjust for angle wrapping
    angle_deltas = np.where(angle_deltas < -180, angle_deltas + 360, angle_deltas)
    angle_deltas = np.where(angle_deltas > 180, angle_deltas - 360, angle_deltas)

    # Check if deltas are positive (clockwise)
    clockwise_movement = all(delta >= 0 for delta in angle_deltas)

    # The endcap is inverted so the rotation is counter-clockwise in the coordinate system of the instrument
    assert not clockwise_movement
    assert np.allclose(angle_deltas, -12, atol=endcap_rotation_precision)


# We are expecting that the backwards endcap rotates anti-clockwise along the z axis
# But still clockwise when standing at the sample position an looking at the detector
# The first pixel should be at a higher y coordinate than the last pixel for this number of sectors
def test_backward_endcap_first_and_last_pixel_clockwise(dream_geometry):
    sumo = np.array([6, 3])
    modules = np.array([0, 10])
    cassettes = np.array([0, 3])
    ctr = np.array([0, 1])
    wires = np.array([15, 0])
    strips = np.array([0, 15])

    pixels = get_endcap_pixels(
        sumo, modules, cassettes, ctr, strips, wires, num_sectors=11, is_backward=True
    )
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    phi = np.rad2deg(np.arctan2(coords[:, 1], coords[:, 0]))

    # Compute angle deltas
    angle_deltas = np.diff(phi)

    # Adjust for angle wrapping
    angle_deltas = np.where(angle_deltas < -180, angle_deltas + 360, angle_deltas)
    angle_deltas = np.where(angle_deltas > 180, angle_deltas - 360, angle_deltas)

    # Check if deltas are positive (clockwise)
    clockwise_movement = all(delta >= 0 for delta in angle_deltas)

    # The endcap is inverted so the rotation is counter-clockwise in the coordinate system of the instrument
    assert not clockwise_movement

    # the first y coordinates should be much higher than the last y coordinates
    assert coords[0, 1] > coords[1, 1]
