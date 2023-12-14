import json
from os import path

import numpy as np
import pytest

from examples.utils.detector_geometry_from_json import retrieve_data_from_json, \
    DETECTOR_NUMBER, BaseDetectorGeometry, X_PIXEL_OFFSET, Y_PIXEL_OFFSET, Z_PIXEL_OFFSET


precision = 0.00002  # general precision aim (2/100 mm) [m]
cuboid_precision = 0.001  # precision aim for cuboid (10/100 mm) [m]
endcap_rotation_precision = 0.01  # precision aim for endcap rotation (1/100 degree) [deg]


"""
Functions for calculating the mantle pixel ID. Expects inputs in ICD form, NOT GEANT4 form.
"""


def calc_mantle_x(strip):
    return strip


def calc_mantle_y(mod, cass, ctr, wire, num_mods):
    # num_mods = len(np.unique(mod))
    return num_mods * 12 * wire + 12 * mod + 2 * cass + ctr


def calc_mantle_pixel(x, y):
    return 256 * y + x + 1 + 229376


def get_mantle_pixels(strip, module, cass, counter, wire, num_mods):
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
    ( 0, 32), (16, 32), (32, 32), (48, 32), (64, 32), (80, 32), (96, 32),
    ( 0, 48), (16, 48), (32, 48),           (64, 48), (80, 48), (96, 48),
    ( 0, 64), (16, 64), (32, 64),           (64, 64), (80, 64), (96, 64),
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


def calc_cuboid_x_local(cass, ctr):
    return 2 * cass + ctr


def calc_cuboid_y_local(wire):
    return 15 - wire


def rotate_cuboid(x, y, sect_seg):
    x_new, y_new = np.empty_like(x), np.empty_like(y)
    transformations = {
        1: (15 - x, 15 - y),
        2: (15 - y, x),
        3: (x, y),
        0: (y, 15 - x)
    }
    for seg, (trans_x, trans_y) in transformations.items():
        mask = sect_seg == seg
        x_new[mask], y_new[mask] = trans_x[mask], trans_y[mask]

    return x_new, y_new


def calc_cuboid_y_global(strip):
    return 112 * strip


def calc_cuboid_offset(sect_seg, module):
    lut_idx = ROTATE_HR
    lut_offsets = OFFSETS_HR

    x = []
    y = []
    for ss, m in zip(sect_seg, module):
        if (ss, m) not in lut_idx:
            print(f"sector-segment {ss} module {m} not found in lookup table")
            continue
        idx = lut_idx.index((ss, m))
        x_o, y_o = lut_offsets[idx]
        x.append(x_o)
        y.append(y_o)

    return np.array(x), np.array(y)


def calc_cuboid_x(x_local, x_offset):
    return x_local + x_offset


def calc_cuboid_y(y_global, y_local, y_offset):
    return y_global + y_local + y_offset


def calc_cuboid_pixel(x, y):
    return 112 * y + x + 1 + 1122304


def get_cuboid_pixels(sect_seg, module, cass, wire, strip, counter):
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


def calc_endcap_sumo(sumo):
    arr = np.zeros_like(sumo)
    arr[sumo == 6] = 0
    arr[sumo == 5] = 20
    arr[sumo == 4] = 36
    arr[sumo == 3] = 48
    return arr


def calc_endcap_sect(sumo, cass, ctr):
    return calc_endcap_sumo(sumo) + 2 * cass + ctr


def calc_endcap_offset(sect):
    return 56 * sect


def calc_endcap_x(sect, sumo, cass, ctr):
    return calc_endcap_offset(sect) + calc_endcap_sect(sumo, cass, ctr)


def calc_endcap_y(strip, wire):
    return 16 * strip + 15 - wire


def calc_endcap_pixel(x, y, num_sectors, is_backward):
    if is_backward:
        offset = 71680
    else:
        offset = 0
    return 56 * num_sectors * y + x + 1 + offset


def get_endcap_pixels(sumo, module, cass, ctr, strip, wire, num_sectors=1, is_backward=True):
    x = calc_endcap_x(
        module,
        sumo,
        cass,
        ctr
    )
    y = calc_endcap_y(
        strip,
        wire
    )
    pixel = calc_endcap_pixel(x, y, num_sectors, is_backward)
    return pixel


class DreamDetectorGeometry(BaseDetectorGeometry):
    def __init__(self, file_name, debug=False, fatal=True):
        self.debug = debug  # enable debug print
        self.fatal = fatal  # terminate on first error
        data: dict
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        x, y, z, pixel_ids, vertices, list_pixel_ids = [], [], [], [], [], []
        retrieve_data_from_json(data, DETECTOR_NUMBER, pixel_ids)
        retrieve_data_from_json(data, X_PIXEL_OFFSET, x)
        retrieve_data_from_json(data, Y_PIXEL_OFFSET, y)
        retrieve_data_from_json(data, Z_PIXEL_OFFSET, z)

        self.id_dict = dict(zip(pixel_ids, zip(x, y, z)))


json_file_name = '/home/jonas/code/dream_data/DREAM_inverted_forward.json'
# json_file_name = '/home/jonas/code/dream_data/DREAM_baseline_with_mantle_detector_new.json'

file_dir = path.dirname(path.abspath(__file__))
script_dir = path.join(file_dir, '..', 'dream')
json_file_path = path.join(script_dir, json_file_name)
GEO_OBJ = DreamDetectorGeometry(json_file_path)



@pytest.fixture(scope='session')
def dream_geometry():
    # json_file_name = '/home/jonas/code/dream_data/DREAM_baseline_with_mantle_detector_new.json'
    # json_file_name = '/home/jonas/code/dream_data/DREAM_inverted_forward.json'
    # file_dir = path.dirname(path.abspath(__file__))
    # script_dir = path.join(file_dir, '..', 'dream')
    # json_file_path = path.join(script_dir, json_file_name)
    # return DreamDetectorGeometry(json_file_path)

    return GEO_OBJ  # If we load the JSON on each test it takes forever to run


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

    a = dream_geometry.pix2angle(front_upper_left, front_upper_right, front_upper_left, front_lower_left)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(front_upper_right, front_upper_left, front_upper_right, front_lower_right)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(front_lower_left, front_lower_right, front_lower_left, front_upper_left)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(front_lower_right, front_lower_left, front_lower_right, front_upper_right)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(back_upper_left, back_upper_right, back_upper_left, back_lower_left)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(back_upper_right, back_upper_left, back_upper_right, back_lower_right)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(back_lower_left, back_lower_right, back_lower_left, back_upper_left)
    assert dream_geometry.expect(a, angle_rad, precision)

    a = dream_geometry.pix2angle(back_lower_right, back_lower_left, back_lower_right, back_upper_right)
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
    modules = np.sort(np.array([i for i in range(5)]*12))
    cassettes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]*5)
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

    a = dream_geometry.pix2angle(front_upper_left, front_upper_right, front_upper_left, front_lower_left)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(front_upper_right, front_upper_left, front_upper_right, front_lower_right)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(front_lower_left, front_lower_right, front_lower_left, front_upper_left)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(front_lower_right, front_lower_left, front_lower_right, front_upper_right)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(back_upper_left, back_upper_right, back_upper_left, back_lower_left)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(back_upper_right, back_upper_left, back_upper_right, back_lower_right)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(back_lower_left, back_lower_right, back_lower_left, back_upper_left)
    assert dream_geometry.expect(a, angle_rad, cuboid_precision)

    a = dream_geometry.pix2angle(back_lower_right, back_lower_left, back_lower_right, back_upper_right)
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
            print(f"Skipping sector {sector} module {module} because it does not exist in the lookup table")
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7 for i in range(16)])
        ctr = np.array([1 for i in range(16)])
        wires = np.array([15 - i for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)
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
            print(f"Skipping sector {sector} module {module} because it does not exist in the lookup table")
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0])
        ctr = np.array([1, 0] * 8)
        wires = np.array([0 for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)
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
            print(f"Skipping sector {sector} module {module} because it does not exist in the lookup table")
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([7 for i in range(16)])
        ctr = np.array([1 for i in range(16)])
        wires = np.array([i for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)
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
            print(f"Skipping sector {sector} module {module} because it does not exist in the lookup table")
            continue

        sector_segment = np.array([sector for i in range(16)])
        modules = np.array([module for i in range(16)])
        cassettes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        ctr = np.array([0, 1] * 8)
        wires = np.array([15 for i in range(16)])
        strips = np.array([0 for i in range(16)])

        pixels = get_cuboid_pixels(sector_segment, modules, cassettes, wires, strips, ctr)
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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False)

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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False)
    coords = np.array([dream_geometry.p2c(pixel) for pixel in pixels])

    R = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)

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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False)
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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=5, is_backward=False)
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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=11, is_backward=True)
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

    pixels = get_endcap_pixels(sumo, modules, cassettes, ctr, strips, wires, num_sectors=11, is_backward=True)
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
