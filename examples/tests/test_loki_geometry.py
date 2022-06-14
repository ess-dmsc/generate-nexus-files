#
# Unit tests for generating and validating LoKI instrument geometry
#
# These tests are written with reference to
# 1) Logical geometry definitions from the ICD
#    https://project.esss.dk/owncloud/index.php/s/CMxvxkXSXyKGxyu
# 2) Technical drawings
#    ISIS: SI-7615-097

import numpy as np
import os
import pytest

#from examples.amor.amor import run_create_geometry
from examples.utils.detector_geometry_from_json import BaseDetectorGeometry



SPT = 7               # straws per tube
NL = 4                # number of layers
NS = 224              # number of straws (per layer)
SR = 512              # straw resolution (pixels along straw)

strawlen = 1.0        # [m]
strawdiam = 0.00775   # [m] - drawing says 8mm

pp_dist = strawlen/(SR - 1)   # pixel - pixel distance along a straw [m]
ss_dist = strawdiam           # straw - straw distance
tt_il_dist       = 0.0284     # [m] tube tube distance (same layer)
tt_z_dist        = 0.02626    # [m] tube-tube distance (between layers)
pa_angle         = 90 - 76.55 # pack angle [deg]
sa_angle         = 60         # angle between adjacent outer straws [deg]

precision      = 0.000001  # general precision aim (1/1000 mm) [m]

# deviations from general precision (minimum value for passing test)
pp_precision   = 0.000008  # [m]
pa_precision   = 0.000006  # [deg]
sa_precision   = 0.03      # straw rotation angle [deg]
tt_z_precision = 0.0000011 # [m]


class LokiGeometry(BaseDetectorGeometry):
    # number of tubes per layer in a bank for the final instrument
    tubes_per_layer = [56, 16, 12, 16, 12, 28, 32, 20, 32]

    def get_pixel(self, bank, tube, lstraw, pos):
        return self.straw_to_pixel(self.get_straw(bank, tube, lstraw), pos)

    # helper function, may not need to be called directly
    def straw_to_pixel(self, straw, pos):
        return straw * SR + pos + 1

    # return global straw from bank, tube and local straw
    # helper function, may not need to be called directly
    def get_straw(self, bank, tube, lstraw):
        if (bank > 8) or (lstraw > 6):
            raise Exception("Invalid bank or lstraw")

        if tube > self.tubes_per_layer[bank] * 4 - 1:
            raise Exception("Invalid tube")

        tube_offs = 0
        for i in range(bank):
            tube_offs = tube_offs + self.tubes_per_layer[i]
        print("tube offset {}".format(tube_offs))
        return (tube_offs * 4 + tube) * SPT + lstraw



@pytest.fixture(scope='session')
def geom(): # formerly known as loki_geometry
    json_file_name = 'config_larmor.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    #run_create_geometry(json_file_path)
    return LokiGeometry(json_file_path)


def test_get_straw(geom):
    #assert geom.get_straw(0, 0, 7) == 0 # raises
    #assert geom.get_straw(9, 0, 0) == 0 # raises
    #assert geom.get_straw(0, 56 * 4, 0) == 0 # raises
    assert geom.get_straw(0, 0, 0) == 0
    assert geom.get_straw(0, 1, 6) == 13
    assert geom.get_straw(0, 56 * 4 - 1, 6) == 1567
    assert geom.get_straw(1, 0, 0) == 1568
    assert geom.get_straw(2, 0, 0) == 2016


def test_max_valid_pixel(geom):
    max_valid_pixel = NS * SR
    try:
        coord = geom.p2c(max_valid_pixel)
    except:
        assert False


def test_first_bad_pixel(geom):
    invalid_pixel = NS * SR + 1
    try:
        coord = amor_geometry.p2c(invalid_pixel)
    except:
        assert True


# Testing that corner pixels in same layer are at right angles
# so far bank 0 only
@pytest.mark.parametrize('layer', [i for i in range(4)])
def test_some_icd_values(geom, layer):
        pixels_per_layer = NS * SR
        pixel_offset = pixels_per_layer * layer

        top_left = 1 + pixel_offset
        top_right = 512 + pixel_offset
        bottom_left = 99841 + pixel_offset
        bottom_right = 100352 + pixel_offset

        a = geom.pix2angle(top_left, top_right, top_left, bottom_left)
        assert geom.expect(a, 90.0, precision)

        a = geom.pix2angle(top_right, bottom_right, top_right, top_left)
        assert geom.expect(a, 90.0, precision)

        a = geom.pix2angle(bottom_right, bottom_left, bottom_right, top_right)
        assert geom.expect(a, 90.0, precision)

        a = geom.pix2angle(bottom_left, top_left, bottom_left, bottom_right)
        assert geom.expect(a, 90.0, precision)


# All straws: distance between neighbouring pixels
@pytest.mark.parametrize('straw', [i for i in range(NS * NL)])
def test_pixel_pixel_dist(geom, straw):
    offset = straw * SR
    for i in range(511):
        pix1 = offset + i + 1
        pix2 = offset + i + 2
        d = geom.dist(pix1, pix2)
        assert geom.expect(d, pp_dist, pp_precision)


# All tubes: distance between first and second straw
# Currently assumes only bank 0 and pos 0, but can be extended
@pytest.mark.parametrize('tube', [i for i in range(128)])
def test_straw_straw_dist(geom, tube, bank = 0):
    pix1 = geom.get_pixel(bank, tube, 0, 0)
    pix2 = geom.get_pixel(bank, tube, 1, 0)
    d = geom.dist(pix1, pix2)
    assert geom.expect(d, ss_dist, precision)


# All tubes: test angle between adjacent straws with center straw as origin
# Currently assumes only bank 0 and pos 0, but can be extended
@pytest.mark.parametrize('tube', [i for i in range(128)])
def test_straw_straw_angle(geom, tube, bank = 0):
    pix1 = geom.get_pixel(bank, tube, 3, 0) # centre straw
    pix2 = geom.get_pixel(bank, tube, 0, 0) # lstraw 0
    pix3 = geom.get_pixel(bank, tube, 1, 0) # lstraw 1
    angle = geom.pix2angle(pix1, pix2, pix1, pix3)
    assert geom.expect(angle, sa_angle, sa_precision)


# Distance between adjacent tubes in same layer (Euclidian and y-projection)
# Currently assumes only bank 0 and pos 0, but can be extended
@pytest.mark.parametrize('tube', [i for i in range(int(NS/7) - 1)])
def test_tube_tube_dist(geom, tube, bank = 0):
    for layer in range(4):
        pix1 = geom.get_pixel(bank, tube, 0, 0)
        pix2 = geom.get_pixel(bank, tube + 1, 0, 0)

        # Calculate Euclidian distance
        d = geom.dist(pix1, pix2)
        assert geom.expect(d, tt_il_dist, precision)

        # Same test not using Euclidian distance but y coordinates
        c1 = geom.p2c(pix1)
        c2 = geom.p2c(pix2)
        dy = np.abs(c2 - c1)[1]
        assert geom.expect(dy, tt_il_dist, precision)


# Detector packs are tilted by about 13 degrees
# in addition we test the distance of the z-projection between tubes
# in adjacent layers
# This ought to use the tubes_per_layer rather than 32 hardcoded
@pytest.mark.parametrize('tube', [i for i in range(32)]) #
def test_angle_of_tubes(geom, tube, bank = 0):
    for layer in range(4 - 1):
        pix1 = geom.get_pixel(bank, tube + layer * 32    , 0, 0)
        pix2 = geom.get_pixel(bank, tube + (layer + 1) * 32, 0, 0)

        angle = geom.pixandv2angle(pix1, pix2, np.array([0.0, 0.0, 1.0]))
        assert geom.expect(angle, pa_angle, pa_precision)

        c1 = geom.p2c(pix1)
        c2 = geom.p2c(pix2)
        dz = np.abs(c2 - c1)[2]
        assert geom.expect(dz, tt_z_dist, tt_z_precision)

# Just some debugging from table of straw coordinates used to
# 'determine' the straw-straw distance originally assumred to be 8mm
# according to the drawing. But was observed to be 7.75mm so the drawing
# was probably rounding up to nearest mm.
def test_specific_from_drawing(geom):
    c1 = np.array([0, -1.14, -7.67])/1000
    c2 = np.array([0, -7.21, -2.85])/1000
    d = np.linalg.norm(c2 - c1)
    assert geom.expect(d, ss_dist, precision)
