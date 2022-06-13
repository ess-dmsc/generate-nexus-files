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
pa_angle         = 90 - 76.55 # pack angle [degrees]

precision      = 0.000001  # general precision aim (1/1000 mm) [m]
pp_precision   = 0.000008  # [m]
pa_precision   = 0.000006  # [degree]
tt_z_precision = 0.0000011 # [m]


class LokiGeometry(BaseDetectorGeometry):
    def strawtopixel(self, straw, pos):
        return straw * SR + pos + 1

    # return global straw from bank, tube and local straw
    def get_straw(self, bank, tube, lstraw):
        # number of tubes in the
        loki = [56, 16, 12, 16, 12, 28, 32, 20, 32]

        if (bank > 8) or (lstraw > 6):
            raise Exception("Invalid bank or lstraw")

        if tube > loki[bank] * 4 - 1:
            raise Exception("Invalid tube")

        tube_offs = 0
        for i in range(bank):
            tube_offs = tube_offs + loki[i]
        print("tube offset {}".format(tube_offs))
        return (tube_offs * 4 + tube) * SPT + lstraw



@pytest.fixture(scope='session')
def loki_geometry():
    json_file_name = 'config_larmor.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    #run_create_geometry(json_file_path)
    return LokiGeometry(json_file_path)


def test_get_straw(loki_geometry):
    #assert loki_geometry.get_straw(0, 0, 7) == 0 # raises
    #assert loki_geometry.get_straw(9, 0, 0) == 0 # raises
    #assert loki_geometry.get_straw(0, 56 * 4, 0) == 0 # raises
    assert loki_geometry.get_straw(0, 0, 0) == 0
    assert loki_geometry.get_straw(0, 1, 6) == 13
    assert loki_geometry.get_straw(0, 56 * 4 - 1, 6) == 1567
    assert loki_geometry.get_straw(1, 0, 0) == 1568
    assert loki_geometry.get_straw(2, 0, 0) == 2016


def test_max_valid_pixel(loki_geometry):
    max_valid_pixel = NS * SR
    try:
        coord = loki_geometry.p2c(max_valid_pixel)
    except:
        assert False


def test_first_bad_pixel(loki_geometry):
    invalid_pixel = NS * SR + 1
    try:
        coord = amor_geometry.p2c(invalid_pixel)
    except:
        assert True


# Testing that corner pixels in same layer are at right angles
# so far bank 0 only
@pytest.mark.parametrize('layer', [i for i in range(4)])
def test_some_icd_values(loki_geometry, layer):
        pixels_per_layer = NS * SR
        pixel_offset = pixels_per_layer * layer

        top_left = 1 + pixel_offset
        top_right = 512 + pixel_offset
        bottom_left = 99841 + pixel_offset
        bottom_right = 100352 + pixel_offset

        a = loki_geometry.pix2angle(top_left, top_right, top_left, bottom_left)
        assert loki_geometry.expect(a, 90.0, precision)

        a = loki_geometry.pix2angle(top_right, bottom_right, top_right, top_left)
        assert loki_geometry.expect(a, 90.0, precision)

        a = loki_geometry.pix2angle(bottom_right, bottom_left, bottom_right, top_right)
        assert loki_geometry.expect(a, 90.0, precision)

        a = loki_geometry.pix2angle(bottom_left, top_left, bottom_left, bottom_right)
        assert loki_geometry.expect(a, 90.0, precision)


# All straws: distance between neighbouring pixels
@pytest.mark.parametrize('straw', [i for i in range(NS * NL)])
def test_pixel_pixel_dist(loki_geometry, straw):
    offset = straw * SR
    for i in range(511):
        pix1 = offset + i + 1
        pix2 = offset + i + 2
        d = loki_geometry.dist(pix1, pix2)
        assert loki_geometry.expect(d, pp_dist, pp_precision)


# All tubes: distance between first and second straw
@pytest.mark.parametrize('tube', [i for i in range(128)])
def test_straw_straw_dist(loki_geometry, tube):
    offset = tube * 7 * SR
    pix1 = offset + 1
    pix2 = offset + 1 + 512
    c1 = loki_geometry.p2c(pix1)
    c2 = loki_geometry.p2c(pix2)
    d = loki_geometry.dist(pix1, pix2)
    assert loki_geometry.expect(d, ss_dist, precision)


# Distance between tubes (in y-direction, but this is
# not currently checked)
# Layer by layer
@pytest.mark.parametrize('tube', [i for i in range(int(NS/7) - 1)])
def test_tube_tube_dist(loki_geometry, tube):
    for layer in range(4):
        straw_offset = layer * NS
        pix1 = loki_geometry.strawtopixel(straw_offset + (tube + 0) * SPT, 0)
        pix2 = loki_geometry.strawtopixel(straw_offset + (tube + 1) * SPT, 0)

        # Calculate Euclidian distance
        d = loki_geometry.dist(pix1, pix2)
        assert loki_geometry.expect(d, tt_il_dist, precision)

        # Same test not using Euclidian distance but y coordinates
        c1 = loki_geometry.p2c(pix1)
        c2 = loki_geometry.p2c(pix2)
        dy = np.abs(c2 - c1)[1]
        assert loki_geometry.expect(dy, tt_il_dist, precision)


# Detector packs are tilted by about 13 degrees
# in addition we test the distance of the z-projection between tubes
# in adjacent layers
@pytest.mark.parametrize('tube', [i for i in range(int(NS/7))])
def test_angle_of_tubes(loki_geometry, tube):
    for layer in range(4 - 1):
        layer_straw_offset = NS * layer
        first_straw_l1 = layer_straw_offset + tube * SPT
        first_straw_l2 = layer_straw_offset + tube * SPT + NS

        # first pixel in first straw of first tube
        pix1 = loki_geometry.strawtopixel(first_straw_l1, 0)
        # first pixel in first straw of same tube in next layer
        pix2 = loki_geometry.strawtopixel(first_straw_l2, 0)

        v1 = loki_geometry.p2v(pix1, pix2)
        v2 = np.array([0.0, 0.0, 1.0])
        angle = loki_geometry.angle(v1, v2)
        angled = loki_geometry.r2d(angle)

        assert loki_geometry.expect(angled, pa_angle, pa_precision)

        c1 = loki_geometry.p2c(pix1)
        c2 = loki_geometry.p2c(pix2)
        dz = np.abs(c2 - c1)[2]
        assert loki_geometry.expect(dz, tt_z_dist, tt_z_precision)

# Just some debugging from table of straw coordinates used to
# 'determine' the straw-straw distance originally assumred to be 8mm
# according to the drawing. But was observed to be 7.75mm so the drawing
# was probably rounding up to nearest mm.
def test_specific_from_drawing(loki_geometry):
    c1 = np.array([0, -1.14, -7.67])/1000
    c2 = np.array([0, -7.21, -2.85])/1000
    d = np.linalg.norm(c2 - c1)
    assert loki_geometry.expect(d, ss_dist, precision)
