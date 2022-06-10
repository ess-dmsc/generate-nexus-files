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


NL = 4                # number of layers
NS = 224              # number of straws (per layer)
SR = 512              # straw resolution (pixels along straw)

strawlen = 1.0        # [m]
strawdiam = 0.00775   # [m] - drawing says 8mm

pp_dist = strawlen/(SR - 1)  # pixel - pixel distance along a straw [m]
ss_dist = strawdiam          # straw - straw distance
tt_il_dist       = 0.0284    # [m] tube tube distance (same layer)
pa_angle = 90 - 76.55        # pack angle [degrees]

precision    = 0.000001 # general precision aim (1/1000 mm) [m]
pp_precision = 0.000008 # [m]
ss_precision = 0.000001 # [m]
pa_precision = 0.000006 # [degree]



class LokiGeometry(BaseDetectorGeometry):
    def strawtopixel(self, straw, pos):
        return straw * SR + pos + 1

@pytest.fixture(scope='session')
def loki_geometry():
    json_file_name = 'config_larmor.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    #run_create_geometry(json_file_path)
    return LokiGeometry(json_file_path)


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
    assert loki_geometry.expect(d, ss_dist, ss_precision)


# Front tubes: Distance from first to second tube
def test_tube_tube_dist(loki_geometry):
    pix1 = 1
    pix2 = 1 + 512 * 7
    c1 = loki_geometry.p2c(pix1)
    c2 = loki_geometry.p2c(pix2)
    #print("{}, {}".format(c1, c2))
    d = loki_geometry.dist(pix1, pix2)
    assert loki_geometry.expect(d, tt_il_dist, precision)


def test_angle_of_tubes(loki_geometry):
    # first pixel in first straw of first tube
    p1 = loki_geometry.strawtopixel(0, 0)
    # first pixel in first straw of same tube in next layer
    p2 = loki_geometry.strawtopixel(224, 0)

    v1 = loki_geometry.p2v(p1, p2)
    v2 = np.array([0.0, 0.0, 1.0])
    angle = loki_geometry.angle(v1, v2)
    angled = loki_geometry.r2d(angle)

    assert loki_geometry.expect(angled, pa_angle, pa_precision)


def test_specific_from_drawing(loki_geometry):
    c1 = np.array([0, -1.14, -7.67])/1000
    c2 = np.array([0, -7.21, -2.85])/1000
    d = np.linalg.norm(c2-c1)
    #print("dist {}".format(d))
    #assert d == ss_dist
    assert loki_geometry.expect(d, ss_dist, precision)
