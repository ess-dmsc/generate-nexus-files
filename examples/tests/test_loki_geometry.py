#
# Unit tests for generating and validating LoKI instrument geometry
#
# These tests are written with reference to
# 1) Logical geometry definitions from the ICD
#    https://project.esss.dk/owncloud/index.php/s/CMxvxkXSXyKGxyu
# 2) Technical drawings
#    ISIS: SI-7615-097
#
#    BE AWARE that 2) doesn't appear to be entirely precise and that not
#    all relevant parameters are defined there

import numpy as np
import os
import pytest

#from examples.amor.amor import run_create_geometry
from examples.utils.detector_geometry_from_json import BaseDetectorGeometry

strawlen = 1.0        # [m]
strawdiam = 0.00775   # [m] - drawing says 8mm

#pp_dist = strawlen/(SR - 1)   # pixel - pixel distance along a straw [m]
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


# Now in a separate class
class ICDGeometry():
    SPT = 7    # straws per tube
    NL = 4     # number of layers
    SR = 512   # straw resolution (pixels along straw)

    # number of tubes per layer in a bank for the final instrument
    #tubes_per_layer = [56, 16, 12, 16, 12, 28, 32, 20, 32] # full instrument
    tubes_per_layer = [32, 0, 0, 0, 0, 0, 0, 0, 0] # current implementation?

    def pixel(self, bank, tube, lstraw, pos):
        return self.straw_to_pixel(self.straw(bank, tube, lstraw), pos)


    # helper function, may not need to be called directly
    def straw_to_pixel(self, straw, pos):
        return straw * self.SR + pos + 1


    # return global tube parameters for a given bank
    def tube_parms(self, bank):
        offset = 0
        for i in range(bank):
            offset = offset + self.tubes_per_layer[i] * self.NL
        return [offset, self.tubes_per_layer[bank]]


    # return global straw from bank, tube and local straw
    # helper function, may not need to be called directly
    def straw(self, bank, tube, lstraw):
        if (bank > 8) or (lstraw > 6):
            raise Exception("Invalid bank or lstraw")

        if tube > self.tubes_per_layer[bank] * 4 - 1:
            raise Exception("Invalid tube")

        tube_offset, tubes_per_layer = self.tube_parms(bank)
        return (tube_offset + tube) * self.SPT + lstraw


class LokiGeometry(BaseDetectorGeometry):
    def __init__(self, path):
        self.icd = ICDGeometry()
        BaseDetectorGeometry.__init__(self,path)


@pytest.fixture(scope='session')
def geom(): # formerly known as loki_geometry
    json_file_name = 'config_larmor.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    #run_create_geometry(json_file_path)
    return LokiGeometry(json_file_path)


## @todo use final geometry and change test accordingly
def test_get_tube(geom):
    assert geom.icd.tube_parms(0)[0] == 0
    assert geom.icd.tube_parms(1)[0] == 128
    assert geom.icd.tube_parms(2)[0] == 128
    assert geom.icd.tube_parms(3)[0] == 128
    assert geom.icd.tube_parms(4)[0] == 128
    assert geom.icd.tube_parms(5)[0] == 128
    assert geom.icd.tube_parms(6)[0] == 128
    assert geom.icd.tube_parms(7)[0] == 128
    assert geom.icd.tube_parms(8)[0] == 128


## @todo use final geometry and change test accordingly
def test_get_straw(geom):
    #assert geom.get_straw(0, 0, 7) == 0 # raises
    #assert geom.get_straw(9, 0, 0) == 0 # raises
    #assert geom.get_straw(0, 56 * 4, 0) == 0 # raises
    assert geom.icd.straw(0, 0, 0) == 0
    assert geom.icd.straw(0, 1, 6) == 13
    assert geom.icd.straw(0, 32 * 4 - 1, 6) == 895


## @todo use final geometry and change test accordingly
def test_max_valid_pixel(geom, bank = 0):
    max_valid_pixel = geom.icd.pixel(bank, 127, 6, 511)
    assert max_valid_pixel == 458752
    try:
        coord = geom.p2c(max_valid_pixel)
    except:
        assert False
        return
    assert True


## @todo use final geometry and change test accordingly
def test_first_bad_pixel(geom, bank = 0):
    invalid_pixel = geom.icd.pixel(bank, 127, 6, 511) + 1
    try:
        coord = geom.p2c(invalid_pixel)
    except:
        assert True
        return
    assert False


## @todo extend test to all banks
def test_all_pixels_positive_z(geom, bank = 0):
    max_valid_pixel = geom.icd.pixel(bank, 127, 6, 511)
    for pixel in range(max_valid_pixel):
        coord = geom.p2c(pixel + 1)
        # assert coord[0] >= 0.0
        # assert coord[1] >= 0.0
        assert coord[2] >= -0.008


# Testing that corner pixels in same layer are at right angles
# so far bank 0 only
## @todo extend test to all banks
@pytest.mark.parametrize('layer', [i for i in range(4)])
def test_some_icd_values(geom, layer, bank = 0):
        pixels_per_layer = geom.icd.tubes_per_layer[bank] * geom.icd.SPT * geom.icd.SR

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
## @todo extend test to all banks
def test_pixel_pixel_dist(geom, bank = 0):
    for straw in range(geom.icd.tubes_per_layer[bank] * geom.icd.NL * geom.icd.SPT):
        offset = straw * geom.icd.SR
        pp_dist = strawlen / (geom.icd.SR - 1)
        for i in range(511):
            pix1 = offset + i + 1
            pix2 = offset + i + 2
            d = geom.dist(pix1, pix2)
            assert geom.expect(d, pp_dist, pp_precision)


# All tubes: distance between first and second straw
# Currently assumes only bank 0 and pos 0, but can be extended
## @todo extend test to all banks
@pytest.mark.parametrize('tube', [i for i in range(128)])
def test_straw_straw_dist(geom, tube, bank = 0):
    pix1 = geom.icd.pixel(bank, tube, 0, 0)
    pix2 = geom.icd.pixel(bank, tube, 1, 0)
    d = geom.dist(pix1, pix2)
    assert geom.expect(d, ss_dist, precision)


# All tubes: test angle between adjacent straws with center straw as origin
# Currently assumes only bank 0 and pos 0, but can be extended
## @todo extend test to all banks
@pytest.mark.parametrize('tube', [i for i in range(128)])
def test_straw_straw_angle(geom, tube, bank = 0):
    pix1 = geom.icd.pixel(bank, tube, 3, 0) # centre straw
    pix2 = geom.icd.pixel(bank, tube, 0, 0) # lstraw 0
    pix3 = geom.icd.pixel(bank, tube, 1, 0) # lstraw 1
    angle = geom.pix2angle(pix1, pix2, pix1, pix3)
    assert geom.expect(angle, sa_angle, sa_precision)


# Distance between adjacent tubes in same layer (Euclidian and y-projection)
# Currently assumes only bank 0 and pos 0, but can be extended
def test_tube_tube_dist(geom, bank = 0):
    tpl = geom.icd.tubes_per_layer[bank]
    for tube in range(tpl - 1):
        for layer in range(4):
            tubeone = layer * tpl + tube
            pix1 = geom.icd.pixel(bank, tubeone, 0, 0)
            pix2 = geom.icd.pixel(bank, tubeone + 1, 0, 0)

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
## @todo extend test to all banks
def test_angle_of_tubes(geom, bank = 0):
    tube_offset, tpl = geom.icd.tube_parms(bank)
    for tube in range(tpl):
        for layer in range(4 - 1):
            tube1 = tube_offset + layer       * tpl + tube
            tube2 = tube_offset + (layer + 1) * tpl + tube
            pix1 = geom.icd.pixel(bank, tube1, 0, 0)
            pix2 = geom.icd.pixel(bank, tube2, 0, 0)

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
