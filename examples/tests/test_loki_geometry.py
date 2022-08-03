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

from examples.loki.LOKI_geometry import run_create_geometry
from examples.loki.detector_banks_geo import STRAW_RESOLUTION, \
    NUM_STRAWS_PER_TUBE, TUBE_DEPTH, det_banks_data, NUM_BANKS
from examples.utils.detector_geometry_from_json import BaseDetectorGeometry

strawlen = [i for i in [det_banks_data[x]["tube_length"]
                        for x in range(NUM_BANKS)]] # [m]
strawdiam = 0.00775   # [m] - drawing says 8mm

#pp_dist = strawlen/(STRAW_RESOLUTION - 1)   # pixel - pixel distance along a straw [m]
ss_dist = strawdiam           # straw - straw distance
tt_il_dist       = 0.0284     # [m] tube tube distance (same layer)
tt_z_dist        = 0.02626    # [m] tube-tube distance (between layers)
pa_angle         = np.deg2rad(90 - 76.55)  # pack angle [rad]
sa_angle         = np.deg2rad(60)          # angle between adjacent outer straws [rad]

precision      = 0.00002  # general precision aim (2/100 mm) [m]

# deviations from general precision (minimum value for passing test)
pp_precision   = 0.000008  # [m]
pa_precision   = 0.000012  # [deg]
sa_precision   = 0.002      # straw rotation angle [rad]
tt_z_precision = 0.0000011 # [m]


# Now in a separate class
class ICDGeometry:

    # number of tubes per layer in a bank for the final instrument
    def __init__(self, banks):
        self._banks = banks
        self.tubes_per_layer = []
        number_of_tubes = []
        for idx in range(NUM_BANKS):
            self.tubes_per_layer.append(int(banks[idx]['num_tubes'] / TUBE_DEPTH))
            number_of_tubes.append(banks[idx]['num_tubes'])
        self.cumulative_tubes = list(np.cumsum(number_of_tubes))

    def pixel(self, bank, tube, lstraw, pos):
        return self.straw_to_pixel(self.straw(bank, tube, lstraw), pos)


    # helper function, may not need to be called directly
    def straw_to_pixel(self, straw, pos):
        return straw * STRAW_RESOLUTION + pos + 1


    # return global tube parameters for a given bank
    def tube_parms(self, bank):
        offset = 0
        for i in range(bank):
            offset = offset + self.tubes_per_layer[i] * TUBE_DEPTH
        return [offset, self.tubes_per_layer[bank]]

    # return global straw from bank, tube and local straw
    # helper function, may not need to be called directly
    def straw(self, bank, tube, lstraw):
        if (bank > NUM_BANKS - 1) or (lstraw > NUM_STRAWS_PER_TUBE - 1):
            raise Exception("Invalid bank or lstraw")
        if tube > self.cumulative_tubes[bank] - 1:
            raise Exception(f"Invalid tube {tube}, "
                            f"max is {self.cumulative_tubes[bank] - 1}")
        tube_offset, tubes_per_layer = self.tube_parms(bank)
        return (tube_offset + tube) * NUM_STRAWS_PER_TUBE + lstraw


class LokiGeometry(BaseDetectorGeometry):
    def __init__(self, path):
        self.icd = ICDGeometry(det_banks_data)
        BaseDetectorGeometry.__init__(self,path)


# TODO: create geometry
@pytest.fixture(scope='session')
def geom():  # formerly known as loki_geometry
    json_file_name = 'config_loki.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    run_create_geometry()
    return LokiGeometry(json_file_path)


# # TODO: use final geometry and change test accordingly
# def test_get_tube(geom):
#     assert geom.icd.tube_parms(0)[0] == 0
#     assert geom.icd.tube_parms(1)[0] == 128
#     assert geom.icd.tube_parms(2)[0] == 128
#     assert geom.icd.tube_parms(3)[0] == 128
#     assert geom.icd.tube_parms(4)[0] == 128
#     assert geom.icd.tube_parms(5)[0] == 128
#     assert geom.icd.tube_parms(6)[0] == 128
#     assert geom.icd.tube_parms(7)[0] == 128
#     assert geom.icd.tube_parms(8)[0] == 128


# # TODO: use final geometry and change test accordingly
# def test_get_straw(geom):
#     #assert geom.get_straw(0, 0, 7) == 0 # raises
#     #assert geom.get_straw(9, 0, 0) == 0 # raises
#     #assert geom.get_straw(0, 56 * 4, 0) == 0 # raises
#     assert geom.icd.straw(0, 0, 0) == 0
#     assert geom.icd.straw(0, 1, 6) == 13
#     assert geom.icd.straw(0, 32 * 4 - 1, 6) == 895


## @todo use final geometry and change test accordingly
# def test_max_valid_pixel(geom, bank = 0):
#     max_valid_pixel = geom.icd.pixel(bank, 127, 6, 511)
#     assert max_valid_pixel == 458752
#     try:
#         coord = geom.p2c(max_valid_pixel)
#     except:
#         assert False
#         return
#     assert True


# TODO: use final geometry and change test accordingly
# def test_first_bad_pixel(geom, bank = 0):
#     invalid_pixel = geom.icd.pixel(bank, 127, 6, 511) + 1
#     try:
#         coord = geom.p2c(invalid_pixel)
#     except:
#         assert True
#         return
#     assert False


# # TODO: extend test to all banks
# def test_all_pixels_positive_z(geom, bank = 0):
#     max_valid_pixel = geom.icd.pixel(bank, 127, 6, 511)
#     for pixel in range(max_valid_pixel):
#         coord = geom.p2c(pixel + 1)
#         # assert coord[0] >= 0.0
#         # assert coord[1] >= 0.0
#         assert coord[2] >= -0.008


# Testing that corner pixels in same layer are at right angles
# so far bank 0 only
# TODO: extend test to all banks
@pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
@pytest.mark.parametrize('layer', [i for i in range(TUBE_DEPTH)])
def test_some_icd_values(geom, layer, bank):
        pixels_per_layer = geom.icd.tubes_per_layer[bank] * STRAW_RESOLUTION * NUM_STRAWS_PER_TUBE

        pixel_offset = pixels_per_layer * layer

        top_left = 1 + pixel_offset
        top_right = STRAW_RESOLUTION + pixel_offset
        bottom_left = pixels_per_layer - STRAW_RESOLUTION + 1 + pixel_offset
        bottom_right = pixels_per_layer + pixel_offset
        angle_rad = np.deg2rad(90.0)

        a = geom.pix2angle(top_left, top_right, top_left, bottom_left)
        assert geom.expect(a, angle_rad, precision)

        a = geom.pix2angle(top_right, bottom_right, top_right, top_left)
        assert geom.expect(a, angle_rad, precision)

        a = geom.pix2angle(bottom_right, bottom_left, bottom_right, top_right)
        assert geom.expect(a, angle_rad, precision)

        a = geom.pix2angle(bottom_left, top_left, bottom_left, bottom_right)
        assert geom.expect(a, angle_rad, precision)


# All straws: distance between neighbouring pixels
# TODO: extend test to all banks
# TODO: check/fix precision
@pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
def test_pixel_pixel_dist(geom, bank):
    pp_dist = strawlen[bank] / (STRAW_RESOLUTION - 1)
    if bank == 0:
        first_straw = 0
    else:
        first_straw = geom.icd.cumulative_tubes[bank - 1] * NUM_STRAWS_PER_TUBE + 1
    last_straw = geom.icd.cumulative_tubes[bank] * NUM_STRAWS_PER_TUBE
    for straw in range(first_straw, last_straw):
        offset = straw * STRAW_RESOLUTION
        for i in range(STRAW_RESOLUTION - 1):
            pix1 = offset + i + 1
            pix2 = offset + i + 2
            d = geom.dist(pix1, pix2)
            assert geom.expect(d, pp_dist, pp_precision)


# A tube in each bank: distance between first and second straw
# Currently assumes only bank 0 and pos 0, but can be extended
# TODO: extend test to all banks
@pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
def test_straw_straw_dist(geom, bank):
    pix1 = geom.icd.pixel(bank, 0, 0, 0)
    pix2 = geom.icd.pixel(bank, 0, 1, 0)
    d = geom.dist(pix1, pix2)
    assert geom.expect(d, ss_dist, precision)


# A tube in each bank: test angle between adjacent straws with center straw as origin
# Currently assumes only bank 0 and pos 0, but can be extended
# TODO: extend test to all banks
# TODO: check/fix precision
@pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
# @pytest.mark.parametrize('tube', [i for i in [det_banks_data[x]["num_tubes"] - 1 for
#                                               x in range(NUM_BANKS)]])
def test_straw_straw_angle(geom, bank):
    pix1 = geom.icd.pixel(bank, 0, 3, 0) # centre straw
    pix2 = geom.icd.pixel(bank, 0, 0, 0) # lstraw 0
    pix3 = geom.icd.pixel(bank, 0, 1, 0) # lstraw 1
    angle = geom.pix2angle(pix1, pix2, pix1, pix3)
    assert geom.expect(angle, sa_angle, sa_precision)


# Distance between adjacent tubes in same layer (Euclidian and y-projection)
# Currently assumes only bank 0 and pos 0, but can be extended
# TODO: extend test to all banks
# TODO: check/fix precision
@pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
def test_tube_tube_dist(geom, bank):
    tpl = geom.icd.tubes_per_layer[bank]
    for tube in range(tpl - 1):
        for layer in range(TUBE_DEPTH):
            tubeone = layer * tpl + tube
            pix1 = geom.icd.pixel(bank, tubeone, 0, 0)
            pix2 = geom.icd.pixel(bank, tubeone + 1, 0, 0)

            # Calculate Euclidian distance
            d = geom.dist(pix1, pix2)
            assert geom.expect(d, tt_il_dist, precision)

            # # Same test not using Euclidian distance but y coordinates
            if bank == 0:
                c1 = geom.p2c(pix1)
                c2 = geom.p2c(pix2)
                dy = np.abs(c2 - c1)[1]
                assert geom.expect(dy, tt_il_dist, precision)


# Detector packs are tilted by about 13 degrees
# in addition we test the distance of the z-projection between tubes
# in adjacent layers
# TODO: extend test to all banks
# TODO: check/fix precision
# @pytest.mark.parametrize('bank', [i for i in range(NUM_BANKS)])
def test_angle_of_tubes(geom, bank = 0):
    tube_offset, tpl = geom.icd.tube_parms(bank)
    for tube in range(tpl):
        for layer in range(TUBE_DEPTH - 1):
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
