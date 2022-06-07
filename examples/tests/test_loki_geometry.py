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

strawlen = 1.0        # [m]
strawdiam = 0.00775   # [m] - drawing says 8mm
strawresolution = 512 # pixels along straw

pp_dist = strawlen/(strawresolution - 1)  # pixel - pixel distance along a straw [m]
ss_dist = strawdiam  # straw - straw distance
tt_il_dist       = 0.0284   # [m] tube tube distance (same layer)

precision        = 0.000001 # general precision aim (1/1000 mm) [m]
pp_precision     = 0.000008 # [m]
ss_precision     = 0.000001   # [m]


class LokiGeometry(BaseDetectorGeometry):
    def cxy2pix(self, cass, y, x):
        return cass * NW * NS + y * NS + x + 1

@pytest.fixture(scope='session')
def loki_geometry():
    json_file_name = 'config_larmor.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'loki')
    json_file_path = os.path.join(script_dir, json_file_name)
    #run_create_geometry(json_file_path)
    return LokiGeometry(json_file_path)


# First straw: distance between neighbouring pixels
def test_pixel_pixel_dist(loki_geometry):
    for i in range(511):
        pix1 = i + 1
        pix2 = i + 2
        d = loki_geometry.dist(pix1, pix2)
        assert loki_geometry.expect(d, pp_dist, pp_precision)

# First tube: distance between straws
def test_straw_straw_dist(loki_geometry):
    pix1 = 1
    pix2 = 1 + 512
    c1 = loki_geometry.p2c(pix1)
    c2 = loki_geometry.p2c(pix2)
    print("{}, {}".format(c1, c2))
    d = loki_geometry.dist(pix1, pix2)
    assert loki_geometry.expect(d, ss_dist, ss_precision)


# Front tubes: Distance from first to second tube
def test_tube_tube_dist(loki_geometry):
    pix1 = 1
    pix2 = 1 + 512 * 7
    c1 = loki_geometry.p2c(pix1)
    c2 = loki_geometry.p2c(pix2)
    print("{}, {}".format(c1, c2))
    d = loki_geometry.dist(pix1, pix2)
    assert loki_geometry.expect(d, tt_il_dist, precision)

def test_specific_from_drawing(loki_geometry):
    c1 = np.array([0, -1.14, -7.67])/1000
    c2 = np.array([0, -7.21, -2.85])/1000
    d = np.linalg.norm(c2-c1)
    print("dist {}".format(d))
    #assert d == ss_dist
    assert loki_geometry.expect(d, ss_dist, precision)
