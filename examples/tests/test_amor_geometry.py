import os

import numpy as np
import pytest

from examples.amor.amor import run_create_geometry
from examples.utils.detector_geometry_from_json import BaseDetectorGeometry

NC = 11 # number of cassettes
NW = 32 # number of wires
NS = 32 # number of strips

radius  = 4.0000    # sample to front wire [m]
iw_dist = 0.0040    # inter-wire distance [m]
is_dist = 0.0040    # inter-strip distance [m]
ww_dist = 0.01011   # front wire-wire dist (between cassettes) [m]
wl_dist = 0.1240    # wire array length [m]
blang   = 0.1448    # angle between blades [degrees]

precision        = 0.0000000001 # general precision [m]
ww_precision     = 0.000002 # wire wire dist prec [m]
radius_precision = 0.00000000000001 # precision for radius [m]
ang_precision    = 0.00000000001 # [degrees]


class AmorGeometry(BaseDetectorGeometry):
    def cxy2pix(self, cass, y, x):
        return cass * NW * NS + y * NS + x + 1


@pytest.fixture(scope='session')
def amor_geometry():
    json_file_name = 'AMOR_nexus_structure.json'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(file_dir, '..', 'amor')
    json_file_path = os.path.join(script_dir, json_file_name)
    run_create_geometry(json_file_path)
    return AmorGeometry(json_file_path)


def test_max_valid_pixel(amor_geometry):
    max_valid_pixel = NC * NS * NW
    try:
        coord = amor_geometry.p2c(max_valid_pixel)
    except:
        assert False


def test_first_bad_pixel(amor_geometry):
    invalid_pixel = NC * NS * NW + 1
    try:
        coord = amor_geometry.p2c(invalid_pixel)
    except:
        assert True


def test_all_pixels_positive_coords(amor_geometry):
    for pixel in range(NC * NS * NW):
        coord = amor_geometry.p2c(pixel + 1)
        assert coord[0] >= -0.065 # because of rotation some pixels are slightly negative
        assert coord[1] >= 0.0
        assert coord[2] >= 0.0


def test_all_pixels_dist_from_origin(amor_geometry):
    for pixel in range(NC * NS * NW):
        coord = amor_geometry.p2c(pixel + 1)
        r = np.linalg.norm(coord)
        assert r >= radius
        assert r <= radius + wl_dist + 0.001


def test_distance_between_x_coordinates_for_all_wires(amor_geometry):
    for y in range(352):
        for x in range(31):
            pix1 = amor_geometry.cxy2pix(0, y, x)
            pix2 = amor_geometry.cxy2pix(0, y, x + 1)
            d = amor_geometry.dist(pix1, pix2)
            amor_geometry.mprint(
                "Wire {}, px1, px2 ({}, {}), dist {}".format(y, pix1, pix2, d))
            assert amor_geometry.expect(d, is_dist, precision)


def test_distance_between_y_coordinates_for_all_wires_witin_cassettes(amor_geometry):
    for cass in range(11):
        for y in range(31):
            pixy1 = amor_geometry.cxy2pix(cass, y, 0)
            pixy2 = amor_geometry.cxy2pix(cass, y + 1, 0)
            d = amor_geometry.dist(pixy1, pixy2)
            amor_geometry.mprint(
                "Casse {}, y {}, py1, py2 ({}, {}), dist {}".format(cass, y, pixy1,
                                                                    pixy2, d))
            assert amor_geometry.expect(d, iw_dist, precision)


def test_radius_of_front_wire(amor_geometry):
    for cass in range(11):
        pix = amor_geometry.cxy2pix(cass, 31, 0)
        c1 = amor_geometry.p2c(pix)
        c1[0] = 0
        z = np.array([0.0, 0.0, 0.0])
        r = np.linalg.norm(c1 - z)
        amor_geometry.mprint("Cassette {}, x {}, y{}, radius {}".format(cass, c1[0], c1[1], r))
        assert amor_geometry.expect(r, radius, radius_precision)


def test_intra_cassete_distance_for_front_wires(amor_geometry):
    for cass in range(10):
        pixy1 = amor_geometry.cxy2pix(cass, 31, 0)
        pixy2 = amor_geometry.cxy2pix(cass + 1, 31, 0)
        d = amor_geometry.dist(pixy1, pixy2)
        amor_geometry.mprint(
            "Cassette {}, py1, py2 ({}, {}), dist {}".format(cass, pixy1, pixy2, d))
        assert amor_geometry.expect(d, ww_dist, ww_precision)


def test_cassette_order(amor_geometry):
    for cass in range(10):
        pixy1 = amor_geometry.cxy2pix(cass, 31, 0)
        pixy2 = amor_geometry.cxy2pix(cass + 1, 31, 0)
        c1 = amor_geometry.p2c(pixy1)
        c2 = amor_geometry.p2c(pixy2)
        amor_geometry.mprint("Cassette {}, py1, py2 ({}, {})".format(cass, pixy1, pixy2))
        assert not (c1[2] >= c2[2])
        assert not (c1[1] <= c2[1])


def test_relative_positions_within_same_cassette(amor_geometry):
    for cass in range(11):
        for y in range(31):
            pixy1 = amor_geometry.cxy2pix(cass, y, 0)
            pixy2 = amor_geometry.cxy2pix(cass, y + 1, 0)
            c1 = amor_geometry.p2c(pixy1)
            c2 = amor_geometry.p2c(pixy2)
            amor_geometry \
              .mprint("Cass {}, y {}, py1, py2 ({}, {}), c1, c1, ({}, {})" \
              .format(cass, y,pixy1, pixy2, c1[1], c2[1]))
        # For two wires w1, w2 where y1 > y2
        # z and y coordinates for w1 must be larger than for w2
        assert not (c1[2] <= c2[2])  # z
        assert not (c1[1] <= c2[1])  # y


def test_distance_from_first_wire_to_last_wire(amor_geometry):
    for cass in range(10):
        c1 = amor_geometry.cxy2pix(cass, 0, 0)
        c2 = amor_geometry.cxy2pix(cass, 31, 0)
        d = amor_geometry.dist(c1, c2)
        amor_geometry.mprint("Cassette {}, wire array length {}".format(cass, d))
        assert amor_geometry.expect(d, wl_dist, precision)


def test_blade_angle(amor_geometry):
    for cass in range(10):
        c1 = amor_geometry.cxy2pix(cass, 31, 0)
        c2 = amor_geometry.cxy2pix(cass, 0, 0)
        v1 = amor_geometry.p2v(c1, c2)
        vz = [0, 0.0, radius]
        ang = amor_geometry.r2d(amor_geometry.angle(v1, vz))
        exp_ang = 5.0 + blang * (10 - cass)
        amor_geometry.mprint("cass {} angle {} degrees, exp {}".format(cass, ang, exp_ang))
        assert amor_geometry.expect(ang, exp_ang, ang_precision)


def test_angle_between_blades(amor_geometry):
    for cass in range(10):
        c11 = amor_geometry.cxy2pix(cass, 31, 0)
        c12 = amor_geometry.cxy2pix(cass, 0, 0)
        c21 = amor_geometry.cxy2pix(cass + 1, 31, 0)
        c22 = amor_geometry.cxy2pix(cass + 1, 0, 0)
        v1 = amor_geometry.p2v(c11, c12)
        v2 = amor_geometry.p2v(c21, c22)
        ang = amor_geometry.r2d(amor_geometry.angle(v1, v2))
        amor_geometry.mprint("cass {}-{} angle {} degrees".format(cass, cass + 1, ang))
        assert amor_geometry.expect(ang, blang, ang_precision)
