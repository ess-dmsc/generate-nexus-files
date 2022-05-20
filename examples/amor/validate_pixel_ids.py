import json, math, sys
import numpy as np

# This file attempts to test the AMOR geometry by comparing a
# known physical description of the detector with the logical
# geometry specified in the ICD
# ICD: https://project.esss.dk/owncloud/index.php/s/Ja7ARvxtRUzEtiK
# PPT: Multi-BladeGeom.ppt (from Francesco P.)

CHILDREN = "children"
CONFIG = "config"
VALUES = "values"

iw_dist = 0.0040  # inter-wire distance [m]
is_dist = 0.0040  # inter-strip distance [m]
ww_dist = 0.01011 # wire-wire dist (between cassettes) [m]
wl_dist = 0.1400  # wire array length [m]
precision = 0.0004  # precision [m]


class AmorGeometry:

    def __init__(self, debug = False, fatal = True):
        self.debug = debug # enable debug print
        self.fatal = fatal # terminate on first error
        data = {}

        with open("AMOR_nexus_structure.json", 'r') as json_file:
            data = json.load(json_file)

        multiblade_detector = data[CHILDREN][0][CHILDREN][1][CHILDREN][0][CHILDREN]
        pixel_ids = multiblade_detector[1][CONFIG][VALUES]
        x, y, z = multiblade_detector[5][CONFIG][VALUES], \
                  multiblade_detector[6][CONFIG][VALUES], \
                  multiblade_detector[7][CONFIG][VALUES]

        self.id_dict = dict(zip(pixel_ids, zip(x, y, z)))


    # pixel to coord
    def p2c(self, pixel):
        return np.asarray(self.id_dict[pixel])

    # return vector from two pixels
    def p2v(self, pix1, pix2):
        c1 = self.p2c(pix1)
        c2 = self.p2c(pix2)
        return c2 - c1

    # distance between two pixels
    def dist(self, pix1, pix2):
        c1 = self.p2c(pix1)
        c2 = self.p2c(pix2)
        return np.linalg.norm(c2 - c1)

    def angle(self, v1, v2):
        return np.arccos(np.dot(v1, v2))

    def mprint(self, str):
        if (self.debug):
            print(str)

    def expect_dist(self, val, expect, precision):
        if abs(val - expect) > precision:
            print("distance error: {} - {} ({}) > {}".format(val, expect, abs(val - expect), precision))
            if self.fatal:
                sys.exit(0)

if __name__ == '__main__':

    ag = AmorGeometry()

    print("Testing distance between x-coordinates for all wires")
    for wire in range(352):
        for x in range(31):
            pix1 = wire * 32 + 1 + x
            pix2 = wire * 32 + 1 + x + 1
            d = ag.dist(pix1, pix2)
            ag.mprint("Wire {}, px1, px2 ({}, {}), dist {}".format(wire, pix1, pix2, d))
            ag.expect_dist(d, is_dist, precision)


    print("Testing distance between y-coordinates for all wires within cassettes")
    for cass in range(11):
        y0 = cass * 32
        for wire in range(31):
            pixy1 = (y0 + wire     ) * 32 + 1
            pixy2 = (y0 + wire +  1) * 32 + 1
            d = ag.dist(pixy1, pixy2)
            ag.mprint("Wire {}, py1, py2 ({}, {}), dist {}".format(wire, pixy1, pixy2, d))
            ag.expect_dist(d, iw_dist, precision)


    print("Testing intra-cassette distance")
    for cass in range(10):
        po0 = (cass    ) * 32 * 32 + 1 # first pixel in first cassette
        po1 = (cass + 1) * 32 * 32 + 1 # first pixel in second cassette
        for wire in range(32):
            pixy1 = po0 + wire * 32
            pixy2 = po1 + wire * 32
            d = ag.dist(pixy1, pixy2)
            ag.mprint("Wire {}, py1, py2 ({}, {}), dist {}".format(wire, pixy1, pixy2, d))
            ag.expect_dist(d, ww_dist, precision)


    print("Testing relative positions (within same cassette)")
    for cass in range(11):
        po0 = cass * 32 * 32 + 1 # first pixel in cassette
        for wire in range(31):
            pixy1 = po0 + (wire    ) * 32
            pixy2 = po0 + (wire + 1) * 32
            c1 = ag.p2c(pixy1)
            c2 = ag.p2c(pixy2)
            ag.mprint("Wire {}, py1, py2 ({}, {}), c1, c1, ({}, {})".format(wire, pixy1, pixy2, c1[0], c2[0]))
            # For two wires w1, w2 where w2 > w1:
            # z and y coordinates for w1 must be larger than for w2
            if c1[2] <= c2[2]:  # z
                print("error")
                sys.exit(0)
            if c1[1] <= c2[1]: # y
                print("error")
                sys.exit(0)

    print("Testing distance from first wire to last wire")
    for cass in range(11):
        ag.debug = True
        ag.fatal = False
        c11 = cass * 32 * 32 + 1
        c12 = c11 + 992
        d = ag.dist(c11, c12)
        ag.mprint("Cassette {}, wire array length {}".format(cass, d))
        ag.expect_dist(d, wl_dist, precision)
        # c21 = ag.p2c(1025)
        # c22 = ag.p2c(2017)
        # print(c11, c12)
        # print(c21, c22)
        # v1 = ag.p2v(1, 993)
        # v2 = ag.p2v(1025, 2017)
        # print(v1)
        # print(v2)
        # print(ag.angle(v1, v2))
