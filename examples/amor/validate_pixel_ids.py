import json, math, sys
import numpy as np

# This file attempts to test the AMOR geometry by comparing a
# known physical description of the detector with the logical
# geometry specified in the ICD
# ICD: https://project.esss.dk/owncloud/index.php/s/Ja7ARvxtRUzEtiK
# PPT: Multi-BladeGeom.ppt (from Francesco P.)

NW = 32 # number of wires
NS = 32 # number of strips

CHILDREN = "children"
CONFIG = "config"
VALUES = "values"

radius  = 4.0000    # sample to front wire [m]
iw_dist = 0.0040    # inter-wire distance [m]
is_dist = 0.0040    # inter-strip distance [m]
ww_dist = 0.01011   # front wire-wire dist (between cassettes) [m]
wl_dist = 0.1240    # wire array length [m]
blang   = 1.448     # angle between blades [degrees]

precision = 0.000002      # general precision [m]
radius_precision = 0.0005 # !! Seems too large
ang_precision = 0.108     # !! Seems too large

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

    def cxy2pix(self, cass, y, x):
        return cass * NW * NS + y * NS + x + 1

    def expect(self, val, expect, precision):
        if abs(val - expect) > precision:
            print("value error: {} - {} ({}) > {}".format(val, expect, abs(val - expect), precision))
            if self.fatal:
                sys.exit(0)

if __name__ == '__main__':

    ag = AmorGeometry()

    print("Testing distance between x-coordinates for all wires")
    for y in range(352):
        for x in range(31):
            pix1 = ag.cxy2pix(0, y, x    )
            pix2 = ag.cxy2pix(0, y, x + 1)
            d = ag.dist(pix1, pix2)
            ag.mprint("Wire {}, px1, px2 ({}, {}), dist {}".format(y, pix1, pix2, d))
            ag.expect(d, is_dist, precision)


    print("Testing distance between y-coordinates for all wires within cassettes")
    for cass in range(11):
        for y in range(31):
            pixy1 = ag.cxy2pix(cass, y, 0)
            pixy2 = ag.cxy2pix(cass, y + 1, 0)
            d = ag.dist(pixy1, pixy2)
            ag.mprint("Casse {}, y {}, py1, py2 ({}, {}), dist {}".format(cass, y, pixy1, pixy2, d))
            ag.expect(d, iw_dist, precision)


    print("Testing radius of front wire")
    for cass in range(11):
        pix = ag.cxy2pix(cass, 31, 0)
        c1 = ag.p2c(pix)
        z = np.array([0.0, 0.0, 0.0])
        r = np.linalg.norm(c1 - z)
        ag.mprint("Cassette {}, x {}, y{}, radius {}".format(cass, c1[0], c1[1], r))
        ag.expect(r, radius, radius_precision)



    print("Testing intra-cassette distance (front wires only)")
    for cass in range(10):
            pixy1 = ag.cxy2pix(cass    , 31, 0)
            pixy2 = ag.cxy2pix(cass + 1, 31, 0)
            d = ag.dist(pixy1, pixy2)
            ag.mprint("Cassette {}, py1, py2 ({}, {}), dist {}".format(cass, pixy1, pixy2, d))
            ag.expect(d, ww_dist, precision)


    print("Testing cassette order")
    for cass in range(10):
            pixy1 = ag.cxy2pix(cass    , 31, 0)
            pixy2 = ag.cxy2pix(cass + 1, 31, 0)
            c1 = ag.p2c(pixy1)
            c2 = ag.p2c(pixy2)
            ag.mprint("Cassette {}, py1, py2 ({}, {})".format(cass, pixy1, pixy2))
            if c1[2] >= c2[2]:  # z values
                print("error:")
                sys.exit(0)
            if c1[1] <= c2[1]:  # y values
                print("error:")
                sys.exit(0)


    print("Testing relative positions (within same cassette)")
    for cass in range(11):
        for y in range(31):
            pixy1 = ag.cxy2pix(cass, y    , 0)
            pixy2 = ag.cxy2pix(cass, y + 1, 0)
            c1 = ag.p2c(pixy1)
            c2 = ag.p2c(pixy2)
            ag.mprint("Cass {}, y {}, py1, py2 ({}, {}), c1, c1, ({}, {})".format(cass, y, pixy1, pixy2, c1[1], c2[1]))
            # For two wires w1, w2 where y1 > y2
            # z and y coordinates for w1 must be larger than for w2
            if c1[2] <= c2[2]:  # z
                print("error")
                sys.exit(0)
            if c1[1] <= c2[1]: # y
                print("error")
                sys.exit(0)


    print("Testing distance from first wire to last wire")
    for cass in range(10):
        c11 = ag.cxy2pix(cass,  0, 0)
        c12 = ag.cxy2pix(cass, 31, 0)
        d = ag.dist(c11, c12)
        ag.mprint("Cassette {}, wire array length {}".format(cass, d))
        ag.expect(d, wl_dist, precision)


    print("Testing angle between blades")
    ag.debug = False
    for cass in range(10):
        c11 = ag.cxy2pix(9,     31, 0)
        c12 = ag.cxy2pix(9,      0, 0)
        c21 = ag.cxy2pix(10, 31, 0)
        c22 = ag.cxy2pix(10,  0, 0)
        ag.mprint("Cass {}".format(cass))
        ag.mprint("c1 {}, {}".format(c11, c12))
        ag.mprint("p1, p2 {}, {}".format(ag.p2c(c11), ag.p2c(c12)))
        ag.mprint("c2 {}, {}".format(c21, c22))
        ag.mprint("p1, p2 {}, {}".format(ag.p2c(c21), ag.p2c(c22)))
        v1 = ag.p2v(c11, c12)
        v2 = ag.p2v(c21, c22)
        ag.mprint("v1 {}".format(v1))
        ag.mprint("v2 {}".format(v2))
        ang = ag.angle(v1, v2)
        ag.expect(ang, blang, ang_precision)
