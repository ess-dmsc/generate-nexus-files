# rotation angle in degrees.
# triplet upper left coordinate in meters and should be cartesian where
# origin is in sample position.
import numpy

TUBE_LENGTH = 0.2  # [m]
TUBE_RADIUS = 0.05  # [m]
DIST_BETWEEN_TUBES = 0.02  # [m]
PIXEL_RESOLUTION_PER_TUBE = 100
NUMBER_OF_TUBES_PER_BANK = 3
PIXEL_LENGTH = TUBE_LENGTH / PIXEL_RESOLUTION_PER_TUBE
INSTRUMENT_NAME = 'instrument'

# BIFROST detector configuration.
COLUMNS = 9
ROWS = 5
MIN_RADIAL_DISTANCE = 0.5
RADIAL_OFFSETS = [MIN_RADIAL_DISTANCE + 0.1,
                  MIN_RADIAL_DISTANCE + 0.2,
                  MIN_RADIAL_DISTANCE + 0.3] * 3
NOMINAL_RADIAL_DISTANCE = 2.
MIN_ANGLE_ROTATION = -30
MAX_ANGLE_ROTATION = 30
X = numpy.linspace(0.5, 1, ROWS)
CURVATURE = [1 / x - 2 for x in X]
