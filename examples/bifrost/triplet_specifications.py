# rotation angle in degrees.
# triplet upper left coordinate in meters and should be cartesian where
# origin is in sample position.

TRIPLETS_SPECS = {
    1: {
        'rotation': 5,
        'position': [1.1, 2.5, 3.]
    },
    2: {
        'rotation': 11,
        'position': [-1.1, 2.5, 3.]
    },
}

TUBE_LENGTH = 1.0  # [m]
TUBE_RADIUS = 0.05  # [m]
DIST_BETWEEN_TUBES = 0.02  # [m]
PIXEL_RESOLUTION_PER_TUBE = 100
NUMBER_OF_TUBES_PER_BANK = 3
PIXEL_LENGTH = TUBE_LENGTH / PIXEL_RESOLUTION_PER_TUBE
INSTRUMENT_NAME = 'instrument'
