from numpy import sqrt, deg2rad

LENGTH_UNIT = 'm'
SCALE_FACTOR = .001

FRACTIONAL_PRECISION = 5
IMAGING_TUBE_D = 25.4 * SCALE_FACTOR
TUBE_DEPTH = 4

# Straw information.
NUM_STRAWS_PER_TUBE = 7
STRAW_DIAMETER = 8.00 * SCALE_FACTOR
STRAW_RESOLUTION = 512
STRAW_Y_LOC = 1.14 * SCALE_FACTOR
STRAW_Z_LOC = 7.67 * SCALE_FACTOR
STRAW_ALIGNMENT_OFFSET_ANGLE = deg2rad(5)
TUBE_OUTER_STRAW_DIST_FROM_CP = sqrt(STRAW_Y_LOC ** 2 + STRAW_Z_LOC ** 2)

user_1 = {
    "name": "John Doe",
    "email": "john.doe@ess.eu",
    "facility_user_id": "jdoe1",
    "affiliation": "ESS",
}

data_users = [user_1]

data_sample = {
    'location': (0, 0, 0),
    'name': 'LOKI_sample'
}

data_source = {
    'location': (0, 0, -23600),
    'name': 'moderator'
}

data_disk_choppers = [{'name': 'chopper_1', 'location': (0, 0, -17100),
                       'rotation_speed': 14.0, 'slits': 2,
                       'disk_rad': 700.0 / 2},
                      {'name': 'chopper_2', 'location': (0, 0, -8570),
                       'rotation_speed': 14.0, 'slits': 2,
                       'disk_rad': 700.0 / 2}]

data_monitors = [{'location': (0, 0, -16800), 'name': 'monitor_0'}]

data_slits = [{'location': (0, 0, -8000), 'name': 'slit_1'
                  , 'x_gap': 30, 'y_gap': 25},
              {'location': (0, 0, -5000), 'name': 'slit_2'
                  , 'x_gap': 30, 'y_gap': 25},
              {'location': (0, 0, -3000), 'name': 'slit_3',
               'x_gap': 30, 'y_gap': 25}]

# Data from lamor document.
d_x = 1.953
dz_sample = 4431
y_1 = 6.28
y_2 = 28.4
z_1 = 26.259
tube_l = 1000


# pixel_num is pixel 1 to 512, n is tube 1 to 128
def L_n(n):
    return (n - 1) % 4 + 1


def R_n(n):
    return int((n - 1) / 4) + 1 - 16


def X_n(pixel_num):
    return -tube_l/2 + d_x / 2 + d_x * (pixel_num - 1)


def Y_n(n):
    return -(-R_n(n) * y_2 + (L_n(n) - 1) * y_1)


def Z_n(n):
    return (L_n(n) - 1) * z_1 + dz_sample


det_banks_data = {0: {'A': [(X_n(1), Y_n(1), Z_n(1)),
                            (X_n(1), Y_n(4), Z_n(4)),
                            (X_n(1), Y_n(125), Z_n(125)),
                            (X_n(1), Y_n(128), Z_n(128))],
                      'B': [(X_n(512), Y_n(1), Z_n(1)),
                            (X_n(512), Y_n(4), Z_n(4)),
                            (X_n(512), Y_n(125), Z_n(125)),
                            (X_n(512), Y_n(128), Z_n(128))],
                      'num_tubes': 128,
                      'bank_offset': (0, 0, 0)
                      },
                  }

print(det_banks_data)
