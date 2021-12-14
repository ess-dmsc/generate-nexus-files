from numpy import sqrt, deg2rad, array

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

z_offset = (0, 0, 25300)


def add_offset(position):
    return tuple(array(position) - array(z_offset))


data_sample = {
    'location': add_offset((0, 0, 25300)),
    'name': 'larmor_sample'
}

data_source = {
    'location': add_offset((0, 0, 0)),
    'name': 'moderator'
}

data_disk_choppers = [{'name': 'chopper_1', 'location': (0, 0, 9689),
                       'rotation_speed': 14.0, 'slits': 1,
                       'disk_rad': 270.0},
                      {'name': 'chopper_2', 'location': (0, 0, 9739),
                       'rotation_speed': 14.0, 'slits': 1,
                       'disk_rad': 270.0}]

data_monitors = [{'location': add_offset((0, 0, 9819.5)), 'name': 'monitor_1'},
                 {'location': add_offset((0, 0, 20313)), 'name': 'monitor_2'},
                 {'location': add_offset((0, 0, 24056)), 'name': 'monitor_3'},
                 {'location': add_offset((0, 0, 25760)), 'name': 'monitor_4'},
                 {'location': add_offset((0, 0, 29650)), 'name': 'monitor_5'}]

s1 = 0.03
s2 = 0.02
s3 = 0.008
s_eq1 = s3 + 0.75 * (s2 - s3) / 4.82

data_slits = [{'location': add_offset((0, 0, 17050)), 'name': 'coarsejaws'
                  , 'x_gap': s1 * 1000, 'y_gap': s1 * 1000},
              {'location': add_offset((0, 0, 20430)), 'name': 'slit_1'
                  , 'x_gap': s2 * 1000, 'y_gap': s2 * 1000},
              {'location': add_offset((0, 0, 24500)), 'name': 'slit_2',
               'x_gap': s_eq1 * 1000, 'y_gap': s_eq1 * 1000},
              {'location': add_offset((0, 0, 25250)), 'name': 'slit_2a',
               'x_gap': 6, 'y_gap': 8},
              {'location': add_offset((0, 0, 25300)), 'name': 'SANS_sample',
               'x_gap': 50, 'y_gap': 50}
              ]

# Data from larmor document.
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
