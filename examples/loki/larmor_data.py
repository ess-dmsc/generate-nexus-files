from os import path

from numpy import sqrt, deg2rad, array, rad2deg

LENGTH_UNIT = 'm'
SCALE_FACTOR = .001

FRACTIONAL_PRECISION = 5
IMAGING_TUBE_D = 25.4 * SCALE_FACTOR
TUBE_DEPTH = 4
NUM_BANKS = 1

# Straw information.
NUM_STRAWS_PER_TUBE = 7
STRAW_DIAMETER = 7.5 * SCALE_FACTOR
STRAW_RESOLUTION = 512  # Might need to switch to 512 though for commisiong tests.
STRAW_Y_LOC = 1.14 * SCALE_FACTOR
STRAW_Z_LOC = 7.67 * SCALE_FACTOR
TUBE_OUTER_STRAW_DIST_FROM_CP = sqrt(STRAW_Y_LOC ** 2 + STRAW_Z_LOC ** 2)
STRAW_ALIGNMENT_OFFSET_ANGLE = deg2rad(-5)
# STRAW_ALIGNMENT_OFFSET_ANGLE = deg2rad(5) + arcsin(STRAW_Y_LOC / TUBE_OUTER_STRAW_DIST_FROM_CP)
print(rad2deg(STRAW_ALIGNMENT_OFFSET_ANGLE))

user_1 = {
    "name": "John Doe",
    "email": "john.doe@ess.eu",
    "facility_user_id": "jdoe1",
    "affiliation": "ESS",
}

data_users = [user_1]

z_sample_rel_source = 25610

z_offset = (0, 0, -z_sample_rel_source)


def add_abs_offset(position):
    return tuple(array(position) + array(z_offset))


data_sample = {
    'location': add_abs_offset((0, 0, z_sample_rel_source)),
    'name': 'larmor_sample'
}

data_sample = {}

data_source = {
    'location': add_abs_offset((0, 0, 0)),
    'name': 'moderator'
}

data_disk_choppers = []

data_monitors = [{'location': add_abs_offset((0, 0, 9819.5)), 'name': 'monitor_1',
                  'topic': 'monitor_channel_0', 'source': 'ttlmon'},
                 {'location': add_abs_offset((0, 0, 25760)), 'name': 'monitor_2',
                  'topic': 'monitor_channel_1', 'source': 'ttlmon'}, ]

print(data_monitors[0]['location'][2], data_monitors[1]['location'][2])

s1 = 0.03
s2 = 0.02
s3 = 0.008
s_eq1 = s3 + 0.75 * (s2 - s3) / 4.82

data_slits = []

# Data from larmor document.
tube_l = 1000
d_x = tube_l/STRAW_RESOLUTION #pixel size (1000/512=1.953125)
dz_sample = 4431
y_1 = 6.28    #y dist of two tubes in subsequent layers
y_2 = 28.4    #y dist of two tubes in a layer
z_1 = 26.259  #z dist of the tubes in subsequent layers


# pixel_num is pixel 1 to 512, n is tube 1 to 128
def l_n(n):
    return (n - 1) % 4 + 1


def r_n(n):
    return int((n - 1) / 4) + 1 - 16


# def x_n(pixel_num):
#     return tube_l/2 - d_x / 2 - d_x * (pixel_num - 1)


def y_n(n):
    return -(-r_n(n) * y_2 + (l_n(n) - 1) * y_1)


def z_n(n):
    return (l_n(n) - 1) * z_1 + dz_sample


tube_dim_1 = 4
tube_dim_2 = 32
tube_dims = tube_dim_1 * tube_dim_2

tube_radius = IMAGING_TUBE_D*0.5 / SCALE_FACTOR
det_pixel_id_start = 1  # starting pixel ID for the 'first' detector bank.
det_banks_data = {0: {'A': [(500, y_n(tube_dims), z_n(tube_dims - tube_dim_1 + 1)),
                            (500, y_n(tube_dims - tube_dim_1 + 1), z_n(tube_dims)),
                            (500, y_n(tube_dim_1), z_n(1)),
                            (500, y_n(1), z_n(tube_dim_1))],
                      'B': [(-500, y_n(tube_dims), z_n(tube_dims - tube_dim_1 + 1)),
                            (-500, y_n(tube_dims - tube_dim_1 + 1), z_n(tube_dims)),
                            (-500, y_n(tube_dim_1), z_n(1)),
                            (-500, y_n(1), z_n(tube_dim_1))],
                      'num_tubes': tube_dim_1 * tube_dim_2,
                      'bank_offset': (0, 23.89, 4099 - z_n(tube_dims - tube_dim_1 + 1)+tube_radius),
                      'name': 'larmor_detector',
                      'topic': 'loki_detector',
                      'source': 'loki',
                      'tube_length': 1.0
                      },
                  }

file_name = 'larmor.nxs'
detector_data_filepath = 'larmor_data.nxs'
isis_larmor_data_filepath = 'LARMOR00058412.nxs'
json_filename = path.join(path.dirname(__file__), "config_larmor.json")
axis_1_size = 1000
axis_2_size = det_banks_data[0]['num_tubes'] * NUM_STRAWS_PER_TUBE * \
              STRAW_RESOLUTION + det_pixel_id_start - 1

print(det_banks_data)
