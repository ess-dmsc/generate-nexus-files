from numpy import sqrt, deg2rad, arcsin, rad2deg

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
TUBE_OUTER_STRAW_DIST_FROM_CP = sqrt(STRAW_Y_LOC ** 2 + STRAW_Z_LOC ** 2)
STRAW_ALIGNMENT_OFFSET_ANGLE = deg2rad(-5)
# STRAW_ALIGNMENT_OFFSET_ANGLE = arcsin(STRAW_Y_LOC / TUBE_OUTER_STRAW_DIST_FROM_CP)
print(rad2deg(STRAW_ALIGNMENT_OFFSET_ANGLE))

user_1 = {
    "name": "John Doe",
    "email": "john.doe@ess.eu",
    "facility_user_id": "jdoe1",
    "affiliation": "ESS",
}

user_2 = {
    "name": "Jane Doe",
    "email": "jane.doe@ess.eu",
    "facility_user_id": "jdoe2",
    "affiliation": "ESS",
}

data_users = [user_1, user_2]

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

data_monitors = [{'location': (0, 0, -16800), 'name': 'monitor_0',
                  'topic': 'monitor_channel_1', 'source': 'ttlmon'},
                 {'location': (0, 0, -8400), 'name': 'monitor_1',
                  'topic': 'monitor_channel_2', 'source': 'ttlmon'},
                 {'location': (0, 0, -2040), 'name': 'monitor_2',
                  'topic': 'monitor_channel_3', 'source': 'ttlmon'},
                 {'location': (0, 0, 200), 'name': 'monitor_3',
                  'topic': 'monitor_channel_4', 'source': 'ttlmon'},
                 {'location': (0, 0, 5091.28 + 10), 'name': 'monitor_4',
                  'topic': 'monitor_channel_5', 'source': 'ttlmon'}]

data_slits = [{'location': (0, 0, -8000), 'name': 'slit_1'
                  , 'x_gap': 30, 'y_gap': 25},
              {'location': (0, 0, -5000), 'name': 'slit_2'
                  , 'x_gap': 30, 'y_gap': 25},
              {'location': (0, 0, -3000), 'name': 'slit_3',
               'x_gap': 30, 'y_gap': 25}]

det_pixel_id_start = 1  # starting pixel ID for the 'first' detector bank.
det_banks_data = {0: {'A': [(-500, -781, 5012.5),
                            (-500, -799.84, 5091.28),
                            (-500, 781, 5012.5),
                            (-500, 762.16, 5091.28)],
                      'B': [(500, -781, 5012.5),
                            (500, -799.84, 5091.28),
                            (500, 781, 5012.5),
                            (500, 762.16, 5091.28)],
                      'num_tubes': 224,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_0',
                      'topic': 'loki_detector',
                      'source': 'loki_0'
                      },
                  1: {'A': [(-500, -710.24, 2899.18),
                            (-500, -699.24, 2979.43),
                            (-500, -286.34, 2941.42),
                            (-500, -275.35, 3021.67)],
                      'B': [(500, -710.24, 2899.18),
                            (500, -699.24, 2979.43),
                            (500, -286.34, 2941.42),
                            (500, -275.35, 3021.67)],
                      'num_tubes': 64,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_1',
                      'topic': 'loki_detector',
                      'source': 'loki_1'
                      },
                  2: {'A': [(-535.94, -250, 3328.75),
                            (-523.26, -250, 3408.75),
                            (-224.49, -250, 3353.18),
                            (-211.82, -250, 3433.19)],
                      'B': [(-535.94, 250, 3328.75),
                            (-523.26, 250, 3408.75),
                            (-224.49, 250, 3353.18),
                            (-211.82, 250, 3433.19)],
                      'num_tubes': 48,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_2',
                      'topic': 'loki_detector',
                      'source': 'loki_2'
                      },
                  3: {'A': [(-500, 286.33, 2941.34),
                            (-500, 275.34, 3021.59),
                            (-500, 710.23, 2899.11),
                            (-500, 699.24, 2979.36)],
                      'B': [(500, 286.33, 2941.34),
                            (500, 275.34, 3021.59),
                            (500, 710.23, 2899.11),
                            (500, 699.24, 2979.36)],
                      'num_tubes': 64,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_3',
                      'topic': 'loki_detector',
                      'source': 'loki_3'
                      },
                  4: {'A': [(224.49, -250, 3353.11),
                            (211.82, -250, 3433.11),
                            (535.93, -250, 3328.67),
                            (523.26, -250, 3408.67)],
                      'B': [(224.49, 250, 3353.11),
                            (211.82, 250, 3433.11),
                            (535.93, 250, 3328.67),
                            (523.26, 250, 3408.67)],
                      'num_tubes': 48,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_4',
                      'topic': 'loki_detector',
                      'source': 'loki_4'
                      },
                  5: {'A': [(-700, -1096.67, 1051.39),
                            (-700, -1102.32, 1132.19),
                            (-700, -365.34, 1281.9),
                            (-700, -370.99, 1362.7)],
                      'B': [(500, -1096.67, 1051.39),
                            (500, -1102.32, 1132.19),
                            (500, -365.34, 1281.9),
                            (500, -370.99, 1362.7)],
                      'num_tubes': 112,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_5',
                      'topic': 'loki_detector',
                      'source': 'loki_5'
                      },
                  6: {'A': [(-1191.15, -585, 1509.59),
                            (-1187.05, -585, 1590.49),
                            (-325.76, -585, 1671.47),
                            (-321.66, -585, 1752.37)],
                      'B': [(-1191.15, 615, 1509.59),
                            (-1187.05, 615, 1590.49),
                            (-325.76, 615, 1671.47),
                            (-321.66, 615, 1752.37)],
                      'num_tubes': 128,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_6',
                      'topic': 'loki_detector',
                      'source': 'loki_6'
                      },
                  7: {'A': [(-500, 365.36, 1281.97),
                            (-500, 371.01, 1362.77),
                            (-500, 880, 1119.78),
                            (-500, 885.65, 1200.58)],
                      'B': [(700, 365.36, 1281.97),
                            (700, 371.01, 1362.77),
                            (700, 880, 1119.78),
                            (700, 885.65, 1200.58)],
                      'num_tubes': 80,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_7',
                      'topic': 'loki_detector',
                      'source': 'loki_7'
                      },
                  8: {'A': [(325.62, -650, 1670.71),
                            (321.52, -650, 1751.61),
                            (1191.14, -650, 1509.52),
                            (1187.04, -650, 1590.41)],
                      'B': [(325.62, 550, 1670.71),
                            (321.52, 550, 1751.61),
                            (1191.14, 550, 1509.52),
                            (1187.04, 550, 1590.41)],
                      'num_tubes': 128,
                      'bank_offset': (0, 0, 0),
                      'name': 'loki_detector_8',
                      'topic': 'loki_detector',
                      'source': 'loki_8'
                      },
                  }

file_name = 'loki.nxs'
detector_data_filepath = 'loki_data.nxs'
json_filename = "config_loki.json"
axis_1_size = 300
axis_2_size = sum([det_banks_data[x]['num_tubes'] for x in det_banks_data])\
              * NUM_STRAWS_PER_TUBE * STRAW_RESOLUTION + det_pixel_id_start - 1