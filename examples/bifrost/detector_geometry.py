import json
from typing import List

import numpy as np

from examples.bifrost.triplet_specifications import TRIPLETS_SPECS, \
    TUBE_RADIUS, PIXEL_LENGTH, NUMBER_OF_TUBES_PER_BANK, DIST_BETWEEN_TUBES, \
    PIXEL_RESOLUTION_PER_TUBE, INSTRUMENT_NAME


CHILDREN = 'children'
NAME = 'name'


def generate_detector_pixel_shape():
    return [0, 1, 2], [[0, 0, 0], [0, TUBE_RADIUS, 0], [PIXEL_LENGTH, 0, 0]]


def local_bank_offsets():
    offsets: List[np.array] = []
    for i in range(NUMBER_OF_TUBES_PER_BANK):
        dist_y_direction = i * (TUBE_RADIUS * 2 + DIST_BETWEEN_TUBES)
        for j in range(PIXEL_RESOLUTION_PER_TUBE):
            dist_x_direction = PIXEL_LENGTH * j
            offsets.append(np.array([dist_x_direction, dist_y_direction, 0]))
    return offsets


def rotation(theta):
    return np.array(
        [[np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0],
         [-np.sin(theta), 0, np.cos(theta)]])


def add_global_rotation_and_offset(local_offsets, bank_specs):
    rotation_matrix = rotation(np.deg2rad(bank_specs['rotation']))
    position = np.array(bank_specs['position'])
    xyz_offsets = []
    for offset in local_offsets:
        global_position = np.dot(rotation_matrix, offset) + position
        xyz_offsets.append(global_position.tolist())
    return list(zip(*xyz_offsets))


def get_nexus_detector_dict(detector_number, cylinders, vertices, x_off, y_off,
                            z_off):
    return {
        "name": "bifrost_detector",
        "type": "group",
        "children": [
            {
                "module": "dataset",
                "config": {
                    "name": "detector_number",
                    "values": detector_number,
                    "type": "int32"
                }
            },
            {
                "name": "pixel_shape",
                "type": "group",
                "children": [
                    {
                        "module": "dataset",
                        "config": {
                            "name": "cylinders",
                            "values": [
                                cylinders
                            ],
                            "type": "int32"
                        }
                    },
                    {
                        "module": "dataset",
                        "config": {
                            "name": "vertices",
                            "values": vertices,
                            "type": "float"
                        },
                        "attributes": [
                            {
                                "name": "units",
                                "values": "m"
                            }
                        ]
                    }
                ],
                "attributes": [
                    {
                        "name": "NX_class",
                        "values": "NXcylindrical_geometry"
                    }
                ]
            },
            {
                "module": "dataset",
                "config": {
                    "name": "x_pixel_offset",
                    "values": x_off,
                    "type": "float"
                },
                "attributes": [
                    {
                        "name": "units",
                        "values": "m"
                    }
                ]
            },
            {
                "module": "dataset",
                "config": {
                    "name": "y_pixel_offset",
                    "values": y_off,
                    "type": "float"
                },
                "attributes": [
                    {
                        "name": "units",
                        "values": "m"
                    }
                ]
            },
            {
                "module": "dataset",
                "config": {
                    "name": "z_pixel_offset",
                    "values": z_off,
                    "type": "float"
                },
                "attributes": [
                    {
                        "name": "units",
                        "values": "m"
                    }
                ]
            }
        ],
        "attributes": [
            {
                "name": "NX_class",
                "values": "NXdetector"
            }
        ]
    }


def create_entry_and_instrument(nexus_det_dict):
    return {
        "children": [
            {
                "name": "entry",
                "type": "group",
                "children": [
                    {
                        "name": "instrument",
                        "type": "group",
                        "children": [nexus_det_dict],
                        "attributes": [
                            {
                                "name": "NX_class",
                                "values": "NXinstrument"
                            }
                        ]
                    },
                ],
                "attributes": [
                    {
                        "name": "NX_class",
                        "values": "NXentry"
                    }
                ]
            }
        ]
    }


def add_detector_to_baseline_json(file_name, nexus_dict, target_file):
    from os import path
    path_to_file_dir = path.dirname(__file__)
    file_path = path.join(path_to_file_dir, file_name)
    with open(file_path) as json_file:
        entry_dict = json.load(json_file)
    entry = entry_dict[CHILDREN][0]
    for child in entry[CHILDREN]:
        if child[NAME] == INSTRUMENT_NAME:
            child[CHILDREN].append(nexus_dict)
            break
    target_file_path = path.join(path_to_file_dir, target_file)
    save_to_json(target_file_path, entry)


def save_to_json(file_name, dict_to_save, compress=False):
    with open(file_name, 'w', encoding='utf-8') as file:
        if compress:
            json.dump(dict_to_save, file, separators=(',', ':'))
        else:
            json.dump(dict_to_save, file, indent=4)


if __name__ == "__main__":
    create_new_json = False
    cylinder_ids, cylinder_vertices = generate_detector_pixel_shape()
    loc_offsets = local_bank_offsets()
    x_offset_total, y_offset_total, z_offset_total = [], [], []
    for key in TRIPLETS_SPECS:
        x_offset, y_offset, z_offset = \
            add_global_rotation_and_offset(loc_offsets, TRIPLETS_SPECS[key])
        x_offset_total += x_offset
        y_offset_total += y_offset
        z_offset_total += z_offset
    pixel_ids = list(range(len(x_offset_total)))
    nexus_detector_dict = get_nexus_detector_dict(pixel_ids, cylinder_ids,
                                                  cylinder_vertices,
                                                  x_offset_total,
                                                  y_offset_total,
                                                  z_offset_total)
    if create_new_json:
        nexus_entry_dict = create_entry_and_instrument(nexus_detector_dict)
        save_to_json("bifrost_detector_baseline.json", nexus_entry_dict)
    else:
        add_detector_to_baseline_json('bifrost_baseline.json',
                                      nexus_detector_dict,
                                      'bifrost_baseline_with_detector.json')
