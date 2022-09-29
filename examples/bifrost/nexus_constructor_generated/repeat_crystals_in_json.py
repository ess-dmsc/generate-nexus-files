import json
import numpy as np
import re

from os import path

from examples.bifrost.detector_geometry import CHILDREN, NAME, save_to_json
from examples.bifrost.triplet_specifications import INSTRUMENT_NAME, ROWS, \
    COLUMNS, MIN_ANGLE_ROTATION, MAX_ANGLE_ROTATION, RADIAL_OFFSETS, \
    NOMINAL_RADIAL_DISTANCE


def repeat_crystals(crystals):
    repeated_crystals = []
    counter = 1
    n_crystals = 5
    angles = np.linspace(MIN_ANGLE_ROTATION,MAX_ANGLE_ROTATION, COLUMNS)
    for col, angle in enumerate(angles):
        distances = np.linspace(RADIAL_OFFSETS[col],
                                NOMINAL_RADIAL_DISTANCE + RADIAL_OFFSETS[col],
                                ROWS)
        for row, distance in enumerate(distances):
            rotation = angle
            for crystal_number in range(1, n_crystals + 1):
                crystal_key = f'focusing_analyzer_{crystal_number}'
                crystal_new_name = f'focusing_analyzer_{counter}'
                crystals[crystal_key][NAME] = crystal_new_name
                crystal_str = json.dumps(crystals[crystal_key])
                crystal_str = crystal_str.replace(f'/entry/instrument/{crystal_key}',
                                                  f'/entry/instrument/{crystal_new_name}')
                if counter - 1:
                    crystal_dependecy_name = f'focusing_analyzer_{crystal_number - 1}'
                    crystal_new_dependecy_name = f'focusing_analyzer_{counter - 1}'
                    depends_on = f'/entry/instrument/{crystal_dependecy_name}/transformations'
                    depends_on_new = f'/entry/instrument/{crystal_new_dependecy_name}/transformations'
                    crystal_str = crystal_str.replace(depends_on, depends_on_new)
                if crystal_number == 1:
                    crystal_str = crystal_str.replace('"values": -30.0',
                                                      f'"values": {rotation}')
                    crystal_str = crystal_str.replace('"values": 0.8',
                                                      f'"values": {distance}')
                tmp_dict = eval(crystal_str)
                repeated_crystals.append(tmp_dict)
                counter += 1
    return repeated_crystals


def repeat_crystal_structure(file_name, target_file):
    path_to_file_dir = path.dirname(__file__)
    file_path = path.join(path_to_file_dir, file_name)
    with open(file_path) as json_file:
        entry_dict = json.load(json_file)
    entry = entry_dict[CHILDREN][0]
    crystals_in_json = {}
    crystal_in_json_keep = []
    for child in entry[CHILDREN]:
        if NAME in child and child[NAME] == INSTRUMENT_NAME:
            for grandchild in child[CHILDREN]:
                if "focusing_analyzer_" in grandchild[NAME]:
                    crystals_in_json[grandchild[NAME]] = grandchild
                else:
                    crystal_in_json_keep.append(grandchild)
            child[CHILDREN].clear()
            child[CHILDREN] += crystal_in_json_keep
            repeated_crystals = repeat_crystals(crystals_in_json)
            child[CHILDREN] += repeated_crystals
    target_file_path = path.join(path_to_file_dir, target_file)
    save_to_json(target_file_path, entry_dict)


if __name__ == "__main__":
    repeat_crystal_structure("BIFROST_baseline_template.json",
                             "BIFROST_baseline_with_crystals.json")