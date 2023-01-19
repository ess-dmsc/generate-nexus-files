import json
from os import path

import pytest

from examples.utils.detector_geometry_from_json import retrieve_data_from_json, \
    DETECTOR_NUMBER, VERTICES, BaseDetectorGeometry


class DreamDetectorGeometry(BaseDetectorGeometry):
    def __init__(self, file_name, debug=False, fatal=True):
        self.debug = debug  # enable debug print
        self.fatal = fatal  # terminate on first error
        data: dict
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        x, y, z, pixel_ids, vertices, list_pixel_ids = [], [], [], [], [], []
        retrieve_data_from_json(data, DETECTOR_NUMBER, list_pixel_ids)
        retrieve_data_from_json(data, VERTICES, vertices)
        for vertex in vertices:
            x.append(vertex[0])
            y.append(vertex[1])
            z.append(vertex[2])
        for pixel_ids_detector in list_pixel_ids:
            pixel_ids += pixel_ids_detector
        self.id_dict = dict(zip(pixel_ids, zip(x, y, z)))


@pytest.fixture(scope='session')
def dream_geometry():
    # TODO: Figure out how to retrieve the DREAM_structure.json file
    # TODO: as it currently is almost 2 GB and not feasible to commit to GitHub.
    json_file_name = 'DREAM_structure.json'
    file_dir = path.dirname(path.abspath(__file__))
    script_dir = path.join(file_dir, '..', 'dream')
    json_file_path = path.join(script_dir, json_file_name)
    return DreamDetectorGeometry(json_file_path)
