import csv
import json
from abc import ABC

from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from enum import Enum
from typing import Dict, List, Optional
from examples.loki.nurf_data import load_one_spectro_file, nurf_file_creator
IMPORT_LARMOR = False  # Change depending on what data set should be used.
DEBUG_LARMOR_DET = False  # Use larmor or loki full system.
if IMPORT_LARMOR:
    from examples.loki.larmor_data import FRACTIONAL_PRECISION, \
        NUM_STRAWS_PER_TUBE, IMAGING_TUBE_D, STRAW_DIAMETER, TUBE_DEPTH, \
        STRAW_ALIGNMENT_OFFSET_ANGLE, TUBE_OUTER_STRAW_DIST_FROM_CP, \
        STRAW_RESOLUTION, SCALE_FACTOR, LENGTH_UNIT, det_banks_data, \
        data_disk_choppers, data_monitors, data_slits, \
        data_source, data_sample, data_users, file_name, det_pixel_id_start, \
        axis_1_size, axis_2_size, detector_data_filepath, \
        isis_larmor_data_filepath, json_filename
else:
    from examples.loki.detector_banks_geo import FRACTIONAL_PRECISION, \
        NUM_STRAWS_PER_TUBE, IMAGING_TUBE_D, STRAW_DIAMETER, TUBE_DEPTH, \
        STRAW_ALIGNMENT_OFFSET_ANGLE, TUBE_OUTER_STRAW_DIST_FROM_CP, \
        STRAW_RESOLUTION, SCALE_FACTOR, LENGTH_UNIT, det_banks_data, \
        data_disk_choppers, data_monitors, data_slits, \
        data_source, data_sample, data_users, file_name, det_pixel_id_start, \
        axis_1_size, axis_2_size, detector_data_filepath, json_filename

if DEBUG_LARMOR_DET and not IMPORT_LARMOR:
    from larmor_data import det_banks_data as larmor_det_data
    det_banks_data[9] = larmor_det_data[0]

    bank_9_offs = np.array(det_banks_data[9]['A'][0]) - \
                  np.array(det_banks_data[0]['A'][0])

    for key in ['A', 'B']:
        tmp = []
        for item in det_banks_data[9][key]:
            tmp.append(tuple(np.array(item) - bank_9_offs))
        det_banks_data[9][key] = tmp


VALID_DATA_TYPES_NXS = (str, int, datetime, float)
VALID_ARRAY_TYPES_NXS = (list, np.ndarray)
N_VERTICES = 3
ATTR = 'attributes'
CHILDREN = 'children'
CONFIG = 'config'
DATASET = 'dataset'
DATATYPE = "dtype"
DEPENDS_ON = 'depends_on'
ENTRY = 'entry'
INSTRUMENT = 'instrument'
LOCATION = 'location'
MODULE = 'module'
NAME = 'name'
NX_CLASS = 'NX_class'
NX_DATA = 'NXdata'
NXLOG_VALUE = 'value'
NXLOG_NAME = 'nx_log'
ROTATION = 'rotation'
SAMPLE = 'sample'
SOURCE = 'source'
STREAM_MODULE = 'stream_module'
TIME = 'time'
TOF = 'tof'
TOPIC = 'topic'
SPECTRUM = 'spectrum'
TRANSFORMATION_TYPE = 'transformation_type'
TRANSFORMATIONS = 'transformations'
TRANSLATION = 'translation'
TYPE = 'type'
UNITS = 'units'
VALUES = 'values'
VECTOR = 'vector'


def reorder_straw_offsets_in_list(straw_offs_unsorted: List):
    """
    Sorting of straw offsets based on information from LoKI detector bank
    description.
    """
    mid_point = int((NUM_STRAWS_PER_TUBE - 1) / 2 + 1)
    straw_offs_sorted = [0] * len(straw_offs_unsorted)
    straw_offs_sorted[mid_point] = straw_offs_unsorted[0]
    straw_offs_sorted[:mid_point] = straw_offs_unsorted[mid_point:]
    straw_offs_sorted[mid_point:] = reversed(straw_offs_unsorted[1:mid_point])
    return straw_offs_sorted

def reorder_straw_offsets_to_flip_upside_down(straw_offs_unsorted: List):
    straw_offs_sorted = [0] * len(straw_offs_unsorted)
    straw_offs_sorted[0] = straw_offs_unsorted[0]
    straw_offs_sorted[1] = straw_offs_unsorted[4]
    straw_offs_sorted[2] = straw_offs_unsorted[5]
    straw_offs_sorted[3] = straw_offs_unsorted[3]
    straw_offs_sorted[4] = straw_offs_unsorted[1]
    straw_offs_sorted[5] = straw_offs_unsorted[2]
    straw_offs_sorted[6] = straw_offs_unsorted[6]
    return straw_offs_sorted

def write_csv_file(csv_data):
    column_names = ['bank id', 'tube id', 'straw id', 'local straw position',
                    'pixel id', 'x', 'y', 'z']
    with open('detector_geometry.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(column_names)
        for row in csv_data:
            csv_writer.writerow(row)


class IdIterator:
    def __init__(self, start=0):
        self._start = start

    def __iter__(self):
        self._id: int = self._start
        return self

    def __next__(self):
        current_id = self._id
        self._id += 1
        return current_id

    @staticmethod
    def reset(start=0):
        return IdIterator(start)

global pixel_id_iter, straw_id_iter, transform_id_iter
pixel_id_iter = iter(IdIterator(det_pixel_id_start))
straw_id_iter = iter(IdIterator())
transform_id_iter = iter(IdIterator(1))


def reset_all_iterators():
    global pixel_id_iter, straw_id_iter, transform_id_iter
    pixel_id_iter = iter(IdIterator(det_pixel_id_start))
    straw_id_iter = iter(IdIterator())
    transform_id_iter = iter(IdIterator(1))


# Static class.
class NexusInfo:

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_entry_class_attr():
        return {NX_CLASS: 'NXentry'}

    @staticmethod
    def get_sample_class_attr():
        return {NX_CLASS: 'NXsample'}

    @staticmethod
    def get_instrument_class_attr():
        return {NX_CLASS: 'NXinstrument'}

    @staticmethod
    def get_detector_class_attr():
        return {NX_CLASS: 'NXdetector'}

    @staticmethod
    def get_source_class_attr():
        return {NX_CLASS: 'NXsource'}

    @staticmethod
    def get_cylindrical_geo_class_attr():
        return {NX_CLASS: 'NXcylindrical_geometry'}

    @staticmethod
    def _get_transformation_class_attr():
        return {NX_CLASS: 'NXtransformations'}

    @staticmethod
    def get_disk_chopper_class_attr():
        return {NX_CLASS: 'NXdisk_chopper'}

    @staticmethod
    def get_nx_log_class_attr():
        return {NX_CLASS: 'NXlog'}

    @staticmethod
    def get_monitor_class_attr():
        return {NX_CLASS: 'NXmonitor'}

    @staticmethod
    def get_slit_class_attr():
        return {NX_CLASS: 'NXslit'}

    @staticmethod
    def get_event_data(nx_event_data, topic, source):
        return {
            VALUES: nx_event_data,
            ATTR: {NX_CLASS: 'NXevent_data'},
            MODULE:
                {
                    MODULE: 'ev42',
                    CONFIG: {
                            TOPIC: topic,
                            SOURCE: source
                    }
                }
        }

    @staticmethod
    def get_nx_user(new_user):
        user_values = {
            u_key: {VALUES: val, ATTR: None} for u_key, val in new_user.items()
        }
        return {
            VALUES: user_values,
            ATTR: {NX_CLASS: 'NXuser'}
        }

    @staticmethod
    def get_char_metadata(char_metadata):
        pass

    @staticmethod
    def get_transform_translation(value, vector, unit, depend_path='.'):
        return NexusInfo._get_transformation(value, vector, unit, TRANSLATION,
                                             depend_path)

    @staticmethod
    def get_transform_rotation(value, vector, unit, depend_path='.'):
        return NexusInfo._get_transformation(value, vector, unit, ROTATION,
                                             depend_path)

    @staticmethod
    def get_nxlog_transform_translation(value, vector, unit, depend_path='.'):
        return NexusInfo._get_nxlog_transformation(value, vector,
                                                   unit, TRANSLATION,
                                                   depend_path)

    @staticmethod
    def get_nxlog_transform_rotation(value, vector, unit, depend_path='.'):
        return NexusInfo._get_nxlog_transformation(value, vector,
                                                   unit, ROTATION,
                                                   depend_path)

    @staticmethod
    def _get_nxlog_transformation(value, vector, unit, transform_type,
                                  depend_path):
        location_dataset = NexusInfo._get_location_dataset(value, vector,
                                                           unit, transform_type,
                                                           depend_path)
        return {
            VALUES:
                {
                    'trans_' + str(next(transform_id_iter)):
                        NexusInfo.get_nx_log_group(nx_log_data=location_dataset)
                },
            ATTR: NexusInfo._get_transformation_class_attr()
        }

    @staticmethod
    def get_nx_log_group(nx_log_data=None, time=None, time_unit='tof',
                         x_label=None, y_label=None, topic="topic", source="source",
                         module="f142"):
        if time is None:
            if isinstance(nx_log_data[VALUES], list):
                time = list(range(0, len(nx_log_data[VALUES])))
            elif isinstance(nx_log_data[VALUES], np.ndarray):
                # TODO: Probably need to change this for some data.
                time = list(range(0, len(nx_log_data[VALUES])))
            else:
                time = [0]
        if nx_log_data is None:
            nx_log_data = {VALUES: {}, ATTR: {}}
        if not x_label:
            x_label = TIME
        if not y_label:
            y_label = NXLOG_VALUE
        attributes = nx_log_data[ATTR]
        if attributes is None:
            attributes = {}
        nx_log_data[ATTR] = {}
        return {
            VALUES:
                {
                    y_label:
                        nx_log_data,
                    x_label:
                        {
                            VALUES: time,
                            ATTR: {UNITS: time_unit}
                        },
                    MODULE:
                        {
                            MODULE: module,
                            CONFIG: {
                                TOPIC: topic,
                                SOURCE: source
                            }
                        }
                },
            ATTR: {**NexusInfo.get_nx_log_class_attr(), **attributes}
        }

    @staticmethod
    def _get_transformation(value, vector, unit, transform_type, depend_path):
        location_dataset = NexusInfo._get_location_dataset(value, vector, unit,
                                                           transform_type,
                                                           depend_path)
        return {VALUES:
                    {'trans_' + str(next(transform_id_iter)): location_dataset},
                ATTR: NexusInfo._get_transformation_class_attr()
                }

    @staticmethod
    def _get_location_dataset(value, vector, unit, transform_type, depend_path):
        return {
            VALUES: value,
            ATTR: {
                UNITS: unit,
                TRANSFORMATION_TYPE: transform_type,
                DEPENDS_ON: depend_path,
                VECTOR: [x for x in vector]
            }
        }

    @staticmethod
    def get_units_attribute(units):
        return {UNITS: units}

    @staticmethod
    def get_values_attrs_as_dict(values, attrs=None):
        return {VALUES: values,
                ATTR: attrs}

    @staticmethod
    def get_transformations_as_dict(geo_data, position, transform_path,
                                    name='', as_nx_log=False):
        norm = np.linalg.norm(position)
        if norm:
            position = position / norm
        else:
            position = np.array([0., 0., 1.0])
        if as_nx_log:
            geo_data[TRANSFORMATIONS] = \
                NexusInfo.get_nxlog_transform_translation([norm],
                                                          tuple(position),
                                                          LENGTH_UNIT)
        else:
            geo_data[TRANSFORMATIONS] = \
                NexusInfo.get_transform_translation([norm], tuple(position),
                                                    LENGTH_UNIT)
        if transform_path:
            abs_path = transform_path
            if as_nx_log:
                transform_name = \
                    list(geo_data[TRANSFORMATIONS][VALUES].keys())
                abs_path = abs_path + transform_name[0] + '/' + NXLOG_VALUE
            else:
                transform_name = list(geo_data[TRANSFORMATIONS][VALUES].keys())
                abs_path += transform_name[0]
            if len(transform_name) > 1:
                print("Warning! Only supply one "
                      "dependency for a transformation. "
                      "Only the first element in the supplied list of "
                      "dependencies will be used.")
            geo_data[DEPENDS_ON] = {VALUES: abs_path,
                                    ATTR: None}
        if name:
            geo_data[NAME] = {VALUES: [name],
                              ATTR: None}
        return geo_data


class DetectorAlignment(Enum):
    HORIZONTAL, VERTICAL = range(2)


class Vertex:
    """
    Description of a single vertex in a 3D space.
    """

    def __init__(self, x: float, y: float, z: float, vertex_id=None):
        self._x, self._y, self._z = x, y, z
        self._id = vertex_id

    def get_coordinates(self) -> np.array:
        return np.array([self._x, self._y, self._z])

    def get_vertex_id(self) -> int:
        return self._id

    def __repr__(self):
        return f'x: {self._x}, y: {self._y}, z: {self._z}, ' \
               f'vertex_id: {self._id}'


class Cylinder:
    """
    Description of a single cylinder in the 3D space fully described by
    3 vertices. In this case it will be used to describe a pixel.
    """

    def __init__(self, vertices_coordinates: List):
        if len(set(vertices_coordinates)) != N_VERTICES:
            raise RuntimeError('Three unique vertices are expected to fully'
                               'describe the cylinder geometry')
        else:
            self._vertices = []
            for vertex_coordinates in vertices_coordinates:
                self._vertices.append(Vertex(*vertex_coordinates))

    def get_vertices_ids(self) -> List[int]:
        return [self._vertices[i].get_vertex_id() for i in range(N_VERTICES)]

    def __repr__(self):
        return f'A: {self._vertices[0]} \n' \
               f'B: {self._vertices[1]} \n' \
               f'C: {self._vertices[2]}'

    def get_vertices_coordinates_as_dict(self):
        return {'Vertex A': self._vertices[0].get_coordinates(),
                'Vertex B': self._vertices[1].get_coordinates(),
                'Vertex C': self._vertices[2].get_coordinates()}

    def get_vertices_coordinates_as_list(self):
        return [self._vertices[0].get_coordinates(),
                self._vertices[1].get_coordinates(),
                self._vertices[2].get_coordinates()]


class Pixel(Cylinder):
    """
    Description of a detector pixel geometrically described as a cylinder.
    """

    def __init__(self, vertices_coordinates: List):
        super().__init__(vertices_coordinates)
        self.pixel_xyz_offsets = []
        self.nominal_vertices_coordinates: Dict = \
            self.get_vertices_coordinates_as_dict()

    def set_pixel_xyz_offsets(self, pixel_offsets: np.array):
        self.pixel_xyz_offsets = pixel_offsets

    def compound_data_in_dict(self, straw_offset: np.array) -> Dict:
        point_a = self.nominal_vertices_coordinates['Vertex A']
        data_dict = {}
        for pixel_offset in self.pixel_xyz_offsets:
            data_dict[next(pixel_id_iter)] = \
                tuple(point_a + pixel_offset + straw_offset)
        return data_dict

    def compound_data_in_list(self, bank_id: int, tube_id: int,
                              straw_id: int, straw_offset: np.array) -> List:
        point_a = np.array(self.nominal_vertices_coordinates['Vertex A'])
        point_a_offsets = point_a + straw_offset
        data_list: List = []
        loc_pixel_id_iter = iter(IdIterator())
        for pixel_offset in self.pixel_xyz_offsets:
            point_to_save = point_a + point_a_offsets + pixel_offset
            data_list.append((bank_id,
                              tube_id,
                              straw_id,
                              next(loc_pixel_id_iter),
                              next(pixel_id_iter),
                              point_to_save[0],
                              point_to_save[1],
                              point_to_save[2]))
        # ret_val = data_list[:2] + data_list[-2:]
        # return ret_val
        return data_list

    def get_pixel_data(self, straw_offset: np.array):
        data_offsets: List = []
        data_detector_num: List = []
        for pixel_offset in self.pixel_xyz_offsets:
            data_offsets.append(straw_offset + pixel_offset)
            data_detector_num.append(next(pixel_id_iter))
        return data_offsets, data_detector_num

    def get_cylinder_geo_data(self):
        return {'cylinders': NexusInfo.get_values_attrs_as_dict([(0, 1, 2)]),
                'vertices': NexusInfo.get_values_attrs_as_dict(
                    self.get_vertices_coordinates_as_list(),
                    NexusInfo.get_units_attribute(LENGTH_UNIT))}


class Straw:
    """
    Abstraction of a single straw consisting of STRAW_RESOLUTION pixels.
    """

    def __init__(self, point_a: tuple, point_b: tuple, point_c: tuple,
                 detector_bank_id: int):
        # Point A is the center point on side A of the detector bank.
        # Point B is on side A of the detector bank, radially out from Point A.
        # Point C is the center point on side B of the detector bank.
        self._point_a = point_a
        self._point_b = point_b
        self._point_c = point_c
        self._straw_xyz_offsets = [None] * NUM_STRAWS_PER_TUBE
        self._detector_bank_id = detector_bank_id
        self._pixel = None

    def set_straw_offsets(self, alignment: DetectorAlignment, base_vector,
                          upside_down: bool, plot_all: bool = False):
        straw_offs = [np.array([0, 0, 0])]
        if alignment is DetectorAlignment.HORIZONTAL:
            def rotation(theta):
                return np.array(
                    [[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])
        else:
            def rotation(theta):
                return np.array(
                    [[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
        rotation_angle = np.deg2rad(360 / (NUM_STRAWS_PER_TUBE - 1))
        for straw_idx in range(NUM_STRAWS_PER_TUBE - 1):
            tmp_angle = rotation_angle * straw_idx + \
                        (STRAW_ALIGNMENT_OFFSET_ANGLE if not upside_down else -STRAW_ALIGNMENT_OFFSET_ANGLE)
            rotated_vector = np.dot(rotation(tmp_angle), base_vector)
            inter_res = \
                (rotated_vector * TUBE_OUTER_STRAW_DIST_FROM_CP).tolist()
            for count, _ in enumerate(inter_res):
                inter_res[count] = round(inter_res[count], 5)
            inter_res = np.array(inter_res)
            straw_offs.append(inter_res)
        straw_offs = reorder_straw_offsets_in_list(straw_offs)
        if upside_down:
          straw_offs = reorder_straw_offsets_to_flip_upside_down(straw_offs)
        if plot_all:
            ax_tmp = plt.axes(projection='3d')
            for count, value in enumerate(straw_offs):
                ax_tmp.scatter3D(value[0] * 1000, value[1] * 1000,
                                 value[2] * 1000)
                ax_tmp.text(value[0] * 1000, value[1] * 1000, value[2] * 1000,
                            '%s' % (str(count)), size=20, zorder=1, color='k')
                # print(count, ':', value * 1000)
                ax_tmp.set_xlabel('X')
                ax_tmp.set_ylabel('Y')
                ax_tmp.set_zlabel('Z')
            plt.show()
        self._straw_xyz_offsets = straw_offs

    def populate_with_pixels(self, plot_all: bool = False):
        """
        Populates the tube straw with pixels according to its pixel resolution.
        """
        vector_along_straw = np.array(self._point_c) - np.array(self._point_a)
        vector_along_straw /= STRAW_RESOLUTION
        print("vector along straw", vector_along_straw)
        pixel_end_point = tuple(np.array(self._point_a) + vector_along_straw)
        vertices_first_pixel = [self._point_a, self._point_b, pixel_end_point]
        self._pixel = Pixel(vertices_first_pixel)
        offsets_pixel = [vector_along_straw * (j+0.5)
                         for j in range(STRAW_RESOLUTION)]
        if plot_all:
            ax_tmp = plt.axes(projection='3d')
            res = pixel_end_point
            x0 = res[0]
            y0 = res[1]
            z0 = res[2]
            ax_tmp.scatter3D([value[0] + x0 for value in offsets_pixel],
                             [value[1] + y0 for value in offsets_pixel],
                             [value[2] + z0 for value in offsets_pixel])
            print(tuple(res + np.array(offsets_pixel[-1])))
            print(self._point_c)
            plt.show()
        self._pixel.set_pixel_xyz_offsets(offsets_pixel)

    def compound_data_in_dict(self, tube_offset: np.array) -> Dict:
        data_dict = {}
        for straw_offset in self._straw_xyz_offsets:
            data_dict[next(straw_id_iter)] = \
                self._pixel.compound_data_in_dict(straw_offset + tube_offset)
        return data_dict

    def compound_data_in_list(self, tube_id: int,
                              tube_offset: np.array) -> List:
        data_list: List = []
        for straw_offset in self._straw_xyz_offsets:
            data_list += self._pixel.compound_data_in_list(
                self._detector_bank_id, tube_id,
                next(straw_id_iter), straw_offset + tube_offset)
        return data_list

    def get_straw_data(self, tube_offset: np.array):
        data_offsets: List = []
        data_detector_num: List = []
        for straw_offset in self._straw_xyz_offsets:
            tmp_offsets, tmp_data_detector_num = \
                self._pixel.get_pixel_data(straw_offset + tube_offset)
            data_offsets += tmp_offsets
            data_detector_num += tmp_data_detector_num
        return data_offsets, data_detector_num

    def get_straw_pixel_geometry(self):
        return self._pixel.get_cylinder_geo_data()


class Tube:
    """
    Abstraction of a tube in a detector bank that holds several straws.
    coordinates should be provided as the center coordinates of the tube.
    It is also possible for the user to provide x, y and z offset to
    repeat the same tube geometry in a detector bank.
    This saves time and space when generating large number of detector banks
    with essentially the same tube geometry, albeit shifted.
    """

    def __init__(self, point_start: tuple, point_end: tuple,
                 alignment: DetectorAlignment):
        self._alignment = alignment
        self._diameter = IMAGING_TUBE_D
        self._point_start = point_start
        self._point_end = point_end
        self._xyz_offsets: List = []
        self._straw: Optional[Straw] = None

    def set_xyz_offsets(self, xyz_offsets):
        # for c, item in enumerate(xyz_offsets):
        #     print(f'x_{c + 1}, y_{c + 1}, z_{c + 1}: ', item)
        self._xyz_offsets = xyz_offsets

    def get_xyz_offsets(self):
        return self._xyz_offsets

    def get_endpoints(self):
        return self._point_start, self._point_end

    def populate_with_uniform_straws(self, detector_bank_id: int,
                                     base_vec_1: np.array,
                                     base_vec_2: np.array):
        """
        Populates the tube with straws based on tube location in the 3D space.
        """
        r_vector = (base_vec_1 + base_vec_2)
        r_vector /= np.linalg.norm(base_vec_1 + base_vec_2)
        point_radial = np.array(self._point_start) + \
                       (STRAW_DIAMETER / 2) * np.array(r_vector)
        self._straw = Straw(self._point_start,
                            tuple(point_radial),
                            self._point_end,
                            detector_bank_id)
        base_vec_norm = base_vec_1 / np.linalg.norm(base_vec_1)
        upside_down = detector_bank_id in [1,2,5,6]
        self._straw.set_straw_offsets(self._alignment,
                                      base_vec_norm,
                                      upside_down,
                                      False)
        self._straw.populate_with_pixels()

    def compound_data_in_dict(self) -> Dict:
        data_dict = {}
        tube_id_iterator = iter(IdIterator())
        for tube_offset in self._xyz_offsets:
            data_dict[next(tube_id_iterator)] = \
                self._straw.compound_data_in_dict(tube_offset)
        return data_dict

    def compound_data_in_list(self) -> List:
        data_list: List = []
        tube_id_iterator = iter(IdIterator())
        for tube_offset in self._xyz_offsets:
            data_list += self._straw.compound_data_in_list(
                next(tube_id_iterator),
                tube_offset)
        return data_list

    def get_geometry_data(self) -> Dict:
        data_offsets: List = []
        data_detector_num: List = []
        if not self._straw:
            empty_nexus_field = NexusInfo.get_values_attrs_as_dict([])
            return {'detector_number': empty_nexus_field,
                    'pixel_shape': empty_nexus_field,
                    'x_pixel_offset': empty_nexus_field,
                    'y_pixel_offset': empty_nexus_field,
                    'z_pixel_offset': empty_nexus_field}

        for tube_offset in self._xyz_offsets:
            tmp_offsets, tmp_data_detector_num = \
                self._straw.get_straw_data(tube_offset)
            data_offsets += tmp_offsets
            data_detector_num += tmp_data_detector_num

        pixel_shape = self._straw.get_straw_pixel_geometry()
        unit_m = NexusInfo.get_units_attribute(LENGTH_UNIT)

        return {
            'detector_number':
                NexusInfo.get_values_attrs_as_dict(data_detector_num),
            'pixel_shape':
                NexusInfo.get_values_attrs_as_dict(
                    pixel_shape,
                    NexusInfo.get_cylindrical_geo_class_attr()),
            'x_pixel_offset':
                NexusInfo.get_values_attrs_as_dict(
                    [x[0] for x in data_offsets], unit_m),
            'y_pixel_offset':
                NexusInfo.get_values_attrs_as_dict(
                    [y[1] for y in data_offsets], unit_m),
            'z_pixel_offset':
                NexusInfo.get_values_attrs_as_dict(
                    [z[2] for z in data_offsets], unit_m)}


class Bank:
    """
    Abstraction of a detector bank consisting of multiple tubes.
    """

    def __init__(self, bank_geo: Dict, bank_id: int):
        self._bank_id = bank_id
        self._nbr_of_tubes = bank_geo['num_tubes']
        self._bank_offset = np.array(bank_geo['bank_offset']) * SCALE_FACTOR
        self._bank_translation = np.array(bank_geo['A'][0]) * SCALE_FACTOR
        self._bank_geometry = self._set_bank_geometry(bank_geo)
        self._tube_depth = TUBE_DEPTH
        self._tube_width = int(self._nbr_of_tubes / TUBE_DEPTH)

        # Check that provided geometry seems feasible.
        self._is_bank_cuboid()
        self._check_tube_center_distance()

        # Check that the corner points are in a plane.
        # Otherwise something is wrong with the provided geometry.
        self._grid_corners_in_plane(self._bank_geometry['A'])

        # The base vectors (in a plane) of the local coordinate system
        # are non-orthogonal and not normalized. This makes it easy to find the
        # global coordinates of the tube center points in the detector bank.
        local_origin = np.array([0, 0, 0])
        self._base_vec_1 = np.array(self._bank_geometry['A'][1]) - local_origin
        self._base_vec_1 /= self._tube_depth - 1
        self._base_vec_2 = np.array(self._bank_geometry['A'][2]) - local_origin
        self._base_vec_2 /= self._tube_width - 1
        self._bank_alignment = self._get_detector_bank_orientation()
        self._detector_tube = Tube(tuple(self._bank_geometry['A'][0]),
                                   tuple(self._bank_geometry['B'][0]),
                                   self._bank_alignment)
        self._nexus_dict = {}

    def _set_bank_geometry(self, bank_geo: Dict) -> Dict:
        for i in range(4):
            bank_geo['A'][i] = np.array(bank_geo['A'][i]) * SCALE_FACTOR \
                               - self._bank_translation
            bank_geo['B'][i] = np.array(bank_geo['B'][i]) * SCALE_FACTOR \
                               - self._bank_translation
            bank_geo['A'][i] = tuple(bank_geo['A'][i])
            bank_geo['B'][i] = tuple(bank_geo['B'][i])
        self._bank_translation += self._bank_offset

        return bank_geo

    def get_corners(self):
        return self._bank_geometry['A'] + self._bank_geometry['B']

    def get_bank_id(self):
        return self._bank_id

    def _get_tube_point_offsets(self) -> List:
        # Generate tube offsets according to tube layout and the provided
        # grid corners.
        xyz_offsets = []
        for j_i in range(self._tube_depth):
            for k_i in range(self._tube_width):
                xyz_offset = self._base_vec_1 * j_i + self._base_vec_2 * k_i
                xyz_offsets.append(xyz_offset)
        return xyz_offsets

    def build_detector_bank(self):
        tube_point_offsets = self._get_tube_point_offsets()
        self._detector_tube.set_xyz_offsets(tube_point_offsets)
        self._detector_tube.populate_with_uniform_straws(self._bank_id,
                                                         self._base_vec_1,
                                                         self._base_vec_2)
        return self._detector_tube

    def _calculate_tube_length(self, index=0):
        return np.linalg.norm(np.array(self._bank_geometry['A'][index]) -
                              np.array(self._bank_geometry['B'][index]))

    def _check_tube_center_distance(self):
        width = np.linalg.norm(np.array(self._bank_geometry['A'][0]) -
                               np.array(self._bank_geometry['A'][1]))
        center_dist = round(width / (self._tube_depth - 1),
                            FRACTIONAL_PRECISION)
        if center_dist < IMAGING_TUBE_D:
            raise ValueError(f'distance between tubes in '
                             f'detector bank {self._bank_id} '
                             f'is smaller than the'
                             f' tube diameter {center_dist} < {IMAGING_TUBE_D}')

    def _is_bank_cuboid(self):
        euclid_dist = []
        for index in range(len(self._bank_geometry['A'])):
            euclid_dist.append(self._calculate_tube_length(index))
        if len(set(euclid_dist)) != 1:
            raise ValueError(f'Error: Bank {self._bank_id} '
                             f'does not form a cuboid.')

    def _grid_corners_in_plane(self, grid_corners):
        for i in range(3):
            point_coord = [point[i] for point in grid_corners]
            if len(set(point_coord)) == 1:
                return
        else:
            print(grid_corners)
            raise RuntimeError(f'Something is wrong with the corner points'
                               f' in bank {self._bank_id}. '
                               f'The corner points are not in a plane.')

    def _get_detector_bank_orientation(self):
        unit_vectors_xyz = {'x': np.array([1, 0, 0]),
                            'y': np.array([0, 1, 0]),
                            'z': np.array([0, 0, 1])}

        tube_direction = np.array(self._bank_geometry['A'][0]) - \
                         np.array(self._bank_geometry['B'][0])
        for label_key in unit_vectors_xyz:
            scalar_product = np.dot(unit_vectors_xyz[label_key], tube_direction)
            if scalar_product and label_key == 'x':
                return DetectorAlignment.HORIZONTAL
            elif scalar_product and label_key == 'y':
                return DetectorAlignment.VERTICAL

        raise ValueError(f'The alignment of bank detector {self._bank_id} is'
                         f'unreasonable.')

    def get_bank_translation(self):
        return self._bank_translation

    def compound_data_in_dict(self) -> Dict:
        return self._detector_tube.compound_data_in_dict()

    def compound_data_in_list(self) -> List:
        return self._detector_tube.compound_data_in_list()

    def compound_detector_geometry(self, transform_path='',
                                   transform_as_nxlog=False):
        """
        Creates a dictionary of the LoKI detector geometry suitable for
        the NexusFileBuilder class.
        """
        detector_geo = self._detector_tube.get_geometry_data()
        geo_data = \
            NexusInfo.get_transformations_as_dict(detector_geo,
                                                  self._bank_translation,
                                                  transform_path,
                                                  as_nx_log=transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_detector_class_attr())
        return self._nexus_dict

    def add_data(self, det_data, time_of_flight):
        data_nexus = \
            NexusInfo.get_nx_log_group(
                nx_log_data=NexusInfo.get_values_attrs_as_dict(det_data),
                time=time_of_flight, x_label=TOF)
        self._nexus_dict[VALUES].update({'data': data_nexus})

    def add_static_data(self, det_data, time_of_flight):
        data_to_nxdata = {
            'data': NexusInfo.get_values_attrs_as_dict(det_data),
        }
        self._nexus_dict[VALUES].update(
            data_to_nxdata
        )
        data_to_nxdata = {
            'tof': NexusInfo.get_values_attrs_as_dict(time_of_flight,
                                                      {UNITS: 's'}),
        }
        self._nexus_dict[VALUES].update(
            data_to_nxdata
        )

    def get_nexus_dict(self):
        return self._nexus_dict

    def get_number_of_pixels(self):
        return self._nbr_of_tubes * STRAW_RESOLUTION * NUM_STRAWS_PER_TUBE


class Entry:
    """
    Simple representation of a NeXus Entry.
    """

    def __init__(self, experiment_id: int, title: str,
                 experiment_desc: str = ''):
        self._experiment_id = experiment_id
        self._title = title
        self._experiment_desc = experiment_desc

    def get_nx_entry(self, start_time):
        return {ENTRY: NexusInfo.get_values_attrs_as_dict(
            {
                INSTRUMENT:
                    NexusInfo.get_values_attrs_as_dict(
                        {}, NexusInfo.get_instrument_class_attr()),
                'title': NexusInfo.get_values_attrs_as_dict(self._title),
                'experiment_identifier': NexusInfo.get_values_attrs_as_dict(
                    self._experiment_id),
                'experiment_description': NexusInfo.get_values_attrs_as_dict(
                    self._experiment_desc),
                # 'start_time': NexusInfo.get_values_attrs_as_dict(start_time)
            },
            NexusInfo.get_entry_class_attr())}


class SimpleNexus(ABC):
    """
    Abstraction of a simple nexus class.
    """

    def __init__(self, position: tuple, name: str = ''):
        self._position = np.array(position) * SCALE_FACTOR
        self._name: str = name
        self._nexus_dict = {}

    def _get_transformation(self, transform_path, transform_nx_log):
        return NexusInfo.get_transformations_as_dict({},
                                                     self._position,
                                                     transform_path,
                                                     self._name,
                                                     as_nx_log=transform_nx_log)

    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        """
            Creates a dictionary of the simple nexus class geometry suitable for
            the NexusFileBuilder class.
        """
        raise NotImplementedError

    def get_nexus_dict(self):
        return self._nexus_dict


class Source(SimpleNexus):
    """
    Abstraction of an instrument source.
    """

    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        """
        Creates a dictionary of the source geometry suitable for
        the NexusFileBuilder class.
        """
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_source_class_attr())
        self._nexus_dict[VALUES]['type'] = NexusInfo.get_values_attrs_as_dict(
            'Spallation Neutron Source')
        self._nexus_dict[VALUES]['probe'] = NexusInfo.get_values_attrs_as_dict(
            'neutron')
        return self._nexus_dict


class Sample(SimpleNexus):
    """
        Abstraction of an instrument sample.
    """

    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        """
        Creates a dictionary of the sample geometry suitable for
        the NexusFileBuilder class.
        """
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_sample_class_attr())
        return self._nexus_dict


class DiskChopper(SimpleNexus):
    """
        Abstraction of an instrument disk chopper.
    """
    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_disk_chopper_class_attr())
        return self._nexus_dict

    def compound_geometry_extended(self, transform_path, rot_speed,
                                   disk_radius, slits, delay,
                                   transform_as_nxlog=False):
        """
            Creates a dictionary of the disk chopper geometry suitable for
            the NexusFileBuilder class.
        """
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        rotation_speed_nexus = NexusInfo.get_values_attrs_as_dict(
            rot_speed, {UNITS: 'Hz'})
        geo_data['rotation_speed'] = \
            NexusInfo.get_nx_log_group(rotation_speed_nexus)
        geo_data['top_dead_center'] = \
            NexusInfo.get_nx_log_group(NexusInfo.get_values_attrs_as_dict('1970-01-01T00:00:00.0Z'), module="tdct")
        delay_nexus = NexusInfo.get_values_attrs_as_dict(
            delay, {UNITS: 'ns'})
        geo_data['delay'] = \
            NexusInfo.get_nx_log_group(delay_nexus)
        geo_data['radius'] = NexusInfo.get_values_attrs_as_dict(
            disk_radius, {UNITS: LENGTH_UNIT})
        geo_data['slits'] = NexusInfo.get_values_attrs_as_dict(slits)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_disk_chopper_class_attr())
        return self._nexus_dict


class Monitor(SimpleNexus):
    """
        Abstraction of an instrument monitor.
    """

    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        """
            Creates a dictionary of the monitor geometry suitable for
            the NexusFileBuilder class.
        """
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data,
            NexusInfo.get_monitor_class_attr())
        return self._nexus_dict

    def add_data(self, mon_data):
        data_nexus = \
            NexusInfo.get_nx_log_group(
                nx_log_data=NexusInfo.get_values_attrs_as_dict(mon_data))
        self._nexus_dict[VALUES].update({'data': data_nexus})


class Slit(SimpleNexus):
    """
        Abstraction of an instrument slit.
    """

    def compound_geometry(self, transform_path, transform_as_nxlog=False):
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data, NexusInfo.get_slit_class_attr())
        return self._nexus_dict

    def compound_geometry_extended(self, transform_path, x_gap, y_gap,
                                   gap_unit='m',
                                   transform_as_nxlog=False,
                                   gaps_as_nx_log=False):
        """
            Creates a dictionary of the slit geometry suitable for
            the NexusFileBuilder class.
        """
        if gaps_as_nx_log:
            x_gap_nexus = NexusInfo.get_nx_log_group(nx_log_data=NexusInfo.get_values_attrs_as_dict(x_gap))
            y_gap_nexus = NexusInfo.get_nx_log_group(nx_log_data=NexusInfo.get_values_attrs_as_dict(y_gap))
        else:
            x_gap_nexus = NexusInfo.get_values_attrs_as_dict(x_gap,
                                                             {UNITS: gap_unit})
            y_gap_nexus = NexusInfo.get_values_attrs_as_dict(y_gap,
                                                             {UNITS: gap_unit})
        geo_data = self._get_transformation(transform_path, transform_as_nxlog)
        geo_data['x_gap'] = x_gap_nexus
        geo_data['y_gap'] = y_gap_nexus
        self._nexus_dict = NexusInfo.get_values_attrs_as_dict(
            geo_data, NexusInfo.get_slit_class_attr())
        return self._nexus_dict


class EventData:

    def __init__(self, ev_id=None, ev_index=None, ev_time_offset=None,
                 ev_time_zero=None):
        if ev_time_zero is None:
            ev_time_zero = []
        if ev_time_offset is None:
            ev_time_offset = []
        if ev_id is None:
            ev_id = []
        if ev_index is None:
            ev_index = []
        self._event_id = ev_id
        self._event_index = ev_index
        self._event_time_zero = ev_time_zero
        self._event_time_offset = ev_time_offset
        self._nx_event_data = {}
        self._extract_and_add_data()

    def _extract_and_add_data(self):
        # Leave empty as we do not need this data and it is not there.
        time_stamp = '2021-05-17T17:58:54'
        cue_index = NexusInfo.get_values_attrs_as_dict([])
        cue_timestamp_zero = \
            NexusInfo.get_values_attrs_as_dict([], {UNITS: 'ns',
                                                    'start': time_stamp})
        self._nx_event_data['cue_index'] = cue_index
        self._nx_event_data['cue_timestamp_zero'] = cue_timestamp_zero
        # event time offset
        self._nx_event_data['event_time_offset'] = \
            NexusInfo.get_values_attrs_as_dict(self._event_time_offset,
                                               {UNITS: 'ns'})
        # event time zero
        self._nx_event_data['event_time_zero'] = \
            NexusInfo.get_values_attrs_as_dict(self._event_time_zero,
                                               {UNITS: 's',
                                                'start': time_stamp})
        # event id
        self._nx_event_data['event_id'] = \
            NexusInfo.get_values_attrs_as_dict(self._event_id)
        # event index
        self._nx_event_data['event_index'] = \
            NexusInfo.get_values_attrs_as_dict(self._event_index)

    def get_nx_event_data(self, topic=None, source=None):
        if not topic:
            topic = 'topic'
        if not source:
            source = 'source'
        return NexusInfo.get_event_data(self._nx_event_data, topic, source)


class NexusFileBuilder:
    """
    Generates a nexus file based on data_struct which provides the overall
    definition and data content of the nexus that is supposed to be created.
    """

    def __init__(self, data_struct: Dict, filename: str = 'loki',
                 file_format: str = 'nxs'):
        self.data_struct = data_struct
        if '.' + file_format not in filename:
            filename = '.'.join([filename, file_format])
        self.hf5_file = h5py.File(filename, 'w')

    def construct_nxs_file(self):
        self._construct_nxs_file(self.data_struct, self.hf5_file)
        self.hf5_file.close()

    def _construct_nxs_file(self, nxs_data, group):
        for element in nxs_data:
            if element == MODULE:
                pass
            elif isinstance(nxs_data[element][VALUES], VALID_ARRAY_TYPES_NXS):
                d_set = group.create_dataset(element,
                                             data=nxs_data[element][VALUES])
                self._add_attributes(nxs_data[element], d_set)
            elif isinstance(nxs_data[element][VALUES], VALID_DATA_TYPES_NXS):
                d_set = group.create_dataset(element,
                                             data=nxs_data[element][VALUES])
                self._add_attributes(nxs_data[element], d_set)
            else:
                new_group = group.create_group(element)
                self._add_attributes(nxs_data[element], new_group)
                self._construct_nxs_file(nxs_data[element][VALUES], new_group)

    @staticmethod
    def _add_attributes(data_, d_set):
        if data_[ATTR]:
            for attr in data_[ATTR]:
                d_set.attrs[attr] = data_[ATTR][attr]


class NexusFileLoader:
    """
    Loads a nexus file.
    """

    def __init__(self, file_path):
        self._file_path = file_path
        self._nexus_content = None

    def close(self):
        if self._nexus_content:
            self._nexus_content.close()

    def load_file(self):
        self._nexus_content = h5py.File(self._file_path, 'r')

    def get_data(self, dot_path, get_attrs=False):
        dot_path_list = dot_path.split('.')
        nx_data = self._get_data(dot_path_list, self._nexus_content)
        if get_attrs:
            nx_attributes = \
                self._get_attributes(dot_path_list, self._nexus_content)
            return nx_data, nx_attributes
        return nx_data

    def get_attributes(self, dot_path):
        dot_path_list = dot_path.split('.')
        return self._get_attributes(dot_path_list, self._nexus_content)

    def _get_data(self, dot_path_list, nexus_data):
        if len(dot_path_list) > 1:
            return self._get_data(dot_path_list[1:],
                                  nexus_data[dot_path_list[0]])
        else:
            return nexus_data[dot_path_list[0]]

    def _get_attributes(self, dot_path_list, nexus_data):
        if len(dot_path_list) > 1:
            return self._get_attributes(dot_path_list[1:],
                                        nexus_data[dot_path_list[0]])
        else:
            return nexus_data[dot_path_list[0]].attrs


FLOAT, INTEGER, STRING = "float", "int32", "string"


class JsonConfigTranslator:

    def __init__(self, nexus_struct, json_filename='config.json'):
        self._nexus_struct = nexus_struct
        self._json_filename = json_filename
        self._json_config = {CHILDREN: []}

    def translate(self):
        entry_name = list(self._nexus_struct)[0]
        res = self._translate(entry_name, self._nexus_struct[entry_name])
        self._json_config[CHILDREN].append(res)

    @staticmethod
    def _check_type(val):
        if isinstance(val, tuple):
            val = val[0]
        if isinstance(val, str):
            dtype = STRING
        elif isinstance(val, int):
            dtype = INTEGER
        else:
            dtype = FLOAT
        return dtype

    def _extract_value_and_dtype(self, t):
        if isinstance(t, float):
            return t, FLOAT
        elif isinstance(t, int):
            return t, INTEGER
        elif isinstance(t, str):
            return t, STRING
        elif np.isscalar(t[0]) and len(t) == 1:
            return t[0], self._check_type(t[0])
        else:
            if isinstance(t[0], np.ndarray):
                t_list = []
                for list_item in t:
                    t_list.append(list_item.tolist())
                return t_list, self._check_type(t_list[0])
            return t, self._check_type(t[0])

    @staticmethod
    def _get_attributes(nexus_dict):
        attributes = []
        if nexus_dict[ATTR]:
            for name, value in nexus_dict[ATTR].items():
                if value:
                    attributes.append({
                        NAME: name,
                        VALUES: value
                    })
        return attributes

    def _translate(self, object_name, nexus_dict):
        children = []

        # Populate children.
        if not isinstance(nexus_dict[VALUES], dict):
            nxs_data, dtype = self._extract_value_and_dtype(nexus_dict[VALUES])
            child = {MODULE: DATASET, CONFIG: {
                NAME: object_name,
                VALUES: nxs_data,
                TYPE: dtype
            }}
            attributes = self._get_attributes(nexus_dict)
            if attributes:
                child[ATTR] = attributes
            return child
        elif MODULE in nexus_dict:
            children.append(
                nexus_dict[MODULE]
            )
        else:
            if NX_CLASS in nexus_dict[ATTR] and nexus_dict[ATTR][NX_CLASS] == 'NXlog':
                children.append(nexus_dict[VALUES][MODULE])
            else:
                for name, value in nexus_dict[VALUES].items():
                    child = self._translate(name, value)
                    if child:
                        children.append(child)
        output_dict = {
            NAME: object_name,
            TYPE: 'group',
            CHILDREN: children,
        }

        # Populate attributes if they exist.
        attributes = self._get_attributes(nexus_dict)
        if attributes:
            output_dict[ATTR] = attributes
        return output_dict

    def save_to_json(self):
        with open(self._json_filename, 'w', encoding='utf-8') as file:
            # json.dump(self._json_config, file, indent=4)
            json.dump(self._json_config, file, separators=(',', ':'))


def run_create_geometry():
    plot_tube_locations = False
    plot_endpoint_locations = False
    generate_nexus_content_into_nxs = True
    generate_nexus_content_into_csv = True
    add_simulated_data_to_nxs = False
    add_larmor_isis_data_to_nxs = False
    add_nurf_to_nxs = False
    # bank_ids_transform_as_nxlog = [n for n in range(0, 9)]
    bank_ids_transform_as_nxlog = [-1]
    detector_banks: List[Bank] = []
    ax = plt.axes(projection='3d')

    detector_data = []
    monitor_data = []
    tof_data = []
    pixel_id_data = []
    event_id = []
    event_index = []
    event_time_offset = []
    event_time_zero = []
    if add_larmor_isis_data_to_nxs:
        nexus_loader = NexusFileLoader(isis_larmor_data_filepath)
        try:
            nexus_loader.load_file()
            event_data_nx = 'raw_data_1.detector_1_events.'
            event_id = nexus_loader.get_data(event_data_nx +
                                             'event_id')
            event_index = nexus_loader.get_data(event_data_nx +
                                                'event_index')
            event_time_offset = nexus_loader.get_data(event_data_nx +
                                                      'event_time_offset')
            event_time_zero = nexus_loader.get_data(event_data_nx +
                                                    'event_time_zero')
            # event id readout
            arr = np.zeros((3095936,), dtype='uint')
            event_id.read_direct(arr)
            event_id = arr
            # event index readout
            arr = np.zeros((4822,), dtype='uint')
            event_index.read_direct(arr)
            event_index = arr
            # event time offset
            arr = np.zeros((3095936,), dtype='uint')
            event_time_offset.read_direct(arr)
            event_time_offset = arr * 1000  # to ns.
            # event time zero
            arr = np.zeros((4822,), dtype='float')
            event_time_zero.read_direct(arr)
            event_time_zero = arr
        except (TypeError, FileNotFoundError) as e:
            print(e)
    elif add_simulated_data_to_nxs:
        nexus_loader = NexusFileLoader(detector_data_filepath)
        try:
            nexus_loader.load_file()
            detector_data = \
                nexus_loader.get_data('mantid_workspace_1.workspace.'
                                      'values')
            monitor_data = \
                nexus_loader.get_data('mantid_workspace_1.instrument.'
                                      'detector.'
                                      'detector_count')

            # detector count data
            monitor_count = 10
            arr = np.zeros((axis_2_size + monitor_count, axis_1_size),
                           dtype='int16')
            detector_data.read_direct(arr)
            detector_data = arr[monitor_count:, :]

            # monitor event count data.
            # arr = np.zeros((axis_2_size, ), dtype='int32')
            # monitor_data.read_direct(arr)
            # monitor_data = arr

            # time of flight data
            tof_data = nexus_loader.get_data('mantid_workspace_1.workspace.'
                                             'axis1')
            arr = np.zeros((axis_1_size + 1,), dtype='int32')
            tof_data.read_direct(arr)
            tof_data = arr

            # pixel id data
            pixel_id_data = nexus_loader. \
                get_data('mantid_workspace_1.workspace.axis2')
            arr = np.zeros((axis_2_size + monitor_count,), dtype='int32')
            pixel_id_data.read_direct(arr)
            pixel_id_data = arr[:-monitor_count]
        except (TypeError, FileNotFoundError) as e:
            detector_data = []
            monitor_data = []
            tof_data = []
            pixel_id_data = []
            print(e)
    # event_data = EventData(pixel_id_data, tof_data, detector_data)
    event_data = EventData(event_id,
                           event_index,
                           event_time_offset,
                           event_time_zero)

    for loki_bank_id in det_banks_data:
        if plot_endpoint_locations:
            for idx in range(4):
                start_point = det_banks_data[loki_bank_id]['A'][idx]
                end_point = det_banks_data[loki_bank_id]['B'][idx]
                color = (random.random(), random.random(), random.random())
                offset = 0.001
                ax.plot([start_point[0] * SCALE_FACTOR,
                         end_point[0] * SCALE_FACTOR],
                        [start_point[1] * SCALE_FACTOR,
                         end_point[1] * SCALE_FACTOR],
                        [start_point[2] * SCALE_FACTOR + offset,
                         end_point[2] * SCALE_FACTOR + offset],
                        color=color)
        bank = Bank(det_banks_data[loki_bank_id], loki_bank_id)
        detector_tube = bank.build_detector_bank()
        bank_translation = bank.get_bank_translation()
        if plot_tube_locations:
            color = (random.random(), random.random(), random.random())
            xyz_offs = detector_tube.get_xyz_offsets()
            x_offset = [item[0] + bank_translation[0] for item in xyz_offs]
            y_offset = [item[1] + bank_translation[1] for item in xyz_offs]
            z_offset = [item[2] + bank_translation[2] for item in xyz_offs]
            start_point, end_point = detector_tube.get_endpoints()
            for idx in range(len(x_offset)):
                x_start = start_point[0] + x_offset[idx]
                x_end = end_point[0] + x_offset[idx]
                ax.plot([start_point[0] + x_offset[idx],
                         end_point[0] + x_offset[idx]],
                        [start_point[1] + y_offset[idx],
                         end_point[1] + y_offset[idx]],
                        [start_point[2] + z_offset[idx],
                         end_point[2] + z_offset[idx]],
                        color=color)
        detector_banks.append(bank)
    if plot_tube_locations or plot_endpoint_locations:
        plt.show()

    data = {}
    if generate_nexus_content_into_csv:
        for bank in detector_banks:
            data[bank.get_bank_id()] = bank.compound_data_in_list()
        if IMPORT_LARMOR:
            write_csv_file(data[0])
        else:
            write_csv_file(data[0] + data[4])
        reset_all_iterators()

    nx_entry = Entry(experiment_id="p1234", title="My experiment",
                     experiment_desc="this is an experiment")
    data = nx_entry.get_nx_entry(start_time=datetime.now().isoformat())

    start_index = 0
    end_index = start_index
    if generate_nexus_content_into_nxs:
        for c, bank in enumerate(detector_banks):
            end_index += bank.get_number_of_pixels()
            if NAME in det_banks_data[c]:
                key_det = det_banks_data[c][NAME]
            else:
                key_det = f'detector_{bank.get_bank_id()}'
            trans_path = f'/{ENTRY}/{INSTRUMENT}/{key_det}/{TRANSFORMATIONS}/'
            transform_nxlog = True if bank.get_bank_id() \
                                      in bank_ids_transform_as_nxlog else False
            bank.compound_detector_geometry(trans_path, transform_nxlog)
            if add_simulated_data_to_nxs:
                bank.add_static_data(detector_data[start_index:end_index],
                                     tof_data)
            item_det = bank.get_nexus_dict()
            data[ENTRY][VALUES][INSTRUMENT][VALUES][key_det] = item_det
            print(f'Detector {key_det} is done!')
            start_index = end_index

        # Create source.
        if data_source:
            loki_source = Source(data_source[LOCATION],
                                 data_source[NAME])
            trans_path = f'/{ENTRY}/{INSTRUMENT}/{SOURCE}/{TRANSFORMATIONS}/'
            data[ENTRY][VALUES][INSTRUMENT][VALUES][SOURCE] = \
                loki_source.compound_geometry(trans_path)
            print(f'Source {SOURCE} is done!')

        # Create sample.
        if data_sample:
            loki_sample = Sample(data_sample[LOCATION], data_sample[NAME])
            transformation_path = f'/{ENTRY}/{SAMPLE}/{TRANSFORMATIONS}/'
            data[ENTRY][VALUES][SAMPLE] = \
                loki_sample.compound_geometry(transformation_path)
            print(f'Sample {SAMPLE} is done!')

        # Create choppers.
        for loki_chopper in data_disk_choppers:
            disk_chopper = DiskChopper(loki_chopper[LOCATION],
                                       loki_chopper[NAME])
            trans_path = f'/{ENTRY}/{INSTRUMENT}/{loki_chopper[NAME]}' \
                         f'/{TRANSFORMATIONS}/'
            data[ENTRY][VALUES][INSTRUMENT][VALUES][loki_chopper[NAME]] = \
                disk_chopper.compound_geometry_extended(
                    trans_path, loki_chopper['rotation_speed'],
                    loki_chopper['disk_rad'] * SCALE_FACTOR,
                    loki_chopper['slits'], loki_chopper['delay'])
            print(f'Chopper {loki_chopper[NAME]} is done!')

        # Create monitors.
        nx_log_transform_monitor = [False, False, False, False, True]
        for c, loki_monitor in enumerate(data_monitors):
            monitor = Monitor(loki_monitor[LOCATION], loki_monitor[NAME])
            trans_path = f'/{ENTRY}/{INSTRUMENT}/{loki_monitor[NAME]}' \
                         f'/{TRANSFORMATIONS}/'
            data[ENTRY][VALUES][INSTRUMENT][VALUES][loki_monitor[NAME]] = \
                monitor.compound_geometry(trans_path,
                                          nx_log_transform_monitor[c])
            if add_simulated_data_to_nxs and False:
                monitor.add_data(mon_data=monitor_data)
            print(f'Monitor {loki_monitor[NAME]} is done!')

        # Create slits.
        for loki_slit in data_slits:
            slit = Slit(loki_slit[LOCATION], loki_slit[NAME])
            trans_path = f'/{ENTRY}/{INSTRUMENT}/{loki_slit[NAME]}' \
                         f'/{TRANSFORMATIONS}/'
            data[ENTRY][VALUES][INSTRUMENT][VALUES][loki_slit[NAME]] = \
                slit.compound_geometry_extended(
                    trans_path, loki_slit['x_gap'] * SCALE_FACTOR,
                                loki_slit['y_gap'] * SCALE_FACTOR,
                    gap_unit=LENGTH_UNIT, gaps_as_nx_log=True)
            print(f'Slit {loki_slit[NAME]} is done!')

        # Create users.
        for c, user in enumerate(data_users):
            user_var = 'user_' + str(c)
            data[ENTRY][VALUES][user_var] = NexusInfo.get_nx_user(user)
            print(f'NXuser {user_var} is done!')

        # Throw everything into event data.
        for bank in det_banks_data:
            data[ENTRY][VALUES][INSTRUMENT][VALUES][det_banks_data[bank][NAME]][
                VALUES]['larmor_detector_events'] = \
                event_data.get_nx_event_data(det_banks_data[bank][TOPIC],
                                             det_banks_data[bank][SOURCE])
        for c, monitor in enumerate(data_monitors):
            data[ENTRY][VALUES][INSTRUMENT][VALUES][monitor[NAME]][VALUES][
                f'monitor_{c + 1}_events'] = \
                EventData().get_nx_event_data(monitor[TOPIC], monitor[SOURCE])

        translator = JsonConfigTranslator(data, json_filename=json_filename)
        translator.translate()
        translator.save_to_json()

        # Construct nexus file.
        nexus_file_builder = NexusFileBuilder(data, filename=file_name)
        nexus_file_builder.construct_nxs_file()

        # Add NURF Data.
        if add_nurf_to_nxs:
            dummy_file = '103418'
            path_to_dummy_nxs_file = '.'
            # get dummy data
            dummy_data = load_one_spectro_file(dummy_file,
                                               path_to_dummy_nxs_file)
            # append to data to Loki Nurf
            nurf_file_creator(file_name, '.', dummy_data)


if __name__ == '__main__':
    run_create_geometry()
