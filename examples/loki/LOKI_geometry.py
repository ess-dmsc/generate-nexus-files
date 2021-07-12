import csv

import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional
from detector_banks_geo import FRACTIONAL_PRECISION, NUM_STRAWS_PER_TUBE, \
    IMAGING_TUBE_D, STRAW_DIAMETER, TUBE_DEPTH, STRAW_ALIGNMENT_OFFSET_ANGLE, \
    TUBE_OUTER_STRAW_DIST_FROM_CP, STRAW_RESOLUTION, loki_banks, SCALE_FACTOR, \
    LENGTH_UNIT

N_VERTICES = 3
ATTR = 'attributes'
ENTRY = 'entry'
INSTRUMENT = 'instrument'
NX_CLASS = 'NX_class'
VALUES = 'values'


def reorder_straw_offsets_in_list(straw_offs_unsorted: List):
    """
    Sorting of straw offsets based on information from LoKI detector bank
    description.
    """
    mid_point = int((NUM_STRAWS_PER_TUBE - 1) / 2 + 1)
    straw_offs_sorted = [0] * len(straw_offs_unsorted)
    straw_offs_sorted[mid_point] = straw_offs_unsorted[0]
    straw_offs_sorted[:mid_point] = straw_offs_unsorted[mid_point:]
    straw_offs_sorted[mid_point:] = reversed(
        straw_offs_unsorted[1:mid_point])
    return straw_offs_sorted


def write_csv_file(csv_data):
    column_names = ['bank id', 'tube id', 'straw id', 'local straw position',
                    'pixel id']
    with open('detector_geometry.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(column_names)
        for row in csv_data:
            csv_writer.writerow(row)


# Static class.
class NexusInfo:

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_entry_class_attr():
        return {NX_CLASS: 'NXentry'}

    @staticmethod
    def get_instrument_class_attr():
        return {NX_CLASS: 'NXinstrument'}

    @staticmethod
    def get_detector_class_attr():
        return {NX_CLASS: 'NXdetector'}

    @staticmethod
    def get_cylindrical_geo_class_attr():
        return {NX_CLASS: 'NXcylindrical_geometry'}

    @staticmethod
    def get_units_attribute(units):
        return {'units': units}

    @staticmethod
    def get_values_attrs_as_dict(values, attrs=None):
        return {VALUES: values,
                ATTR: attrs}


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
    def reset():
        return IdIterator()


pixel_id_iter = iter(IdIterator(1))
straw_id_iter = iter(IdIterator())


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

    def compound_data_in_dict(self, offset: np.array) -> Dict:
        point_a = self.nominal_vertices_coordinates['Vertex A']
        data_dict = {}
        for pixel_offset in self.pixel_xyz_offsets:
            data_dict[next(pixel_id_iter)] = \
                tuple(point_a + pixel_offset + offset)
        return data_dict

    def compound_data_in_list(self, bank_id: int, tube_id: int,
                              straw_id: int, offset: np.array) -> List:
        # point_a = np.array(self.nominal_vertices_coordinates['Vertex A'])
        data_list: List = []
        loc_pixel_id_iter = iter(IdIterator())
        print(offset)
        for pixel_offset in self.pixel_xyz_offsets:
            data_list.append((bank_id,
                              tube_id,
                              straw_id,
                              next(loc_pixel_id_iter),
                              next(pixel_id_iter)))
            print(pixel_offset)
        ret_val = data_list[:2] + data_list[-2:]
        return ret_val

    def get_pixel_data(self, offset: np.array):
        data_offsets: List = []
        data_detector_num: List = []
        for pixel_offset in self.pixel_xyz_offsets:
            data_offsets.append(offset + pixel_offset)
            data_detector_num.append(next(pixel_id_iter))
        return data_offsets, data_detector_num

    def get_cylinder_geo_data(self):
        return {'cylinders': NexusInfo.get_values_attrs_as_dict([0, 1, 2]),
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

    def set_straw_offsets(self, alignment: DetectorAlignment,
                          base_vec_1: np.array, plot_all: bool = False):
        straw_offs = [np.array([0, 0, 0])]
        if alignment is DetectorAlignment.HORIZONTAL:
            def rotation(theta):
                return np.array(
                    [[1, 0, 0],
                     [0, -np.sin(theta), np.cos(theta)],
                     [0, np.cos(theta), np.sin(theta)]])
        else:
            def rotation(theta):
                return np.array(
                    [[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
        base_vec_1 = base_vec_1 / np.linalg.norm(base_vec_1)
        rotation_angle = np.deg2rad(360 / (NUM_STRAWS_PER_TUBE - 1))
        for straw_idx in range(NUM_STRAWS_PER_TUBE - 1):
            tmp_angle = rotation_angle * straw_idx + \
                        STRAW_ALIGNMENT_OFFSET_ANGLE
            rotated_vector = np.dot(rotation(tmp_angle), base_vec_1)
            straw_offs.append(rotated_vector * TUBE_OUTER_STRAW_DIST_FROM_CP)
        if plot_all:
            ax_tmp = plt.axes(projection='3d')
            ax_tmp.scatter3D([value[0] for value in straw_offs],
                             [value[1] for value in straw_offs],
                             [value[2] for value in straw_offs])
            plt.show()
        straw_offs = reorder_straw_offsets_in_list(straw_offs)
        self._straw_xyz_offsets = straw_offs

    def populate_with_pixels(self, plot_all: bool = False):
        """
        Populates the tube straw with pixels according to its pixel resolution.
        """
        vector_along_straw = np.array(self._point_c) - np.array(self._point_a)
        vector_along_straw /= STRAW_RESOLUTION
        pixel_end_point = tuple(np.array(self._point_a) + vector_along_straw)
        vertices_first_pixel = [self._point_a, self._point_b, pixel_end_point]
        self._pixel = Pixel(vertices_first_pixel)
        offsets_pixel = [tuple(vector_along_straw * j)
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

    def compound_data_in_dict(self, offset: np.array) -> Dict:
        data_dict = {}
        for straw_offset in self._straw_xyz_offsets:
            data_dict[next(straw_id_iter)] = \
                self._pixel.compound_data_in_dict(straw_offset + offset)
        return data_dict

    def compound_data_in_list(self, tube_id: int, offset: np.array) -> List:
        data_list: List = []
        for straw_offset in self._straw_xyz_offsets:
            data_list += self._pixel.compound_data_in_list(
                self._detector_bank_id, tube_id,
                next(straw_id_iter), straw_offset + offset)
        return data_list

    def get_straw_data(self, offset: np.array):
        data_offsets: List = []
        data_detector_num: List = []
        for straw_offset in self._straw_xyz_offsets:
            tmp_offsets, tmp_data_detector_num = \
                self._pixel.get_pixel_data(straw_offset + offset)
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
        self._xyz_offsets: Dict = {}
        self._straw: Optional[Straw] = None

    def set_xyz_offsets(self, xyz_offsets):
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
        self._straw.set_straw_offsets(self._alignment, base_vec_1)
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
            return {"detector_number": empty_nexus_field,
                    "pixel_shape": empty_nexus_field,
                    "x_pixel_offset": empty_nexus_field,
                    "y_pixel_offset": empty_nexus_field,
                    "z_pixel_offset": empty_nexus_field}
        for tube_offset in self._xyz_offsets:
            tmp_offsets, tmp_data_detector_num = \
                self._straw.get_straw_data(tube_offset)
            data_offsets += tmp_offsets
            data_detector_num += tmp_data_detector_num

        pixel_shape = self._straw.get_straw_pixel_geometry()
        unit_m = NexusInfo.get_units_attribute(LENGTH_UNIT)
        return {
            "detector_number":
                NexusInfo.get_values_attrs_as_dict(data_detector_num),
            "pixel_shape":
                NexusInfo.get_values_attrs_as_dict(
                    pixel_shape,
                    NexusInfo.get_cylindrical_geo_class_attr()),
            "x_pixel_offset":
                NexusInfo.get_values_attrs_as_dict(
                    [x[0] for x in data_offsets], unit_m),
            "y_pixel_offset":
                NexusInfo.get_values_attrs_as_dict(
                    [y[1] for y in data_offsets], unit_m),
            "z_pixel_offset":
                NexusInfo.get_values_attrs_as_dict(
                    [z[2] for z in data_offsets], unit_m)}


class Bank:
    """
    Abstraction of a detector bank consisting of multiple tubes.
    """

    def __init__(self, bank_geo: Dict, bank_id: int):
        self._bank_id = bank_id
        self._bank_offset = np.array(bank_geo['bank_offset']) * SCALE_FACTOR
        self._bank_geometry = self._set_bank_geometry(bank_geo)
        self._tube_depth = TUBE_DEPTH
        self._tube_width = int(bank_geo['num_tubes'] / TUBE_DEPTH)

        # Check that provided geometry seems feasible.
        self._is_bank_cuboid()
        self._check_tube_center_distance()

        # Check that the corner points are in a plane.
        # Otherwise something is wrong with the provided geometry.
        self._grid_corners_in_plane(self._bank_geometry['A'])

        # The base vectors (in a plane) of the local coordinate system
        # are non-orthogonal and not normalized. This makes it easy to find the
        # global coordinates of the tube center points in the detector bank.
        local_origin = np.array(self._bank_geometry['A'][0])
        self._base_vec_1 = np.array(self._bank_geometry['A'][1]) - local_origin
        self._base_vec_1 /= self._tube_depth - 1
        self._base_vec_2 = np.array(self._bank_geometry['A'][3]) - local_origin
        self._base_vec_2 /= self._tube_width - 1
        self._bank_alignment = self._get_detector_bank_orientation()
        self._detector_tube = Tube(tuple(self._bank_geometry['A'][0]),
                                   tuple(self._bank_geometry['B'][0]),
                                   self._bank_alignment)

    @staticmethod
    def _set_bank_geometry(bank_geo: Dict) -> Dict:
        bank_offset = np.array(bank_geo['bank_offset'])
        for i in range(4):
            bank_geo['A'][i] = np.array(bank_geo['A'][i]) - bank_offset
            bank_geo['B'][i] = np.array(bank_geo['B'][i]) - bank_offset
            bank_geo['A'][i] = tuple(bank_geo['A'][i] * SCALE_FACTOR)
            bank_geo['B'][i] = tuple(bank_geo['B'][i] * SCALE_FACTOR)

        return bank_geo

    def get_corners(self):
        return self._bank_geometry['A'] + self._bank_geometry['B']

    def get_bank_id(self):
        return self._bank_id

    def _get_tube_point_offsets(self) -> OrderedDict:
        # Generate tube offsets according to tube layout and the provided
        # grid corners.
        xyz_offsets = OrderedDict()
        tube_id = 1
        for x_i in range(self._tube_depth):
            for y_i in range(self._tube_width):
                xyz_offset = self._base_vec_1 * x_i + self._base_vec_2 * y_i
                xyz_offsets[tube_id] = tuple(xyz_offset)
                tube_id += 1
        return xyz_offsets

    def build_detector_bank(self):
        tube_point_offsets = self._get_tube_point_offsets().values()
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
                               f' in bank {self._bank_id}.'
                               f'The corner points are not in a plane.')

    def _get_detector_bank_orientation(self):
        unit_vectors_xyz = {'x': np.array([1, 0, 0]),
                            'y': np.array([0, 1, 0]),
                            'z': np.array([0, 0, 1])}

        tube_direction = np.array(self._bank_geometry['A'][0]) - \
                         np.array(self._bank_geometry['B'][0])
        for key in unit_vectors_xyz:
            scalar_product = np.dot(unit_vectors_xyz[key], tube_direction)
            if scalar_product and key == 'x':
                return DetectorAlignment.HORIZONTAL
            elif scalar_product and key == 'y':
                return DetectorAlignment.VERTICAL

        raise ValueError(f'The alignment of bank detector {self._bank_id} is'
                         f'unreasonable.')

    def compound_data_in_dict(self) -> Dict:
        return self._detector_tube.compound_data_in_dict()

    def compound_data_in_list(self) -> List:
        return self._detector_tube.compound_data_in_list()

    def compound_detector_geometry(self):
        """
        Creates a dictionary of the LoKI detector geometry suitable for
        the NexusFileBuilder class.
        """
        return self._detector_tube.get_geometry_data()


class NexusFileBuilder:
    """
    Generates a nexus file based on data_struct which provides the overall
    definition and data content of the nexus that is supposed to be created.
    """

    def __init__(self, data_struct: Dict, file_name: str = 'loki',
                 file_format: str = 'nxs'):
        self.data_struct = data_struct
        self.hf5_file = h5py.File('.'.join([file_name, file_format]), 'w')

    def construct_nxs_file(self):
        self._construct_nxs_file(self.data_struct, self.hf5_file)
        self.hf5_file.close()

    def _construct_nxs_file(self, nxs_data, group):
        for element in nxs_data:
            if isinstance(nxs_data[element][VALUES], list):
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


if __name__ == '__main__':
    plot_tube_locations = True
    generate_nexus_content_into_csv = False
    generate_nexus_content_into_nxs = True
    detector_banks: List[Bank] = []
    ax = plt.axes(projection='3d')
    for loki_bank_id in loki_banks:
        bank = Bank(loki_banks[loki_bank_id], loki_bank_id)
        detector_tube = bank.build_detector_bank()
        if plot_tube_locations:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            xyz_offs = detector_tube.get_xyz_offsets()
            x_offset = [item[0] for item in xyz_offs]
            y_offset = [item[1] for item in xyz_offs]
            z_offset = [item[2] for item in xyz_offs]
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
    if plot_tube_locations:
        plt.show()

    data = {}
    if generate_nexus_content_into_csv:
        for bank in detector_banks:
            data[bank.get_bank_id()] = bank.compound_data_in_list()
        write_csv_file(data[0] + data[4])

    data = {ENTRY: NexusInfo.get_values_attrs_as_dict(
        {INSTRUMENT:
             NexusInfo.get_values_attrs_as_dict(
                 {}, NexusInfo.get_instrument_class_attr())
         }, NexusInfo.get_entry_class_attr())}
    if generate_nexus_content_into_nxs:
        for bank in detector_banks:
            key_det = f'detector_{bank.get_bank_id()}'
            item_det = NexusInfo.get_values_attrs_as_dict(
                bank.compound_detector_geometry(),
                NexusInfo.get_detector_class_attr()
            )
            data[ENTRY][VALUES][INSTRUMENT][VALUES][key_det] = item_det
            print(f'Detector {key_det} is done!')
    nexus_file_builder = NexusFileBuilder(data)
    nexus_file_builder.construct_nxs_file()
