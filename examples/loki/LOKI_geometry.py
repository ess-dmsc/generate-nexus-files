from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List

from numpy.linalg import norm

from detector_banks_geo import FRACTIONAL_PRECISION, NUM_STRAWS_PER_TUBE, \
    IMAGING_TUBE_D, STRAW_DIAMETER, TUBE_DEPTH, loki_banks

N_VERTICES = 3
STRAW_RESOLUTION = 512


def calc_straw_offset(bank_id) -> int:
    """
    Calculates the straw offset.

    ::return:: returns the straw offset based on the bank ID.
    """
    return bank_id


def calc_straw_local() -> int:
    """
    Calculates the local straw number.
    """
    pass


def calc_pixel_id(straw, local_pixel_position) -> int:
    return 1 + local_pixel_position + straw * STRAW_RESOLUTION


class VertexIdIterator:
    def __iter__(self):
        self._id: int = 0
        return self

    def __next__(self):
        current_id = self._id
        self._id += 1
        return current_id


vertex_id_iter = iter(VertexIdIterator())


class Vertex:
    """
    Description of a single vertex in a 3D space.
    """

    def __init__(self, x: float, y: float, z: float, vertex_id=None):
        self._x, self._y, self._z = x, y, z
        self._id = vertex_id

    def get_coordinates(self) -> tuple:
        return self._x, self._y, self._z

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

    def __init__(self, vertices_coordinates: tuple):
        if len(set(vertices_coordinates)) != N_VERTICES:
            raise RuntimeError('Three unique vertices are expected to fully'
                               'describe the cylinder geometry')
        else:
            self._vertices = []
            for vertex_coordinates in vertices_coordinates:
                self._vertices.append(Vertex(*vertex_coordinates,
                                             next(vertex_id_iter)))

    def get_vertices_ids(self) -> List[int]:
        return [self._vertices[i].get_vertex_id() for i in range(N_VERTICES)]

    def __repr__(self):
        return f'A: {self._vertices[0]} \n' \
               f'B: {self._vertices[1]} \n' \
               f'C: {self._vertices[2]}'


class Pixel(Cylinder):
    """
    Description of a detector pixel geometrically described as a cylinder.
    """

    def __init__(self, vertices_coordinates: tuple, pixel_number: int):
        super().__init__(vertices_coordinates)
        self._pixel_number = pixel_number

    def get_pixel_number(self) -> int:
        return self._pixel_number

    def __repr__(self):
        return f'Pixel number {self._pixel_number}: \n' + \
               super().__repr__()


class Straw:
    """
    Abstraction of a single straw consisting of STRAW_RESOLUTION pixels.
    """

    def __init__(self, point_a: tuple, point_b: tuple, point_c: tuple,
                 detector_bank_id: int):
        self._point_a = point_a
        self._point_b = point_b
        self._point_c = point_c
        self._xyz_offset = [None] * NUM_STRAWS_PER_TUBE
        self._calculate_straw_offsets()
        self._detector_bank_id = detector_bank_id
        # self._straw_id = self._get_straw_start()
        # self._define_pixels_in_straw()
        self._pixels = []

    def _calculate_straw_offsets(self):
        straw_offs = 0
        self._xyz_offset = straw_offs

    def _define_pixels_in_straw(self):
        for i in range(STRAW_RESOLUTION):
            straw_number = self._get_straw_number()
            tmp_vertices = [Vertex(calc_pixel_id()) for i in range(N_VERTICES)]
            self._pixels.append(Pixel(tmp_vertices, ))

    def _get_straw_number(self) -> int:
        # needs equations to determine pixel ids.
        return calc_straw_offset(self._detector_bank_id) + calc_straw_local()


class Tube:
    """
    Abstraction of a tube in a detector bank that holds several straws.
    coordinates should be provided as the center coordinates of the tube.
    It is also possible for the user to provide x, y and z offset to
    repeat the same tube geometry in a detector bank.
    This saves time and space when generating large number of detector banks
    with essentially the same tube geometry, albeit shifted.
    """

    def __init__(self, point_start: tuple, point_end: tuple):
        self._diameter = IMAGING_TUBE_D
        self._point_start = point_start
        self._point_end = point_end
        self._xyz_offsets: Dict = {}
        self._straw: Straw = None

    def set_xyz_offsets(self, xyz_offsets: Dict):
        self._xyz_offsets = xyz_offsets

    def get_xyz_offsets(self):
        return self._xyz_offsets

    def get_endpoints(self):
        return self._point_start, self._point_end

    def populate_with_uniform_straws(self, detector_bank_id: int,
                                     radial_vector: tuple):
        """
        Populates the tube with straws based on tube location in the 3D space.
        """
        point_radial = np.array(self._point_start) + \
                       (STRAW_DIAMETER / 2) * np.array(radial_vector)
        self._straw = Straw(self._point_start,
                            tuple(point_radial),
                            self._point_end,
                            detector_bank_id)


class Bank:
    """
    Abstraction of a detector bank consisting of multiple tubes.
    """

    def __init__(self, bank_geo: Dict, bank_id: int):
        self._bank_id = bank_id
        self._bank_side_a = bank_geo['A']
        self._bank_side_b = bank_geo['B']
        self._tube_depth = TUBE_DEPTH
        self._tube_width = int(bank_geo['num_tubes'] / TUBE_DEPTH)

        # Check that provided geometry seems feasible.
        self._is_bank_cuboid()
        self._check_tube_center_distance()

        # Check that the corner points are in a plane.
        # Otherwise something is wrong with the provided geometry.
        self._grid_corners_in_plane(self._bank_side_a)

        # The base vectors (in a plane) of the local coordinate system
        # are non-orthogonal and not normalized. This makes it easy to find the
        # global coordinates of the tube center points in the detector bank.
        local_origin = np.array(self._bank_side_a[0])
        self._base_vec_1 = np.array(self._bank_side_a[1]) - local_origin
        self._base_vec_1 /= self._tube_depth - 1
        self._base_vec_2 = np.array(self._bank_side_a[3]) - local_origin
        self._base_vec_2 /= self._tube_width - 1
        self._detector_tube = Tube(tuple(self._bank_side_a[0]),
                                   tuple(self._bank_side_b[0]))

    def get_corners(self):
        return self._bank_side_a + self._bank_side_b

    def get_bank_id(self):
        return self._bank_id

    def _get_tube_point_offsets(self, grid_corners) -> List[tuple]:

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
        tube_point_offsets = self._get_tube_point_offsets(self._bank_side_a)
        self._detector_tube.set_xyz_offsets(tube_point_offsets)
        r_vector = (self._base_vec_1 + self._base_vec_2)
        r_vector /= np.linalg.norm(self._base_vec_1 + self._base_vec_2)
        self._detector_tube.populate_with_uniform_straws(self._bank_id,
                                                         tuple(r_vector))
        return self._detector_tube

    def _calculate_tube_length(self, index=0):
        return np.linalg.norm(np.array(self._bank_side_a[index]) -
                              np.array(self._bank_side_b[index]))

    def _check_tube_center_distance(self):
        width = np.linalg.norm(np.array(self._bank_side_a[0]) -
                               np.array(self._bank_side_a[1]))
        center_dist = round(width / (self._tube_depth - 1),
                            FRACTIONAL_PRECISION)
        if center_dist < IMAGING_TUBE_D:
            raise ValueError(f'distance between tubes in '
                             f'detector bank {self._bank_id} '
                             f'is smaller than the'
                             f' tube diameter {center_dist} < {IMAGING_TUBE_D}')

    def _is_bank_cuboid(self):
        euclid_dist = []
        for idx in range(len(self._bank_side_a)):
            euclid_dist.append(self._calculate_tube_length(idx))
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


if __name__ == '__main__':
    plot_tube_locations = True
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
            xyz_offs = detector_tube.get_xyz_offsets().values()
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
    plt.show()
