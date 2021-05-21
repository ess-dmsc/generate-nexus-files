import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List
from detector_banks_geo import FRACTIONAL_PRECISION, IMAGING_TUBE_D, \
    TUBE_DEPTH, loki_banks

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

    def __init__(self, x: float, y: float, z: float, detector_bank_id: int):
        self._x, self._y, self.z = x, y, z
        self._detector_bank_id = detector_bank_id
        self._straw_id = self._get_straw_start()
        self._pixels = []
        self._define_pixels_in_straw()

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
    """

    def __init__(self, point_a: tuple, point_b: tuple, tube_id: int):
        self._diameter = IMAGING_TUBE_D
        self._straws = []
        self._tube_id = tube_id
        self._vertex_a = Vertex(*point_a)
        self._vertex_b = Vertex(*point_b)

    def populate_with_straws(self, detector_bank_id):
        """
        Populates the tube with straws based on the detector bank id and
        tube location in the 3D space.
        """
        pass


class Bank:
    """
    Abstraction of a detector bank consisting of multiple tubes.
    """
    axis_label = ['x', 'y', 'z']
    idx_3D_mesh_dict = {axis_label[0]: (1, 2, 0),
                        axis_label[1]: (0, 2, 1),
                        axis_label[2]: (0, 1, 2)}

    def __init__(self, bank_geo: Dict, bank_id: int):
        self._bank_id = bank_id
        self._bank_side_a = bank_geo['A']
        self._bank_side_b = bank_geo['B']
        self._tube_depth = TUBE_DEPTH
        self._tube_width = int(bank_geo['num_tubes'] / TUBE_DEPTH)
        self._tubes = {}

        # Check that provided geometry seems feasible.
        self._is_bank_cuboid()
        self._check_tube_center_distance()

    def get_corners(self):
        return self._bank_side_a + self._bank_side_b

    def get_bank_id(self):
        return self._bank_id

    def _get_tube_points(self, grid_corners) -> List[tuple]:

        # First make a check that the corner points are in a plane.
        # Otherwise something is wrong with the provided geometry.
        self._grid_corners_in_plane(grid_corners)

        # The base vectors (in a plane) of the local coordinate system
        # are non-orthogonal and not normalized. This makes it easy to find the
        # global coordinates of the tube center points.
        local_origin = np.array(grid_corners[0])
        vec_1 = np.array(grid_corners[1]) - local_origin
        vec_1 /= self._tube_depth - 1
        vec_2 = np.array(grid_corners[3]) - local_origin
        vec_2 /= self._tube_width - 1

        # Generate tube grid for one side of the detector bank that
        # is defined by grid corners.
        tube_points = []
        for x_i in range(self._tube_depth):
            for y_i in range(self._tube_width):
                new_point = local_origin + vec_1 * x_i + vec_2 * y_i
                tube_points.append(tuple(new_point))

        return tube_points

    def build_detector_bank(self):
        tube_points_side_a = self._get_tube_points(self._bank_side_a)
        tube_points_side_b = self._get_tube_points(self._bank_side_b)
        tube_id = 1
        for point_a, point_b in zip(tube_points_side_a, tube_points_side_b):
            self._tubes[tube_id] = Tube(point_a, point_b, tube_id)

        return tube_points_side_a, tube_points_side_b

    def _calculate_tube_length(self, idx=0):
        return np.linalg.norm(np.array(self._bank_side_a[idx]) -
                              np.array(self._bank_side_b[idx]))

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
        side_a, side_b = bank.build_detector_bank()
        if plot_tube_locations:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            ax.scatter3D([x[0] for x in side_a],
                         [y[1] for y in side_a],
                         [z[2] for z in side_a], color=color)
            ax.scatter3D([x[0] for x in side_b],
                         [y[1] for y in side_b],
                         [z[2] for z in side_b], color=color)
        detector_banks.append(bank)
    plt.show()
