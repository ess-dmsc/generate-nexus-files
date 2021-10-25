from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, Tuple

import numpy as np
import pandas as pd  # type:ignore
from alive_progress import alive_bar

from utils import write_to_nexus_file, write_to_off_file

"""
Generates mesh geometry for DREAM Endcap detector from information from a GEANT4 simulation
"""


def find_voxel_vertices(
    dz: float,
    theta: float,
    phi: float,
    dy1: float,
    dx1: float,
    dx2: float,
    alp1: float,
    dy2: float,
    dx3: float,
    dx4: float,
    alp2: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ported from GEANT4
    http://www.apc.univ-paris7.fr/~franco/g4doxy/html/G4Trap_8cc-source.html
    http://geant4-userdoc.web.cern.ch/geant4-userdoc/UsersGuides/ForApplicationDeveloper/html/Detector/Geometry/geomSolids.html
    """
    ttheta_cphi = np.tan(theta) * np.cos(phi)
    ttheta_sphi = np.tan(theta) * np.sin(phi)
    talpha1 = np.tan(alp1)
    talpha2 = np.tan(alp2)

    pt_0 = np.array(
        (-dz * ttheta_cphi - dy1 * talpha1 - dx1, -dz * ttheta_sphi - dy1, -dz,)
    )
    pt_1 = np.array(
        (-dz * ttheta_cphi - dy1 * talpha1 + dx1, -dz * ttheta_sphi - dy1, -dz,)
    )
    pt_2 = np.array(
        (-dz * ttheta_cphi + dy1 * talpha1 - dx2, -dz * ttheta_sphi + dy1, -dz,)
    )
    pt_3 = np.array(
        (-dz * ttheta_cphi + dy1 * talpha1 + dx2, -dz * ttheta_sphi + dy1, -dz,)
    )
    pt_4 = np.array(
        (+dz * ttheta_cphi - dy2 * talpha2 - dx3, +dz * ttheta_sphi - dy2, +dz,)
    )
    pt_5 = np.array(
        (+dz * ttheta_cphi - dy2 * talpha2 + dx3, +dz * ttheta_sphi - dy2, +dz,)
    )
    pt_6 = np.array(
        (+dz * ttheta_cphi + dy2 * talpha2 - dx4, +dz * ttheta_sphi + dy2, +dz,)
    )
    pt_7 = np.array(
        (+dz * ttheta_cphi + dy2 * talpha2 + dx4, +dz * ttheta_sphi + dy2, +dz,)
    )
    return pt_0, pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7


def create_winding_order(
    number_of_voxels: int,
    vertices_in_voxel: int,
    vertices_in_each_face: int,
    vertex_start_index: int,
) -> np.ndarray:
    index_0 = []
    index_1 = []
    index_2 = []
    index_3 = []
    for voxel in range(number_of_voxels):
        start_index = (voxel * vertices_in_voxel) + vertex_start_index
        index_0.extend(
            [
                start_index,
                start_index,
                start_index,
                start_index + 1,
                start_index + 2,
                start_index + 4,
            ]
        )
        index_1.extend(
            [
                start_index + 2,
                start_index + 4,
                start_index + 1,
                start_index + 3,
                start_index + 6,
                start_index + 5,
            ]
        )
        index_2.extend(
            [
                start_index + 3,
                start_index + 6,
                start_index + 5,
                start_index + 7,
                start_index + 7,
                start_index + 7,
            ]
        )
        index_3.extend(
            [
                start_index + 1,
                start_index + 2,
                start_index + 4,
                start_index + 5,
                start_index + 3,
                start_index + 6,
            ]
        )

    data = np.column_stack(
        (vertices_in_each_face, index_0, index_1, index_2, index_3,)
    ).astype(np.int32)
    return data


def rotate_around_x(angle_degrees: float, vertex: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return rotation_matrix.dot(vertex)


def rotate_around_y(angle_degrees: float, vertex: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return rotation_matrix.dot(vertex)


def rotate_around_z(angle_degrees: float, vertex: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rotation_matrix.dot(vertex)


# TODO these numbers are approximate, check with Irina what they should be
sumo_number_to_angle: Dict[int, float] = {3: 10.0, 4: 17.0, 5: 23.0, 6: 29.0}
sumo_number_to_translation: Dict[int, np.ndarray] = {
    3: np.array([0, 410, -1300.0]),
    4: np.array([0, 590, -1310.0]),
    5: np.array([0, 780, -1325.0]),
    6: np.array([0, 1000, -1350.0]),
}


def create_voxelids_and_faces(geant_df, max_face_index, max_vertex_index):
    number_of_voxels = len(geant_df.index)
    vertices_in_voxel = 8
    faces_in_voxel = 6
    number_of_faces = faces_in_voxel * number_of_voxels

    voxel_ids = np.zeros((number_of_faces, 2))
    max_voxel_index = max_face_index / faces_in_voxel

    for voxel in range(number_of_voxels):
        # Map each face in the voxel to the voxel ID
        for face_number_in_voxel in range(faces_in_voxel):
            face = voxel * faces_in_voxel + face_number_in_voxel + max_face_index
            voxel_ids[voxel * faces_in_voxel + face_number_in_voxel, 0] = face
            voxel_ids[voxel * faces_in_voxel + face_number_in_voxel, 1] = (
                voxel + max_voxel_index
            )

    # Vertices making up each face of each voxel
    vertices_in_each_face = 4 * np.ones(number_of_faces)

    faces = create_winding_order(
        number_of_voxels, vertices_in_voxel, vertices_in_each_face, max_vertex_index
    )
    return faces, voxel_ids


def create_sector(geant_df, z_rotation_angle):
    number_of_voxels = len(geant_df.index)
    vertices_in_voxel = 8
    number_of_vertices = vertices_in_voxel * number_of_voxels

    x_coords = np.zeros(number_of_vertices)
    y_coords = np.zeros(number_of_vertices)
    z_coords = np.zeros(number_of_vertices)

    x_centre_coords = np.zeros(number_of_voxels)
    y_centre_coords = np.zeros(number_of_voxels)
    z_centre_coords = np.zeros(number_of_voxels)

    for voxel in range(number_of_voxels):
        voxel_vertices = find_voxel_vertices(
            geant_df["z"][voxel] / 2,
            0.0,
            0.0,
            geant_df["y2"][voxel] / 2,
            geant_df["x1"][voxel] / 2,
            geant_df["x1"][voxel] / 2,
            0.0,
            geant_df["y1"][voxel] / 2,
            geant_df["x2"][voxel] / 2,
            geant_df["x2"][voxel] / 2,
            0.0,
        )

        voxel_position = np.array(
            [
                geant_df["x_centre"][voxel],
                geant_df["y_centre"][voxel],
                geant_df["z_centre"][voxel],
            ]
        )

        for vert_number, vertex in enumerate(voxel_vertices):
            # Translate voxel to position in SUMO
            vertex += voxel_position

            # Rotate 10 degrees around y
            # This means the SUMO doesn't face the sample, and is done to
            # increase efficiency of the detector
            vertex = rotate_around_y(-10, vertex)

            sumo_number = geant_df["sumo"][voxel]
            vertex = rotate_around_x(sumo_number_to_angle[sumo_number], vertex)
            vertex += sumo_number_to_translation[sumo_number]

            # Rotate sector
            vertex = rotate_around_z(z_rotation_angle, vertex)

            x_coords[voxel * vertices_in_voxel + vert_number] = vertex[0]
            y_coords[voxel * vertices_in_voxel + vert_number] = vertex[1]
            z_coords[voxel * vertices_in_voxel + vert_number] = vertex[2]

        x_centre_coords[voxel] = np.mean(
            x_coords[
                voxel * vertices_in_voxel : voxel * vertices_in_voxel
                + vertices_in_voxel
            ]
        )
        y_centre_coords[voxel] = np.mean(
            y_coords[
                voxel * vertices_in_voxel : voxel * vertices_in_voxel
                + vertices_in_voxel
            ]
        )
        z_centre_coords[voxel] = np.mean(
            z_coords[
                voxel * vertices_in_voxel : voxel * vertices_in_voxel
                + vertices_in_voxel
            ]
        )

    vertex_coords = np.column_stack((x_coords, y_coords, z_coords))

    return (
        vertex_coords,
        x_centre_coords,
        y_centre_coords,
        z_centre_coords,
    )


if __name__ == "__main__":
    df = pd.read_csv(
        "LookupTableDreamEndCap_noRRT.txt", delim_whitespace=True, header=None
    )
    df.columns = [
        "sumo",
        "sect-seg",
        "strip",
        "wire",
        "counter",
        "x_centre",
        "y_centre",
        "z_centre",
        "x1",
        "x2",
        "y1",
        "y2",
        "z",
    ]

    faces_in_voxel = 6

    total_vertices = None
    total_faces = None
    total_ids = None
    x_offsets_total = None
    y_offsets_total = None
    z_offsets_total = None
    max_vertex_index = 0
    max_face_index = 0

    # TODO start and stop angle are inferred from diagrams, need to check
    n_sectors = 23
    z_rotation_angles_degrees = np.linspace(-138.0, 138.0, num=n_sectors)

    _create_sector = partial(create_sector, df)
    _create_voxelids_and_faces = partial(create_voxelids_and_faces, df)

    with ProcessPoolExecutor(max_workers=12) as executor:
        with alive_bar(
            len(z_rotation_angles_degrees), bar="blocks", spinner="triangles"
        ) as bar:
            for sector_vertices, x_offsets, y_offsets, z_offsets in executor.map(
                _create_sector, z_rotation_angles_degrees
            ):

                sector_faces, sector_ids = _create_voxelids_and_faces(
                    max_face_index, max_vertex_index
                )
                if total_vertices is None:
                    total_vertices = sector_vertices
                    total_faces = sector_faces
                    total_ids = sector_ids
                    x_offsets_total = x_offsets
                    y_offsets_total = y_offsets
                    z_offsets_total = z_offsets
                else:
                    total_vertices = np.vstack((total_vertices, sector_vertices))
                    total_faces = np.vstack((total_faces, sector_faces))
                    total_ids = np.vstack((total_ids, sector_ids))
                    x_offsets_total = np.vstack((x_offsets_total, x_offsets))
                    y_offsets_total = np.vstack((y_offsets_total, y_offsets))
                    z_offsets_total = np.vstack((z_offsets_total, z_offsets))

                max_vertex_index = total_vertices.shape[0]
                max_face_index = total_ids.shape[0]
                bar()

    write_to_off_file(
        f"DREAM_endcap_{n_sectors}_sectors.off",
        total_vertices.shape[0],
        total_faces.shape[0],
        total_vertices,
        total_faces,
    )

    write_to_nexus_file(
        f"DREAM_endcap_{n_sectors}_sectors.nxs",
        total_vertices,
        total_faces,
        total_ids,
        x_offsets_total,
        y_offsets_total,
        z_offsets_total,
    )
