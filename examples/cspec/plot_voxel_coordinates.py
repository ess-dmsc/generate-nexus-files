import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_off_file(file_name, vertices, faces, first_line):
    with open(file_name, 'w') as f:
        f.write("OFF\n")
        f.write("# CSPEC geometry\n")
        pd.DataFrame(first_line).to_csv(f, sep=" ", header=None, index=False)
        for item in vertices:
            pd.DataFrame(item).to_csv(f, sep=" ", header=None, index=False)
        for item in faces:
            pd.DataFrame(item).to_csv(f, sep=" ", header=None, index=False)


if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__),
                             'CSPEC_LET_Geometry.csv')

    fig = plt.figure()
    plot_voxes = False
    ax = fig.add_subplot(projection='3d')
    geo_data_list = []
    # x_size = 25 / 1000
    # y_size = 25 / 1000
    # z_size = 10 / 1000
    x_size = 25 / 1000
    y_size = 25 / 1000
    z_size = 10 / 1000
    nbr_voxels = 0
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        offset = None
        for row in reader:
            if offset is None:
                # (x, y, z) -> (y, z, x)
                offset = np.array([float(row['Y']) / 1000,
                                   float(row['Z']) / 1000,
                                   float(row['X']) / 1000])
            else:
                geo_data = {'Grid': f'Grid_1', 'Row': row['Row'],
                            'Voxel': row['Voxel'],
                            'location':
                            # (x, y, z) -> (y, z, x)
                                np.array([float(row['Y']) / 1000,
                                          float(row['Z']) / 1000,
                                          float(row['X']) / 1000])  # + offset,
                            }
                geo_data_list.append(geo_data)
                nbr_voxels += 1

        pixel_id = 1
        voxel_full_geometry = []
        vertices_list = []
        faces_list = []
        nbr_grids = 51
        for i in range(nbr_grids):
            nbr_voxels = 0
            for voxel in geo_data_list:
                x, y, z = voxel['location'].tolist()
                y += y_size * i
                geo_data = {'Grid': f'Grid_{i + 1}',
                            'Row': voxel['Row'],
                            'Voxel': voxel['Voxel'],
                            'location': np.array([x, y, z]),
                            'pixel_id': pixel_id}
                pixel_id += 1
                vertices_temp = [[x - x_size, y, z + z_size],
                                 [x, y, z + z_size],
                                 [x - x_size, y + y_size, z + z_size],
                                 [x, y + y_size, z + z_size],
                                 [x - x_size, y + y_size, z],
                                 [x, y + y_size, z],
                                 [x - x_size, y, z],
                                 [x, y, z]]
                vertices = np.array(vertices_temp)
                faces = np.array(
                    [[4, 0 + 8 * i, 1 + 8 * i, 3 + 8 * i, 2 + 8 * i],
                     [4, 2 + 8 * i, 3 + 8 * i, 5 + 8 * i, 4 + 8 * i],
                     [4, 4 + 8 * i, 5 + 8 * i, 7 + 8 * i, 6 + 8 * i],
                     [4, 6 + 8 * i, 7 + 8 * i, 1 + 8 * i, 0 + 8 * i],
                     [4, 1 + 8 * i, 7 + 8 * i, 5 + 8 * i, 3 + 8 * i],
                     [4, 6 + 8 * i, 0 + 8 * i, 2 + 8 * i, 4 + 8 * i]])
                voxel_full_geometry.append(geo_data)
                vertices_list.append(vertices)
                faces_list.append(faces)
                nbr_voxels += 1
                if plot_voxes:
                    for item in vertices_temp:
                        ax.scatter(item[0], item[1], item[2],
                                   marker='o',
                                   color='b',
                                   edgecolor='r')
        if plot_voxes:
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            plt.show()
        write_off_file("CSPEC.off", vertices_list, faces_list,
                       np.array([[nbr_voxels * nbr_grids * 8,
                                  nbr_voxels * nbr_grids * 6,
                                  0]]))
