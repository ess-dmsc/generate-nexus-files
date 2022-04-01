import csv
import os
import matplotlib.pyplot as plt
import numpy as np

Z_STEP = 25.0

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__),
                             'CSPEC_LET_Geometry.csv')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    geo_data_list = []
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        offset = None
        for row in reader:
            if offset is None:
                # (x, y, z) -> (y, z, x)
                offset = np.array([float(row['Y']),
                                   float(row['Z']),
                                   float(row['X'])])
            else:
                geo_data = {'Grid': f'Grid_1', 'Row': row['Row'],
                            'Voxel': row['Voxel'],
                            'location':
                            # (x, y, z) -> (y, z, x)
                                np.array([float(row['Y']),
                                          float(row['Z']),
                                          float(row['X'])]) - offset,
                            }
                geo_data_list.append(geo_data)

        pixel_id = 1
        voxel_full_geometry = []
        for i in range(51):
            for voxel in geo_data_list:
                x, y, z = voxel['location'].tolist()
                y += 25.0 * i
                geo_data = {'Grid': f'Grid_{i + 1}',
                            'Row': voxel['Row'],
                            'Voxel': voxel['Voxel'],
                            'location': np.array([x, y, z]),
                            'pixel_id': pixel_id}
                pixel_id += 1
                voxel_full_geometry.append(geo_data)
                ax.scatter(x, y, z, marker='o', color='b', edgecolor='r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()


