import csv
import os
import matplotlib.pyplot as plt
import numpy as np

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
                offset = np.array([float(row['X']),
                                   float(row['Y']),
                                   float(row['Z'])])
            else:
                geo_data = {'Grid': row['Grid'], 'Row': row['Row'],
                            'Voxel': row['Voxel'],
                            'location':
                                np.array([float(row['X']),
                                          float(row['Y']),
                                          float(row['Z'])]) - offset}
                geo_data_list.append(geo_data)
                loc = tuple(geo_data['location'].tolist())
                x, y, z = loc
                ax.scatter(x, y, z, marker='o', color='b', edgecolor='r')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm')
        ax.set_zlabel('Z (mm)')
        plt.show()


