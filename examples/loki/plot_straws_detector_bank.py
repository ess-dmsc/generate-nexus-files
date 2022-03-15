import csv
import os
from larmor_data import STRAW_RESOLUTION, STRAW_DIAMETER
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'detector_geometry.csv')
    fig, ax = plt.subplots()
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        radius = STRAW_DIAMETER * 1000 / 2
        for row in reader:
            if (int(row['pixel id']) - 1) % STRAW_RESOLUTION == 0:
                pos = (float(row['z']) * 1000, float(row['y']) * 1000)
                circle = plt.Circle(pos, radius)
                ax.set_aspect(1)
                ax.add_artist(circle)
                x, y = pos
                plt.text(x - radius / 2, y - radius / 4, row['straw id'])
        plt.xlim(-200, 50)
        plt.ylim(-900, 20)
        plt.xlabel('z (mm)')
        plt.ylabel('y (mm)')
        plt.show()
