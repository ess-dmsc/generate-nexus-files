import json

CHILDREN = "children"
CONFIG="config"
VALUES = "values"

if __name__ == '__main__':
    data = {}
    with open("AMOR_nexus_structure.json", 'r') as json_file:
        data = json.load(json_file)

    multiblade_detector = data[CHILDREN][0][CHILDREN][1][CHILDREN][0][CHILDREN]
    pixel_ids = multiblade_detector[1][CONFIG][VALUES]
    x, y, z = multiblade_detector[5][CONFIG][VALUES], \
              multiblade_detector[6][CONFIG][VALUES], \
              multiblade_detector[7][CONFIG][VALUES]

    id_dict = dict(zip(pixel_ids, zip(x, y, z)))
    pixel_id = 2
    print("local coordinates of pixel:", id_dict[pixel_id])

    # from examples.amor.amor import STRIPS_PER_BLADE
    # number_of_rows = int(len(x) / STRIPS_PER_BLADE)
    # x_list = []
    # y_list = []
    # z_list = []
    # pixel_ids_list = []
    # for i in range(number_of_rows):
    #     x_list.append(x[i * STRIPS_PER_BLADE: (i + 1) * STRIPS_PER_BLADE])
    #     y_list.append(y[i * STRIPS_PER_BLADE: (i + 1) * STRIPS_PER_BLADE])
    #     z_list.append(z[i * STRIPS_PER_BLADE: (i + 1) * STRIPS_PER_BLADE])
    #     pixel_ids_list.append(pixel_ids_list[i * STRIPS_PER_BLADE: (i + 1) * STRIPS_PER_BLADE])

