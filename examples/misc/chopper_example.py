from nexusutils.nexusbuilder import NexusBuilder
import numpy as np

"""
Creates an example file containing an NXdisk_chopper
For use in https://github.com/nexusformat/features
"""

if __name__ == '__main__':
    output_filename = 'example_nx_disk_chopper.nxs'
    nx_entry_name = 'entry'
    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        inst_group = builder.add_instrument('CHP')

        chopper_group = builder.add_nx_group(inst_group, 'example_chopper', 'NXdisk_chopper')

        builder.add_dataset(chopper_group, 'name', 'Juelich, WFM Chopper, Disc 1')
        builder.add_dataset(chopper_group, 'slit_edges', np.array(
            [83.71, 94.7, 140.49, 155.79, 193.26, 212.56, 242.32, 265.33, 287.91, 314.37, 330.3, 360.]) + 15.0,
                            attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 6)
        builder.add_dataset(chopper_group, 'slit_height', 130., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 300., attributes={'units': 'mm'})

        chopper_feature_id = "B89B086951FEFDDF"
        builder.add_dataset(builder.root, "features", [int(chopper_feature_id, 16)])
