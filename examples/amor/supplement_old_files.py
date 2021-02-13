from examples.amor.amor import add_shape_to_detector, create_detector_shape_info
import argparse
from nexusutils.nexusbuilder import NexusBuilder
import h5py

"""
This script is intended to supplement old files from AMOR with detector geometry information
and make other tweaks to produce a file compatible with the NeXus Streamer tool
https://github.com/ess-dmsc/nexus-streamer

Tested with these files:
amor2020n000346.hdf
amor2020n000366.hdf
from approx Dec 2021
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--input-file",
        type=str,
        help="Local filename or full path to input NeXus file from AMOR, "
        "output file will be saved to save directory",
        required=True,
    )
    args = parser.parse_args()
    output_filename = f"{''.join(args.input_file.split('.')[:-1])}_tweaked.nxs"

    with NexusBuilder(
        output_filename, compress_type="gzip", compress_opts=1, nx_entry_name="entry"
    ) as builder:
        instrument_group = builder.add_instrument("AMOR", "instrument")
        detector_group = builder.add_nx_group(
            instrument_group, "multiblade_detector", "NXdetector"
        )
        vertices, voxels, detector_ids = create_detector_shape_info()
        transforms_group = add_shape_to_detector(
            builder, detector_group, detector_ids, voxels, vertices
        )

        with h5py.File(args.input_file, 'r') as input_file:
            input_file.copy("/instrument/stages", builder.root)
            input_file.copy("/experiment/user", builder.root)
            input_file.copy("/experiment/data", detector_group)
            input_file.copy("/experiment/proposal_id", builder.root)
            input_file.copy("/experiment/title", builder.root)
            input_file.copy("/instrument/facility", builder.root)

            # TODO, don't just copy start_time, convert it to ISO8601
            input_file.copy("/experiment/start_time", builder.root)
