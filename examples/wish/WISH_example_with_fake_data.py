from nexusutils.nexusbuilder import NexusBuilder
import numpy as np

if __name__ == '__main__':
    output_filename = 'WISH_example.nxs'

    with NexusBuilder(output_filename, idf_file='WISH_Definition_10Panels.xml',
                      compress_type='gzip', compress_opts=1) as builder:
        builder.add_instrument_geometry_from_idf()
        instrument_name = 'WISH'

        det_ids = builder.add_fake_event_data(10, 10)

        # Create a detector-spectrum map for use with NeXus-Streamer
        spectrum_numbers = np.arange(1, len(det_ids))
        with open(instrument_name + '_detspecmap.dat', 'w') as map_file:
            map_file.write("Number_of_entries\n")
            map_file.write("{}\n".format(len(det_ids)))
            map_file.write("Detector  Spectrum\n")
            for det_id, spec_number in zip(det_ids, spectrum_numbers):
                map_file.write("{}    {}\n".format(det_id, spec_number))

        builder.add_dataset(builder.root, 'name', instrument_name, {'short_name': instrument_name})
