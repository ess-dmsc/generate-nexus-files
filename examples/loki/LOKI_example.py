from nexusutils.nexusbuilder import NexusBuilder
from nexusutils.detectorplotter import DetectorPlotter
import numpy as np

if __name__ == '__main__':
    output_filename = 'LOKI_example_gzip.hdf5'
    with NexusBuilder(output_filename, idf_file='LOKI_Tube_Definition.xml',
                      compress_type='gzip', compress_opts=1) as builder:
        builder.add_instrument_geometry_from_idf()

        # A few more details to flesh out the example
        builder.add_user('LOKI Team', 'ESS')

        det_ids = builder.add_fake_event_data(10, 10)

        # Create a detector-spectrum map for use with NeXus-Streamer
        spectrum_numbers = np.arange(1, len(det_ids))
        with open('LOKI_detspecmap.dat', 'w') as map_file:
            map_file.write("Number_of_entries\n")
            map_file.write("{}\n".format(len(det_ids)))
            map_file.write("Detector  Spectrum\n")
            for det_id, spec_number in zip(det_ids, spectrum_numbers):
                map_file.write("{}    {}\n".format(det_id, spec_number))

        builder.add_dataset(builder.root, 'name', 'LOKI', {'short_name': 'LOKI'})

    with DetectorPlotter(output_filename) as plotter:
        plotter.plot_pixel_positions()
