from nexusutils.nexusbuilder import NexusBuilder
import numpy as np

output_filename = 'example_nx_geometry.nxs'
with NexusBuilder(output_filename, compress_type='gzip', compress_opts=1) as builder:
    instrument_group = builder.add_instrument("TEAPOT")
    mon_1_group = builder.add_monitor("monitor_1", 0, np.array([0., 0., 1.]))

    builder.add_shape_from_file("teapot.off", mon_1_group, "shape")
    # Add an icosahedral sample
    sample_group = builder.add_sample("sample")
    builder.add_source("test_source", "source")
    # builder.add_shape_from_file("icosa.off", sample_group, "shape")

    mon_2_group = builder.add_monitor("monitor_2", 1, np.array([0., 0., 10.0]))
    builder.add_shape_from_file("death_star.off", mon_2_group, "shape")
