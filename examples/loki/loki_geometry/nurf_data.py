import errno
import os
import h5py
import numpy as np


def load_one_spectro_file(file_handle, path_rawdata):
    """
    This function loads one .nxs file containing spectroscopy data (fluo, uv). Data is stored in multiple np.ndarrays.

    In:
     file_handle: file_handle is a file number, one element is expected otherwise an error is raised
                type: list

    path_rawdata: Path to the raw data
                type: str

    Out:
        data: contains all relevant HDF5 entries and their content for the Nurf project (keys and values)
            type: dict

    """

    # create path to file, convert file_number to string
    file_path_spectro = os.path.join(path_rawdata, file_handle + '.nxs')

    # check if file exists
    if not os.path.isfile(file_path_spectro):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                file_path_spectro)

    # we are ready to load the data set
    # open the .nxs file and read the values
    with h5py.File(file_path_spectro, "r") as f:
        # access nurf sub-group
        nurf_group = '/entry0/D22/nurf/'

        # access keys in sub-group
        nurf_keys = list(f[nurf_group].keys())
        # how many keys exist
        len_nurf_keys = len(nurf_keys)

        # print(nurf_keys)
        # print(len_nurf_keys)

        # this returns a list with HDF5datasets (I think)
        # data_spectro_file=list(f[nurf_group].values())

        # data_spectro_file=f[nurf_group].values()
        # data_spectro_file=f[nurf_group]

        # extract all data of the nurf subgroup and store it in a new dict

        # initialise an empty dict
        data = {}

        for key in f[nurf_group].keys():
            # print(key)
            # this is how I get string giving the full path to this dataset
            path_dataset = f[nurf_group][key].name
            # print(path_dataset)
            # print(f[nurf_group][key].name) #this prints the full path to the dataset
            # print(type(f[path_dataset][:])) #this gives me access to the content of the data set, I could use : or () inside [], out: np.ndarray

            # This gives a dict with full path name as dict entry followed by value. No, I don't want this, but good to know.
            # data[f[nurf_group][key].name]=f[path_dataset][:]

            # This gives a dict where the keys corresponds to the key names of the h5 file.
            data[key] = f[path_dataset][:]

        # print(f[nurf_group].get('Fluo_spectra'))

        # print a hierachical view of the file (simple)
        # like this is would only go through subgroups
        # f[nurf_group].visititems(lambda x, y: print(x))

        # walk through the whole file and show the attributes (found on github as an explanation for this function)
        # def print_attrs(name, obj):
        #    print(name)
        #    for key, val in obj.attrs.items():
        #        print("{0}: {1}".format(key, val))
        # f[nurf_group].visititems(print_attrs)

        # print(data_spectro_file)

    # file_handle is returned as np.ndarray and its an np.array, elements correspond to row indices
    return data


def nurf_file_creator(loki_file, path_to_loki_file, data):
    """
    Appends NUrF group to LOKI NeXus file for ESS

    Args:
        loki_file (str): filename of NeXus file for Loki
        path_to_loki_file (str): File path where the NeXus file for LOKI is stored
        data (dict): Dictionary with dummy data for Nurf
    """

    # change directory where the loki.nxs is located
    os.chdir(path_to_loki_file)

    # open the file and append
    with h5py.File(loki_file, 'a') as hf:
        
        
        #print(list(hf['/entry/'].keys()))
        
        # create the various subgrous and their attributes

        # UV subgroup
        grp_uv = hf.create_group("/entry/instrument/UV")
        grp_uv.attrs["NX_class"] = 'NXdetector'
        grp_uv.create_dataset('description', data='UV')
        grp_uv.create_dataset('type',
                              data='HR4PRO-UV-VIS-ES, DH-2000-FHS-DUV-TTL, QP600-2-SR')
    
        # create the various datasets for UV
        uv_inttime = grp_uv.create_dataset('UV_IntegrationTime',
                                           data=data['UV_IntegrationTime'],
                                           shape=data[
                                               'UV_IntegrationTime'].shape,
                                           dtype=np.int32)
        uv_inttime.attrs['long_name'] = 'UV_IntegrationTime'
        uv_inttime.attrs['units'] = 's'  # TODO: unit to be verified

        uv_bkg = grp_uv.create_dataset('UV_background',
                                       data=data['UV_background'],
                                       shape=data['UV_background'].shape,
                                       dtype=np.float32)
        uv_bkg.attrs['long name'] = 'UV_background'

        uv_int0 = grp_uv.create_dataset('UV_intensity0',
                                        data=data['UV_intensity0'],
                                        shape=data['UV_intensity0'].shape,
                                        dtype=np.float32)
        uv_int0.attrs['long name'] = 'UV_intensity0'

        uv_spectra = grp_uv.create_dataset('UV_spectra',
                                           data=data['UV_spectra'],
                                           shape=data['UV_spectra'].shape,
                                           dtype=np.float32)
        uv_spectra.attrs['long name'] = 'UV_spectra'

        uv_wavelength = grp_uv.create_dataset('UV_wavelength',
                                              data=data['UV_wavelength'],
                                              shape=data['UV_wavelength'].shape,
                                              dtype=np.float32)
        uv_wavelength.attrs['units'] = 'nm'  # TODO: unit to be verified
        uv_wavelength.attrs['long name'] = 'UV_wavelength'

        # Fluorescence subgroup
        grp_fluo = hf.create_group("/entry/instrument/Fluorescence")
        grp_fluo.attrs["NX_class"] = 'NXdetector'
        # grp_fluo.attrs["description"] = 'Fluorescence'
        # grp_fluo.attrs["Fluo lightsource"] = 'LSM-265A,-310A with LDC-1'
        grp_fluo.create_dataset('description', data='LSM-265A,-310A with LDC-1')

        # create the various datasets for fluo
        fluo_inttime = grp_fluo.create_dataset('Fluo_IntegrationTime',
                                               data=data[
                                                   'Fluo_IntegrationTime'],
                                               shape=data[
                                                   'Fluo_IntegrationTime'].shape,
                                               dtype=np.int32)
        fluo_inttime.attrs['units'] = 's'  # TODO: unit to be verified
        fluo_inttime.attrs['long name'] = 'Fluo_IntegrationTime'

        fluo_bkg = grp_fluo.create_dataset('Fluo_background',
                                           data=data['Fluo_background'],
                                           shape=data['Fluo_background'].shape,
                                           dtype=np.float32)
        fluo_bkg.attrs['long name'] = 'Fluo_background'

        fluo_int0 = grp_fluo.create_dataset('Fluo_intensity0',
                                            data=data['Fluo_intensity0'],
                                            shape=data['Fluo_intensity0'].shape,
                                            dtype=np.float32)
        fluo_int0.attrs['long name'] = 'Fluo_intensity0'

        fluo_monowavelengths = grp_fluo.create_dataset('Fluo_monowavelengths',
                                                       data=data[
                                                           'Fluo_monowavelengths'],
                                                       shape=data[
                                                           'Fluo_monowavelengths'].shape,
                                                       dtype=np.float32)
        fluo_monowavelengths.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_monowavelengths.attrs['long name'] = 'Fluo_monowavelengths'

        fluo_spectra = grp_fluo.create_dataset('Fluo_spectra',
                                               data=data['Fluo_spectra'],
                                               shape=data['Fluo_spectra'].shape,
                                               dtype=np.float32)
        fluo_spectra.attrs['long name'] = 'Fluo_spectra'

        fluo_wavelength = grp_fluo.create_dataset('Fluo_wavelength',
                                                  data=data['Fluo_wavelength'],
                                                  shape=data[
                                                      'Fluo_wavelength'].shape,
                                                  dtype=np.float32)
        fluo_wavelength.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_wavelength.attrs['long name'] = 'Fluo_wavelength'

        # dummy groups, no information currently available
        grp_sample_cell = hf.create_group("/entry/sample/Sample_cell")
        grp_sample_cell.attrs["NX_class"] = 'NXenvironment'
        grp_sample_cell.create_dataset('description', data='NUrF sample cell')
        grp_sample_cell.create_dataset('type', data='SQ1-ALL')

        grp_pumps = hf.create_group("/entry/sample/HPLC_pump")
        grp_pumps.attrs["NX_class"] = 'NXenvironment'
        grp_pumps.create_dataset("description", data='HPLC_pump')

        #no more valves
        #grp_valves = grp_nurf.create_group("Valves")
        #grp_valves.attrs["NX_class"] = 'NXenvironment'
        #grp_valves.create_dataset("description", data='Valves')

        grp_densito = hf.create_group("/entry/instrument/Densitometer")
        grp_densito.attrs["NX_class"] = 'NXdetector'
        grp_densito.create_dataset("description", data='Densitometer')