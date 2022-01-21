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
        grp_uv = hf.create_group("/entry/instrument/uv")
        grp_uv.attrs["NX_class"] = 'NXdetector_group'
        
        # subgroup for uv_spectra
        uv_spectra=grp_uv.create_group("uv_spectra")
        uv_spectra.attrs["NX_class"] = 'NXdetector'
        uv_spectra_data=uv_spectra.create_dataset('data', data=data['UV_spectra'],
                                           shape=data[
                                               'UV_spectra'].shape,
                                           dtype=np.int32)
        uv_spectra_data.attrs['long_name']= 'uv_spectra'
        uv_spectra_data.attrs['units']= 'a.u.'
        
        # define type of detector 
        string_dt = h5py.special_dtype(vlen=str) # variable-length string 
        uv_spectra_type=uv_spectra.create_dataset('type', data='ccd', dtype=string_dt)
        

    
        # subgroup for uv_integration time
        uv_inttime=grp_uv.create_group("uv_integration_time")
        uv_inttime.attrs["NX_class"] = 'NXdetector'
    
        uv_inttime_data=uv_inttime.create_dataset('uv_integration_time',
                                           data=data['UV_IntegrationTime'],
                                           shape=data[
                                               'UV_IntegrationTime'].shape,
                                           dtype=np.int32)
        uv_inttime_data.attrs['long_name'] = 'uv_integration_time'
        uv_inttime_data.attrs['units'] = 's'  # TODO: unit to be verified

        # subgroup for uv_background
        uv_bkg=grp_uv.create_group("uv_background")
        uv_bkg.attrs["NX_class"] = 'NXdetector'

        uv_bkg_data = uv_bkg.create_dataset('data',
                                       data=data['UV_background'],
                                       shape=data['UV_background'].shape,
                                       dtype=np.float32)
        uv_bkg_data.attrs['long name'] = 'uv_background'
        uv_bkg_data.attrs['data'] = 'a.u.'

        # subgroup for uv_intensity0
        uv_int0=grp_uv.create_group("uv_intensity0")
        uv_int0.attrs["NX_class"] = 'NXdetector'
        uv_int0_data = uv_int0.create_dataset('data', data=data['UV_intensity0'], 
                                            shape=data['UV_intensity0'].shape, 
                                            dtype=np.float32)
        uv_int0_data.attrs['long name'] = 'uv_intensity0'
        uv_int0_data.attrs['units'] = 'a.u.'

        
        # subgroup for uv_wavelength
        uv_wavelength=grp_uv.create_group("uv_wavelength")
        uv_wavelength.attrs["NX_class"] = 'NXdetector'
        
        uv_wavelength_data = uv_wavelength.create_dataset('data',
                                              data=data['UV_wavelength'],
                                              shape=data['UV_wavelength'].shape,
                                              dtype=np.float32)
        uv_wavelength_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        uv_wavelength_data.attrs['long name'] = 'uv_wavelength'

        # Fluorescence subgroup
        grp_fluo = hf.create_group("/entry/instrument/fluorescence")
        grp_fluo.attrs["NX_class"] = 'NXdetector_group'
        
        
        # subgroup for fluo_integration_time
        fluo_inttime=grp_fluo.create_group("fluo_integration_time")
        fluo_inttime.attrs["NX_class"] = 'NXdetector'
        
        fluo_inttime_data = fluo_inttime.create_dataset('fluo_integration_time',
                                               data=data['Fluo_IntegrationTime'],
                                               shape=data['Fluo_IntegrationTime'].shape,
                                               dtype=np.int32)
        fluo_inttime_data.attrs['units'] = 's'  # TODO: unit to be verified
        fluo_inttime_data.attrs['long name'] = 'fluo_integration_time'


        # subgroup for fluo_bkg
        fluo_bkg=grp_fluo.create_group("fluo_background")
        fluo_bkg.attrs["NX_class"] = 'NXdetector'
        fluo_bkg_data = fluo_bkg.create_dataset('fluo_background',
                                           data=data['Fluo_background'],
                                           shape=data['Fluo_background'].shape,
                                           dtype=np.float32)
        fluo_bkg_data.attrs['long name'] = 'fluo_background'
        fluo_bkg_data.attrs['unit'] = 'a.u.'

        # subgroup for fluo_intensity0
        fluo_int0=grp_fluo.create_group("fluo_intensity0")
        fluo_int0.attrs["NX_class"] = 'NXdetector'
        fluo_int0_data = fluo_int0.create_dataset('fluo_intensity0',
                                            data=data['Fluo_intensity0'],
                                            shape=data['Fluo_intensity0'].shape,
                                            dtype=np.float32)
        fluo_int0_data.attrs['long name'] = 'fluo_intensity0'
        fluo_int0_data.attrs['units'] = 'a.u.'


        # subgroup for fluo_monowavelengths
        fluo_monowavelengths=grp_fluo.create_group("fluo_monowavelengths")
        fluo_monowavelengths.attrs["NX_class"] = 'NXdetector'
        fluo_monowavelengths_data = fluo_monowavelengths.create_dataset('fluo_monowavelengths',
                                                       data=data[
                                                           'Fluo_monowavelengths'],
                                                       shape=data[
                                                           'Fluo_monowavelengths'].shape,
                                                       dtype=np.float32)
        fluo_monowavelengths_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_monowavelengths_data.attrs['long name'] = 'fluo_monowavelengths'

        # subgroup for fluo_spectra
        fluo_spectra=grp_fluo.create_group("fluo_spectra")
        fluo_spectra.attrs["NX_class"] = 'NXdetector'
        
        # define type of detector 
        fluo_spectra_type=fluo_spectra.create_dataset('type', data='ccd', dtype=string_dt)
        
        fluo_spectra_data = fluo_spectra.create_dataset('fluo_spectra',
                                               data=data['Fluo_spectra'],
                                               shape=data['Fluo_spectra'].shape,
                                               dtype=np.float32)
        fluo_spectra_data.attrs['long name'] = 'fluo_spectra'
        fluo_spectra_data.attrs['units'] = 'a.u.'

        # subgroup for fluo_wavelength
        fluo_wavelength=grp_fluo.create_group("fluo_wavelength")
        fluo_wavelength.attrs["NX_class"] = 'NXdetector'
        fluo_wavelength_data= fluo_wavelength.create_dataset('fluo_wavelength',
                                                  data=data['Fluo_wavelength'],
                                                  shape=data[
                                                      'Fluo_wavelength'].shape,
                                                  dtype=np.float32)
        fluo_wavelength_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_wavelength_data.attrs['long name'] = 'fluo_wavelength'

        # dummy groups, no information currently available
        #grp_sample_cell = hf.create_group("/entry/sample/sample_cell")
        #grp_sample_cell.attrs["NX_class"] = 'NXenvironment'
        #grp_sample_cell.create_dataset('description', data='NUrF sample cell')
        #grp_sample_cell.create_dataset('type', data='SQ1-ALL')

        #grp_pumps = hf.create_group("/entry/sample/hplc_pump")
        #grp_pumps.attrs["NX_class"] = 'NXenvironment'
        #grp_pumps.create_dataset("description", data='HPLC_pump')

        #no more valves
        #grp_valves = grp_nurf.create_group("Valves")
        #grp_valves.attrs["NX_class"] = 'NXenvironment'
        #grp_valves.create_dataset("description", data='Valves')

        #grp_densito = hf.create_group("/entry/instrument/densitometer")
        #grp_densito.attrs["NX_class"] = 'NXdetector'
        #grp_densito.create_dataset("description", data='Densitometer')