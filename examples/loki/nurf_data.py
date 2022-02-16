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
                
        #comment on names
        #UV/FL_Background is the dark
        #UV/FL_Intensity0 is the reference
        #UV/FL_Spectra is the sample
        
        # image key for the uv_dark
        # number of frames (nFrames) given indirectly as part of the shape of the arrays 
        # TODO: keep in mind what happens if multiple dark or reference frames are taken
        
        #I need to reshape data['UV_spectra']
        data['UV_spectra']=np.reshape(data['UV_spectra'],(np.shape(data['UV_spectra'])[0],np.shape(data['UV_spectra'])[1]))
    
        # I want it this way 
        uv_all_data=np.column_stack((data['UV_spectra'].T,data['UV_background'], data['UV_intensity0']))
       
        # assemble image_key #TODO needs later to be verified with real data from hardware
        nb_spectra=np.shape(data['UV_spectra'])[0]
        uv_ik_spectra=np.zeros((1,nb_spectra))  #interperation here: 0 for sample (in comparison to projections)
      
        # find out how many nFrames each item (sample, dark, reference) has
        if data['UV_background'].ndim==1:
            nb_darks=1
        else: 
            nb_darks=np.shape(data['UV_background'])[1]  #TODO: needs to be verified with real data from Judith's setup
   
        uv_ik_dark=2* np.ones((1,nb_darks)) 
        #print(ik_dark)
        
        if data['UV_intensity0'].ndim==1:
            nb_ref=1
        else:
            nb_ref=np.shape(data['UV_intensity0'])[1]  #TODO: needs to be verified with real data from Judith's setup
        uv_ik_ref=4*np.ones((1,nb_ref))  #new image key: 4 for reference
        #print(ik_ref)
        
        # assmebling of image_key
        uv_image_key=np.column_stack((uv_ik_spectra,uv_ik_dark, uv_ik_ref))  #all_data is as well organised in columns
        
        
        # UV subgroup
        grp_uv = hf.create_group("/entry/instrument/uv")
        grp_uv.attrs["NX_class"] = 'NXdata'
        
        # subgroup for uv all data (sample, dark, reference)
        uv_signal=grp_uv.create_group("uv_all_data")
        uv_signal_data=uv_signal.create_dataset('data', data=uv_all_data,
                                           shape=uv_all_data.shape)
        uv_signal_data.attrs['long_name']= 'uv_all_data'
        uv_signal_data.attrs['units']= ''
        uv_signal_data.attrs['signal']= 'data'  #indicate that the main signal is data 
        uv_signal_data.attrs['axes']= [ "wavelength", "." ]   #see example in NXdata, time as x-axis is first entry
        uv_signal_image_key=uv_signal.create_dataset('image_key',data=uv_image_key,shape=uv_image_key.shape, dtype=np.int32)
       
     
        # uv_wavelength
        uv_wavelength_data=grp_uv.create_dataset('uv_wavelength', data=data['UV_wavelength'],
                                              shape=data['UV_wavelength'].shape,
                                              dtype=np.float32)
        
        uv_wavelength_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        uv_wavelength_data.attrs['long name'] = 'uv_wavelength'
                
                
        # uv_integration_time
        uv_inttime_data=grp_uv.create_dataset("uv_integration_time",data=data['UV_IntegrationTime'],
                                           shape=data[
                                               'UV_IntegrationTime'].shape, 
                                           dtype=np.int32)
                   
        uv_inttime_data.attrs['long_name'] = 'uv_integration_time'
        uv_inttime_data.attrs['units'] = 's'  # TODO: unit to be verified

        
        
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
        fluo_spectra.attrs["NX_class"] = 'NXdata'
        
        
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