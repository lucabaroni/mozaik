from mozaik.storage.queries import param_filter_query
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from mozaik.tools.distribution_parametrization import PyNNDistribution
import os
import numpy as np
from collections import OrderedDict
import h5py
import pickle
import json
from quantities import Quantity
import copy
import h5py
import numpy as np
import contextlib
import logging

def read_file(file_path):
    """
    Read data from a file, attempting different formats if the file extension is not recognized.

    Args:
    file_path (str): Path to the file.

    Returns:
    object: Data loaded from the file, or None if unsuccessful.
    """
    try:
        # First, try to read as pickle
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Data successfully loaded from {file_path} as pickle")
            return data
        except:
            pass

        # If pickle fails, try JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logging.info(f"Data successfully loaded from {file_path} as JSON")
            return data
        except:
            pass

        # If JSON fails, try plain text
        try:
            with open(file_path, 'r') as f:
                data = f.read()
            logging.info(f"Data successfully loaded from {file_path} as plain text")
            return data
        except:
            pass

        # If all attempts fail, raise an exception
        raise ValueError(f"Unable to read file: {file_path}")

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
    
    return None


def reorder_lists(object_list, list_to_order, ordering_parameters):
    """
    Reorders `object_list` (list of MozaikParametrized objects) based on the attributes 
    specified in `ordering_parameters`, and applies the same reordering to `list_to_order`.
    
    Parameters:
    - object_list: List of MozaikParametrized objects to be sorted.
    - list_to_order: List that needs to be reordered in the same order as `object_list`.
    - ordering_parameters: List of strings representing the attributes of the objects in 
                           `object_list` to sort by.
    
    Returns:
    - Tuple containing the reordered `object_list` and `list_to_order`.
    """
    def sort_by_multiple_attributes(attributes):
        """
        Returns a sorting key function that can sort by multiple attributes of a MozaikParametrized object.
        """
        def key_func(obj):
            return tuple(getattr(obj, attr) for attr in attributes)
        return key_func

    # Combine the object list and the list to order to maintain pairwise association
    combined = list(zip(object_list, list_to_order))

    # Sort the combined list based on the specified attributes of the object list
    combined_sorted = sorted(combined, key=lambda pair: sort_by_multiple_attributes(ordering_parameters)(pair[0]))

    # Unzip the sorted pairs back into the object list and the other list
    reordered_object_list, reordered_list_to_order = zip(*combined_sorted)

    # Convert to list before returning
    return list(reordered_object_list), list(reordered_list_to_order)


def get_model_info_and_parameters(base_folder, separate_modified_params=False):
        """
        Retrieves and processes model information and parameters 
        (dividing or merging default and modified parameters)
        from the given base folder.

        Parameters:
        base_folder (str): The path to the base folder containing model information and parameters.
        separate_modified_params (bool): If True, return modified parameters separately. Default is False.

        Returns:
        tuple: A tuple containing (parameters, info) if separate_modified_params is False,
               or (modified_parameters, default_parameters, info) if separate_modified_params is True.
        """
        modified_parameters_path = os.path.join(base_folder, 'modified_parameters')
        default_parameters_path = os.path.join(base_folder, 'parameters')
        info_path = os.path.join(base_folder, 'info')

        modified_parameters = eval(read_file(modified_parameters_path))
        default_parameters = eval(read_file(default_parameters_path))
        info = eval(read_file(info_path)) 

        # Remove 'experiments.' parameters and 'results_dir' from modified_parameters
        modified_parameters = {k: v for k, v in modified_parameters.items() if not k.startswith('experiments.') and k != 'results_dir'}

        # Compare remaining items in modified_parameters with default_parameters
        for key in list(modified_parameters.keys()):
            if key in default_parameters and modified_parameters[key] == default_parameters[key]:
                del modified_parameters[key]

        if separate_modified_params:
            return modified_parameters, default_parameters, info
        else:
            # Merge modified parameters into default parameters
            merged_parameters = default_parameters.copy()
            merged_parameters.update(modified_parameters)
            return merged_parameters, info

def classify_stimulus_parameters_into_constant_and_varying(stims):
    """
    Classify the parameters of a list of stimuli into constant and varying categories.

    Args:
    stims (list): A list of stimulus objects.

    Returns:
    tuple: A tuple containing two OrderedDict objects:
        - constant_params: Parameters that remain constant across all stimuli.
        - varying_params: Parameters that vary across stimuli, with sorted values.
    """
    constant_params = OrderedDict()
    varying_params = OrderedDict()

    for param in stims[0].getParams().keys():
        values = [getattr(stim, param) for stim in stims]
        if all(value == values[0] for value in values):
            constant_params[param] = values[0]
        else:
            varying_params[param] = sorted(set(values))

    # Ensure 'trial' is the first key if it exists in varying_params
    if 'trial' in varying_params:
        varying_params = OrderedDict([('trial', varying_params['trial'])] + 
                                     [(k, v) for k, v in varying_params.items() if k != 'trial'])

    return constant_params, varying_params

def export_from_datastore_to_hdf5(data_store, st_name, data_type, cut_start=None, cut_end=None):
    """
    Export data from a datastore to a HDF5 file.

    Parameters:
    base_folder (str): The path to the base folder containing the datastore and model information.
    h5py_path (str): The path to the HDF5 file where the data will be saved.
    st_name (str): The name of the stimulus to be exported.
    data_type (str): The type of data to be exported. Options are 'mean_rates', 'spiketrains', or 'segments'.
    cut_start (int): The start time for data extraction. Default is 0.
    cut_end (int): The end time for data extraction. Default is None.

    Returns:
    None
    """

    def serialize_parameters(params):
        serialized = {}
        for key, value in params.items():
            if isinstance(value, dict):
                serialized[key] = serialize_parameters(value)
            elif isinstance(value, PyNNDistribution):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized
    
    def create_hdf5_structure(hf, info, default_parameters, modified_parameters, st_name, varying_stim_params, constant_stim_params, data_type, cut_start, cut_end):
        # Add default parameters and info as metadata to the group
        hf.attrs['default_parameters'] = str(serialize_parameters(default_parameters))
        hf.attrs['info'] = str(info)
        hf.attrs['data_type'] = data_type
        hf.attrs['st_name'] = st_name

        # Create a subgroup based on modified parameters
        if modified_parameters:
            model_subgroup_name = str(modified_parameters)
        else:
            model_subgroup_name = "default"
        model_subgroup = hf.create_group(model_subgroup_name)

        # Merge modified parameters into default parameters
        merged_parameters = default_parameters.copy()
        merged_parameters.update(modified_parameters)

        # Add merged parameters as metadata to the model_subgroup
        model_subgroup.attrs['parameters'] = str(serialize_parameters(merged_parameters))
        logging.info(f"Model subgroup '{model_subgroup_name}' created with merged parameters as metadata.")

        # Create a stimuli subgroup
        stimuli_subgroup = model_subgroup.create_group(st_name)
        logging.info(f"Datasets subgroup created under 'stimuli' in '{model_subgroup_name}'.")

        # Add varying parameters as metadata to the stimuli subgroup
        stimuli_subgroup.attrs['varying_paramers'] = list(varying_stim_params.keys())
        for param_name, param_values in varying_stim_params.items():
            stimuli_subgroup.attrs[f'{param_name}'] = param_values
        stimuli_subgroup.attrs['data_dimensions'] = [len(varying_stim_params[param]) for param in varying_stim_params.keys()]

        # Add constant parameters as metadata to the stimuli subgroup
        stimuli_subgroup.attrs['constant_paramers'] = list(constant_stim_params.keys())
        for param_name, param_values in constant_stim_params.items():
            stimuli_subgroup.attrs[f'{param_name}'] = str(param_values) if param_values is not None else "None"

        # Add data related metadata to the stimuli subgroup
        stimuli_subgroup.attrs['data_type'] = data_type
        stimuli_subgroup.attrs['data_cut_start'] = str(cut_start) if cut_start is not None else "None"
        stimuli_subgroup.attrs['data_cut_end'] = str(cut_end) if cut_end is not None else "None"

        return stimuli_subgroup

    def get_segments_and_stimuli_and_constant_and_varying_parameters(data_store, sheet_name, st_name):
        # Get segments and stimuli
        dsv = param_filter_query(data_store, st_name=st_name, sheet_name=sheet_name)
        segs = dsv.get_segments(ordered=True)
        stims = [MozaikParametrized.idd(seg.annotations['stimulus']) for seg in segs]

        # Get varying parameters
        constant_stim_params, varying_stim_params = classify_stimulus_parameters_into_constant_and_varying(stims)  # alternative: params = OrderedDict((param, sorted(list(parameter_value_list(stims, param)))) for param in varying_parameters(stims))

        # Assert all possible combinations of varying stimulus parameters are present in the stimuli in the list stims and that there are no duplicates
        assert len(stims) == np.prod([len(varying_stim_params[param]) for param in varying_stim_params.keys()]), "Number of stimuli does not match the product of the number of varying parameter values"
        assert len(set([str(stim) for stim in stims])) == len(stims), "There are duplicate stimuli"
        return segs, stims, constant_stim_params, varying_stim_params

    def extract_sheet_data_and_save_to_h5py(stims, segs, varying_stim_params, data_type, stimuli_subgroup, sheet_name, cut_start, cut_end):
        # Get data to export
        logging.info(f"Extracting {data_type} data from {len(segs)} segments in sheet {sheet_name}")
        data = []
        for seg in segs:  
            if data_type == 'mean_rates':
                if cut_start is not None:
                    cut_start = Quantity(cut_start, 'ms')
                if cut_end is not None:
                    cut_end = Quantity(cut_end, 'ms')
                data.append(seg.mean_rates(start=cut_start, end=cut_end))
            elif data_type == 'spiketrains':
                data.append([spiketrain.time_slice(cut_start, cut_end).magnitude for spiketrain in seg.get_spiketrains()])
            else:
                raise ValueError("Invalid data type")
        data = np.array(data)

        # Reorder stimuli and data in tensors whose number of dimensions corresponds to the number of varying parameters
        stims_sorted, data_sorted = reorder_lists(stims, data, varying_stim_params.keys())
        # stims_tensor = np.reshape(stims_sorted, [len(varying_stim_params[param]) for param in varying_stim_params.keys()])
        params_dims = [len(varying_stim_params[param]) for param in varying_stim_params.keys()]
        
        data_tensor = np.reshape(np.array(data_sorted).flatten(), [*params_dims, -1])
  
        # Add dataset to the stimuli subgroup
        sheet_name_cleaned = sheet_name.replace('/', '')
        if data_type =='spiketrains':
            dset = stimuli_subgroup.create_dataset(sheet_name_cleaned, shape=data_tensor.shape, dtype=h5py.special_dtype(vlen=np.dtype('float')))
            dset[:] = data_tensor
        else:
            stimuli_subgroup.create_dataset(sheet_name_cleaned, data=data_tensor)

    def add_stimuli_dataset(stimuli_subgroup, stims, varying_stim_params, ds):
        # Reorder stimuli and reshape to match the dimensions of varying parameters
        reordered_stims, _ = reorder_lists(stims, [str(s) for s in stims], varying_stim_params.keys()) 
        reordered_stims = np.array(reordered_stims).reshape([len(varying_stim_params[param]) for param in varying_stim_params.keys()])

        # Identify which dimension corresponds to trial
        trial_dim = None
        for i, param in enumerate(varying_stim_params.keys()):
            if param == 'trial':
                trial_dim = i
                stimuli_subgroup.attrs['trial_dim'] = trial_dim
                break

        # Drop the trial dimension by selecting the first element along it
        if trial_dim is not None:
            reordered_stims = np.take(reordered_stims, 0, axis=trial_dim)
        
        # create index to reinsert trial dimension
        reordered_stims_flat = reordered_stims.flatten()
        reordered_stims_idx = np.arange(len(reordered_stims_flat)).reshape(reordered_stims.shape)
        reordered_stims_idx = np.expand_dims(reordered_stims_idx, axis=trial_dim).repeat(len(varying_stim_params['trial']), axis=trial_dim)

        sensory_stim = np.array(ds.get_sensory_stimulus([str(s) for s in reordered_stims_flat])).squeeze()
        stimuli_subgroup.create_dataset('stimuli', data=sensory_stim)   
        stimuli_subgroup.create_dataset('stimuli_idx', data=reordered_stims_idx)


    ## Create an HDF5 file (main function)
    base_folder = data_store.parameters['root_directory']
    with h5py.File(os.path.join(base_folder, 'exported_data.h5'), 'w') as hf:
        # Get model info and parameters
        modified_parameters, default_parameters, info = get_model_info_and_parameters(base_folder, separate_modified_params=True)
        sheets =  data_store.sheets() 
        hf.attrs['sheets'] = [sheet.replace('/', '') for sheet in sheets]
        
        # Iterate over all sheets, extract data and save to h5py
        for i, sheet_name in enumerate(sheets):
            
            # Get segments and stimuli and constant and varying parameters for the current sheet
            segs, stims, constant_stim_params, varying_stim_params = get_segments_and_stimuli_and_constant_and_varying_parameters(data_store=data_store, sheet_name=sheet_name, st_name=st_name)

            # Create HDF5 structure for the first sheet
            if i == 0:
                stimuli_subgroup = create_hdf5_structure(
                    hf, info, default_parameters, modified_parameters, st_name,
                    varying_stim_params, constant_stim_params, data_type, cut_start, cut_end
                )

            # Extract data and save to h5py
            extract_sheet_data_and_save_to_h5py(stims, segs, varying_stim_params, data_type, stimuli_subgroup, sheet_name, cut_start, cut_end)

        # Add stimuli dataset
        add_stimuli_dataset(stimuli_subgroup, stims, varying_stim_params, data_store)

    logging.info(f"HDF5 file created with default parameters, info, list of sheets as metadata, stimuli subgroup, and datasets subgroup.")


def merge_hdf5_files(file_list, output_file):
    """
    Merge multiple HDF5 files created by Mozaik into a single HDF5 file.

    This function combines data from multiple HDF5 files, assuming they were generated
    from similar Mozaik simulations with varying parameters. It checks for consistency
    across files and merges data along the dimension that differs.

    Parameters:
    -----------
    file_list : list of str
        List of paths to the input HDF5 files to be merged.
    output_file : str
        Path to the output merged HDF5 file.

    Returns:
    --------
    None
        The function doesn't return anything, but creates a new merged HDF5 file.

    Raises:
    -------
    ValueError
        If inconsistencies are found between the input files or if merge conditions are not met.

    Notes:
    ------
    - The function assumes that the input files have a specific structure created by
      Mozaik's export_to_hdf5 function.
    - It merges data only if there is exactly one varying parameter with different values across
      files. If this condition is not met, the merge operation is skipped.
    - The merged file maintains the structure of the original files, combining data
      along the dimension that varies across files.
    - Logging is used to provide information about the merging process.
    """
    with h5py.File(output_file, 'w') as f_merged:
        # Initialize variables to store common information
        default_parameters = None
        info = None
        results_dirs = []
        creation_data = []
        sheets = None
        data_type = None
        st_name = None

        # Open all input files
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(h5py.File(file, 'r')) for file in file_list]

            # Check if all files have the same attributes
            for idx, f in enumerate(files):
                current_default_params = eval(f.attrs['default_parameters'])
                current_info = eval(f.attrs['info'])
                current_sheets = f.attrs['sheets']
                current_data_type = f.attrs['data_type']
                current_st_name = f.attrs['st_name']

                results_dirs.append(current_default_params.pop('results_dir'))
                creation_data.append(current_info.pop('creation_data'))

                if idx == 0:
                    default_parameters = copy.deepcopy(current_default_params)
                    info = copy.deepcopy(current_info)
                    sheets = current_sheets
                    data_type = current_data_type
                    st_name = current_st_name
                else:
                    if current_default_params != default_parameters:
                        raise ValueError(f"File {file_list[idx]} has different default parameters")
                    if current_info != info:
                        raise ValueError(f"File {file_list[idx]} has different info")
                    if not np.array_equal(current_sheets, sheets):
                        raise ValueError(f"File {file_list[idx]} has different sheets")
                    if current_data_type != data_type:
                        raise ValueError(f"File {file_list[idx]} has different data_type")
                    if current_st_name != st_name:
                        raise ValueError(f"File {file_list[idx]} has different st_name")

            # Write attributes to merged file
            f_merged.attrs['default_parameters'] = str(default_parameters)
            f_merged.attrs['info'] = str(info)
            f_merged.attrs['source_dataset_metadata'] = str({
                'results_dirs': results_dirs,
                'creation_data': creation_data
            })
            f_merged.attrs['data_type'] = data_type
            f_merged.attrs['st_name'] = st_name
            f_merged.attrs['sheets'] = sheets

            # Process model subgroups
            for model_key in files[0].keys():
                model_subgroups = [f[model_key] for f in files]
                model_params = [eval(msg.attrs['parameters']) for msg in model_subgroups]
                for mp in model_params:
                    mp.pop('results_dir')
                if not all(mp == model_params[0] for mp in model_params):
                    raise ValueError("Model parameters differ across files")

                model_subgroup_merged = f_merged.create_group(model_key)
                model_subgroup_merged.attrs['parameters'] = str(model_params[0])

                # Process stimulus subgroups
                for stim_key in model_subgroups[0].keys():
                    stim_subgroups = [msg[stim_key] for msg in model_subgroups]
                    
                    # Check constant parameters
                    constant_params = stim_subgroups[0].attrs['constant_paramers']
                    if not all(np.array_equal(ssg.attrs['constant_paramers'], constant_params) for ssg in stim_subgroups):
                        raise ValueError("Constant parameters differ across files")
                    for constant_key in constant_params:
                        if not all(ssg.attrs[constant_key] == stim_subgroups[0].attrs[constant_key] for ssg in stim_subgroups):
                            raise ValueError(f"Constant parameter {constant_key} differs across files")

                    # Check varying parameters
                    varying_params = stim_subgroups[0].attrs['varying_paramers']
                    if not all(np.array_equal(ssg.attrs['varying_paramers'], varying_params) for ssg in stim_subgroups):
                        raise ValueError("Varying parameters differ across files")

                    different_key = None
                    different_param_dim = None
                    for varying_key_idx, varying_key in enumerate(varying_params):
                        values = [set(ssg.attrs[varying_key]) for ssg in stim_subgroups]
                        if not all(v == values[0] for v in values):
                            if different_key is not None:
                                raise ValueError("More than one varying parameter has different values across files")
                            if len(set.union(*values)) != sum(len(v) for v in values):
                                raise ValueError(f"Values for {varying_key} are not completely different across all files")
                            different_key = varying_key
                            different_param_dim = varying_key_idx

                    if different_key is None:
                        raise ValueError("No varying parameter has different values across files")

                    merged_stim_subgroup = model_subgroup_merged.create_group(stim_key)

                    # Copy attributes
                    for k in stim_subgroups[0].attrs.keys():
                        if k != different_key:
                            merged_stim_subgroup.attrs[k] = stim_subgroups[0].attrs[k]
                        else:
                            merged_stim_subgroup.attrs[k] = np.concatenate([ssg.attrs[k] for ssg in stim_subgroups])

                    # Merge datasets
                    for sheet in sheets:
                        sheet_data = np.concatenate([ssg[sheet][:] for ssg in stim_subgroups], axis=different_param_dim)
                        merged_stim_subgroup.create_dataset(sheet, data=sheet_data)

                    stimuli_idx_merged = np.concatenate([ssg['stimuli_idx'][:] + sum(len(s['stimuli'][:]) for s in stim_subgroups[:i]) for i, ssg in enumerate(stim_subgroups)], axis=different_param_dim)
                    merged_stim_subgroup.create_dataset('stimuli_idx', data=stimuli_idx_merged)

                    stimuli_merged = np.concatenate([ssg['stimuli'][:] for ssg in stim_subgroups])
                    merged_stim_subgroup.create_dataset('stimuli', data=stimuli_merged)

                logging.info(f'Merged {stim_key} in {model_key}')

    logging.info(f'Successfully merged {len(file_list)} files into {output_file}')