# -*- coding: utf-8 -*-
"""
"""
from enum import unique
from imagen.image import ImageSampler
import matplotlib
from numpy import sign
from param import Filename

matplotlib.use("Agg")
import pandas as pd
from analysis_and_visualization import perform_analysis_and_visualization
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys


import os
import psutil
import sys
import mozaik
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.vision import *

from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from visualization_functions import *

import numpy as np

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from scipy import signal
from scipy import stats
import imagen

import pickle


def export_stim_and_resp_from_natural_images_stimuli(
    datastore, sheets_delay_dictionary=None, time_window=None, output_file=None
):
    def reconstruct_stimulus(stim):
        image = imagen.image.FileImage(
            filename=stim.image_location,
            x=stim.location_x,
            y=stim.location_y,
            xdensity=stim.density,
            ydensity=stim.density,
            size=stim.size,
            bounds=imagen.image.BoundingBox(
                points=(
                    (-stim.size_x / 2, -stim.size_y / 2),
                    (stim.size_x / 2, stim.size_y / 2),
                )
            ),
            scale=2 * stim.background_luminance,
            pattern_sampler=stim.pattern_sampler,
        )()
        return image

    if sheets_delay_dictionary == None:
        sheets_delay_dictionary = dict.fromkeys(
            datastore.sheets(), 0
        )  # consider all sheets without any delay
    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())
    trials = list(
        set(
            [
                MozaikParametrized.idd(s).trial
                for s in queries.param_filter_query(
                    datastore, st_name="NaturalImage"
                ).get_stimuli()
            ]
        )
    )
    img_paths = list(
        set(
            [
                MozaikParametrized.idd(s).image_location
                for s in queries.param_filter_query(
                    datastore, st_name="NaturalImage"
                ).get_stimuli()
            ]
        )
    )
    for trial in trials:
        stim_list = []
        resp_list = []
        for img_path in img_paths:
            resp_list_per_sheet = []
            for sheet in sheets_delay_dictionary.keys():
                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="NaturalImage",
                    st_images_locations_file=img_path,
                    st_trial=trial,
                )
                PSTHLowRAM(dsv, ParameterSet({"bin_length": 1.0})).perform_analysis()
                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="NaturalImage",
                    st_images_locations_file=img_path,
                    st_trial=trial,
                )
                signals, stimuli = zip(
                    *[
                        (asl.asl, MozaikParametrized.idd(asl.stimulus_id))
                        for asl in dsv.get_analysis_result()
                    ]
                )
                # reconstruct stimulus
                stimulus = stimuli[0]
                if sheet == list(sheets_delay_dictionary.keys())[0]:
                    stim = reconstruct_stimulus(stimulus)

                # compute response
                delay = sheets_delay_dictionary[sheet]
                if time_window == None:
                    time_window = stimulus.image_duration
                signal = signals[0]
                signal_arr = np.array(
                    signal.duration.rescale(munits.sp / qt.ms).magnitude
                )
                signal_arr = np.sum(
                    signal_arr[:, int(delay) : int(delay + time_window)], axis=-1
                )
                signal_arr = np.swapaxes(signal_arr, 0, 1)
                resp_list_per_sheet.append(signal_arr)

            stim_list.append(stimulus)
            resp_list.append(np.concatenate(resp_list_per_sheet, axis=1))

    stim_resp_dict = {
        "stim": np.concatenate(stim_list),
        "resp": np.concatenate(resp_list),
    }
    if output_file == None:
        output_file = "StimRespExport.pickle"
    with open(os.path.join(Global.root_directory, output_file), "wb") as f:
        pickle.dump(stim_resp_dict, f)
    return stim_resp_dict


def ExportStimResp(datastore, sheets_delay_dictionary=None, time_window=None):
    "export to pickle file"

    def reconstruct_sensory_stimulus(stim):
        images = []
        with open(stim.images_locations_file, "rb") as f:
            img_paths = pickle.load(f)
        pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_longest",
            whole_pattern_output_fns=[
                mozaik.stimuli.vision.topographica_based.MaximumDynamicRange()
            ],
        )
        for img_path in img_paths:
            img = imagen.image.FileImage(
                filename=img_path,
                x=stim.location_x,
                y=stim.location_y,
                xdensity=stim.density,
                ydensity=stim.density,
                size=stim.size,
                bounds=imagen.image.BoundingBox(
                    points=(
                        (-stim.size_x / 2, -stim.size_y / 2),
                        (stim.size_x / 2, stim.size_y / 2),
                    )
                ),
                scale=2 * stim.background_luminance,
                pattern_sampler=pattern_sampler,
            )()
            images.append(img)
        return np.array(images)

    if sheets_delay_dictionary == None:
        # consider all sheets without any delay
        sheets_delay_dictionary = dict.fromkeys(datastore.sheets(), 0)

    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())

    trials = list(
        set(
            [
                MozaikParametrized.idd(s).trial
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )

    imgfiles = list(
        set(
            [
                MozaikParametrized.idd(s).images_locations_file
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )

    for trial in trials:
        stim_list = []
        resp_list = []
        for sheet in sheets_delay_dictionary.keys():
            print("running psth on " + sheet + " on trial" + str(trial))

            stim_per_sheet = []
            resp_per_sheet = []

            for imgfile in imgfiles:

                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="ImagesSequence",
                    st_images_locations_file=imgfile,
                    st_trial=trial,
                )

                PSTHLowRAM(dsv, ParameterSet({"bin_length": 1.0})).perform_analysis()

                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="ImagesSequence",
                    st_images_locations_file=imgfile,
                    st_trial=trial,
                )

                signals, stimuli = zip(
                    *[
                        (asl.asl, MozaikParametrized.idd(asl.stimulus_id))
                        for asl in dsv.get_analysis_result()
                    ]
                )

                # check
                assert all(
                    stim.time_per_image == stimuli[0].time_per_image for stim in stimuli
                ), "time per image must be the same across all stimuli"
                assert all(
                    stim.time_per_blank == stimuli[0].time_per_blank for stim in stimuli
                ), "time per blank must be the same across all stimuli"
                time_per_image = stimuli[0].time_per_image
                # time_per_blank = stimuli[0].time_per_blank

                signals_arr = np.array(
                    [
                        np.array(
                            [
                                np.reshape(
                                    s.rescale(munits.sp / qt.ms).magnitude,
                                    (
                                        int(
                                            s.duration.rescale(qt.ms)
                                            / (
                                                (
                                                    stim.time_per_image
                                                    + stim.time_per_blank
                                                )
                                                * qt.ms
                                            )
                                        ),
                                        -1,
                                    ),
                                )
                                for s in sig
                            ]
                        )
                        for sig, stim in zip(signals, stimuli)
                    ]
                )
                signals_arr = np.swapaxes(signals_arr, 0, 1)
                signals_arr = [np.concatenate(s, axis=0) for s in signals_arr]
                signals_arr = np.swapaxes(signals_arr, 0, 1)
                signals_arr = np.swapaxes(signals_arr, 1, 2)

                # slice according to sheet delay and time window and then sum to get num of spikes
                delay = sheets_delay_dictionary[sheet]

                if time_window == None:
                    time_window = time_per_image
                signals_arr = numpy.sum(
                    signals_arr[:, int(delay) : int(delay + time_window), :], axis=1
                )

                # reconstruct the sensory stimulus ()
                sensory_stim = np.array(
                    [reconstruct_sensory_stimulus(stim) for stim in stimuli]
                )
                print("sensory_stim shape ", sensory_stim.shape)
                # is this line necessary?
                sensory_stim = sensory_stim.reshape(
                    (-1, sensory_stim.shape[-2], sensory_stim.shape[-1])
                )
                print("sensory_stim shape ", sensory_stim.shape)

                dsv.remove_ads_from_datastore()

                stim_per_sheet.append(sensory_stim)
                resp_per_sheet.append(signals_arr)

            print(np.concatenate(stim_per_sheet).shape)
            print(np.concatenate(resp_per_sheet).shape)
            stim_list.append(np.concatenate(stim_per_sheet))
            resp_list.append(np.concatenate(resp_per_sheet))

        print("_list len: ", len(stim_list))
        print("stim_list element", stim_list[0].shape)

        unique_stim = np.unique(np.concatenate(stim_list), axis=0)
        print("unique stim shape: ", unique_stim.shape)
        resp_ = []

        for st in unique_stim:
            idx = [
                i for i, s in enumerate(np.concatenate(stim_list)) if (s == st).all()
            ]
            print(idx)
            idx -= np.arange(0, len(idx)) * len(resp_list[0])
            print(idx)

            resp_.append(np.concatenate([resp_list[i][j] for i, j in enumerate(idx)]))
        resp_final = np.array(resp_)

        if trial == 0:
            stim_resp_dict = {
                "stim": unique_stim,
                "resp": resp_final,
            }
        else:
            stim_resp_dict["stim"] = np.concatenate(
                (stim_resp_dict["stim"], unique_stim), axis=0
            )
            stim_resp_dict["resp"] = np.concatenate(
                (stim_resp_dict["resp"], resp_final), axis=0
            )

    with open(os.path.join(Global.root_directory, "StimRespExport.pickle"), "wb") as f:
        pickle.dump(stim_resp_dict, f)
    return stim_resp_dict


def ExportStimResp2(
    datastore, sheets_delay_dictionary=None, time_window=None, output_file=None
):
    "export to pickle file responses"

    def reconstruct_sensory_stimulus(stim):
        images = []
        with open(stim.images_locations_file, "rb") as f:
            img_paths = pickle.load(f)
        pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_longest",
            whole_pattern_output_fns=[
                mozaik.stimuli.vision.topographica_based.MaximumDynamicRange()
            ],
        )
        for img_path in img_paths:
            img = imagen.image.FileImage(
                filename=img_path,
                x=stim.location_x,
                y=stim.location_y,
                xdensity=stim.density,
                ydensity=stim.density,
                size=stim.size,
                bounds=imagen.image.BoundingBox(
                    points=(
                        (-stim.size_x / 2, -stim.size_y / 2),
                        (stim.size_x / 2, stim.size_y / 2),
                    )
                ),
                scale=2 * stim.background_luminance,
                pattern_sampler=pattern_sampler,
            )()
            images.append(img)
        return np.array(images)

    if sheets_delay_dictionary == None:
        sheets_delay_dictionary = dict.fromkeys(datastore.sheets(), 0)
    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())

    trials = list(
        set(
            [
                MozaikParametrized.idd(s).trial
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )
    imgfiles = list(
        set(
            [
                MozaikParametrized.idd(s).images_locations_file
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )
    
    stim_list = []
    resp_list = []

    for trial in trials:
        print(trial)
        for imgfile in imgfiles:
        
            print("running psth on " + imgfile, flush=True)
            resp_imgfile_list = []

            for sheet in sheets_delay_dictionary.keys():

                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="ImagesSequence",
                    st_images_locations_file=imgfile,
                    st_trial=trial,
                )
                PSTHLowRAM(dsv, ParameterSet({"bin_length": 1.0})).perform_analysis()
                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="ImagesSequence",
                    st_images_locations_file=imgfile,
                    st_trial=trial,
                )
                print(dsv.get_analysis_result())
                signals, stimuli = zip(
                    *[
                        (asl.asl, MozaikParametrized.idd(asl.stimulus_id))
                        for asl in dsv.get_analysis_result()
                    ]
                )

                # reconstruct the sensory stimulus()
                if sheet == list(sheets_delay_dictionary.keys())[0]:
                    sensory_stim = np.array(
                        [reconstruct_sensory_stimulus(stim) for stim in stimuli]
                    ).squeeze()

                # slice psth results in dividing per image
                delay = sheets_delay_dictionary[sheet]
                if time_window == None:
                    time_window = stimuli[0].time_per_image

                signals_arr = np.array(
                    [
                        np.array(
                            [
                                np.reshape(
                                    s.rescale(munits.sp / qt.ms).magnitude,
                                    (
                                        int(
                                            s.duration.rescale(qt.ms)
                                            / (
                                                (
                                                    stim.time_per_image
                                                    + stim.time_per_blank
                                                )
                                                * qt.ms
                                            )
                                        ),
                                        -1,
                                    ),
                                )
                                for s in sig
                            ]
                        )
                        for sig, stim in zip(signals, stimuli)
                    ]
                ).squeeze()
                # slice according to sheet delay and time window and then sum to get num of spikes
                signals_arr = np.sum(
                    signals_arr[:, :, int(delay) : int(delay + time_window)], axis=-1
                )
                signals_arr = np.swapaxes(signals_arr, 0, 1)
                resp_imgfile_list.append(signals_arr)
                dsv.remove_ads_from_datastore()
            stim_list.append(sensory_stim)
            resp_list.append(np.concatenate(resp_imgfile_list, axis=1))

    stim_resp_dict = {
        "stim": np.concatenate(stim_list),
        "resp": np.concatenate(resp_list),
    }
    if output_file == None:
        output_file = "StimRespExport.pickle"
    with open(os.path.join(Global.root_directory, output_file), "wb") as f:
        pickle.dump(stim_resp_dict, f)

    return stim_resp_dict

def ExportStimResp3(
    datastore, sheets_delay_dictionary=None, time_window=None, output_file=None
):
    "export to pickle file responses"

    def reconstruct_sensory_stimulus(stim):
        images = []
        with open(stim.images_locations_file, "rb") as f:
            img_paths = pickle.load(f)
        pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_longest",
            whole_pattern_output_fns=[
                mozaik.stimuli.vision.topographica_based.MaximumDynamicRange()
            ],
        )
        for img_path in img_paths:
            img = imagen.image.FileImage(
                filename=img_path,
                x=stim.location_x,
                y=stim.location_y,
                xdensity=stim.density,
                ydensity=stim.density,
                size=stim.size,
                bounds=imagen.image.BoundingBox(
                    points=(
                        (-stim.size_x / 2, -stim.size_y / 2),
                        (stim.size_x / 2, stim.size_y / 2),
                    )
                ),
                scale=2 * stim.background_luminance,
                pattern_sampler=pattern_sampler,
            )()
            images.append(img)
        return np.array(images)

    if sheets_delay_dictionary == None:
        sheets_delay_dictionary = dict.fromkeys(datastore.sheets(), 0)
    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())

    imgfiles = list(
        set(
            [
                MozaikParametrized.idd(s).images_locations_file
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )
    
    stim_list = []
    resp_list = []

    for imgfile in imgfiles:
    
        print("running psth on " + imgfile, flush=True)
        resp_imgfile_list = []

        for sheet in sheets_delay_dictionary.keys():

            dsv = queries.param_filter_query(
                datastore,
                sheet_name=sheet,
                st_name="ImagesSequence",
                st_images_locations_file=imgfile,
            )
            PSTHLowRAM(dsv, ParameterSet({"bin_length": 1.0})).perform_analysis()
            dsv = queries.param_filter_query(
                datastore,
                sheet_name=sheet,
                st_name="ImagesSequence",
                st_images_locations_file=imgfile,
            )
            print(dsv.get_analysis_result())
            signals, stimuli = zip(
                *[
                    (asl.asl, MozaikParametrized.idd(asl.stimulus_id))
                    for asl in dsv.get_analysis_result()
                ]
            )

            # reconstruct the sensory stimulus()
            if sheet == list(sheets_delay_dictionary.keys())[0]:
                sensory_stim = np.array(
                    [reconstruct_sensory_stimulus(stim) for stim in stimuli]
                ).squeeze()

            # slice psth results in dividing per image
            delay = sheets_delay_dictionary[sheet]
            if time_window == None:
                time_window = stimuli[0].time_per_image

            signals_arr = np.array(
                [
                    np.array(
                        [
                            np.reshape(
                                s.rescale(munits.sp / qt.ms).magnitude,
                                (
                                    int(
                                        s.duration.rescale(qt.ms)
                                        / (
                                            (
                                                stim.time_per_image
                                                + stim.time_per_blank
                                            )
                                            * qt.ms
                                        )
                                    ),
                                    -1,
                                ),
                            )
                            for s in sig
                        ]
                    )
                    for sig, stim in zip(signals, stimuli)
                ]
            ).squeeze()
            # slice according to sheet delay and time window and then sum to get num of spikes
            signals_arr = np.sum(
                signals_arr[:, :, int(delay) : int(delay + time_window)], axis=-1
            )
            signals_arr = np.swapaxes(signals_arr, 0, 1)
            resp_imgfile_list.append(signals_arr)
            dsv.remove_ads_from_datastore()
        stim_list.append(sensory_stim)
        resp_list.append(np.concatenate(resp_imgfile_list, axis=1))

    stim_resp_dict = {
        "stim": np.concatenate(stim_list),
        "resp": np.concatenate(resp_list),
    }
    if output_file == None:
        output_file = "StimRespExport.pickle"
    with open(os.path.join(Global.root_directory, output_file), "wb") as f:
        pickle.dump(stim_resp_dict, f)

    return stim_resp_dict



def ExportStimResp4(
    datastore, sheets_delay_dictionary=None, time_window=None, output_file=None
):
    "export to pickle file responses"

    def reconstruct_sensory_stimulus(stim):
        images = []
        with open(stim.images_locations_file, "rb") as f:
            img_paths = pickle.load(f)
        pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_longest",
            whole_pattern_output_fns=[
                mozaik.stimuli.vision.topographica_based.MaximumDynamicRange()
            ],
        )
        for img_path in img_paths:
            img = imagen.image.FileImage(
                filename=img_path,
                x=stim.location_x,
                y=stim.location_y,
                xdensity=stim.density,
                ydensity=stim.density,
                size=stim.size,
                bounds=imagen.image.BoundingBox(
                    points=(
                        (-stim.size_x / 2, -stim.size_y / 2),
                        (stim.size_x / 2, stim.size_y / 2),
                    )
                ),
                scale=2 * stim.background_luminance,
                pattern_sampler=pattern_sampler,
            )()
            images.append(img)
        return np.array(images)

    if sheets_delay_dictionary == None:
        sheets_delay_dictionary = dict.fromkeys(datastore.sheets(), 0)
    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())

    imgfiles = list(
        set(
            [
                MozaikParametrized.idd(s).images_locations_file
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )
    
    stim_list = []
    resp_list = []

    for imgfile in imgfiles:
    
        print("running psth on " + imgfile, flush=True)
        resp_imgfile_list = []

        for sheet in sheets_delay_dictionary.keys():
            
            dsv = queries.param_filter_query(
                datastore,
                sheet_name=sheet,
                st_name="ImagesSequence",
                st_images_locations_file=imgfile,
            )
            if dsv.get_analysis_results() == []:
                PSTHLowRAM(dsv, ParameterSet({"bin_length": 1.0})).perform_analysis()
                dsv = queries.param_filter_query(
                    datastore,
                    sheet_name=sheet,
                    st_name="ImagesSequence",
                    st_images_locations_file=imgfile,
                )
                print(dsv.get_analysis_result())
            signals, stimuli = zip(
                *[
                    (asl.asl, MozaikParametrized.idd(asl.stimulus_id))
                    for asl in dsv.get_analysis_result()
                ]
            )

            # reconstruct the sensory stimulus()
            if sheet == list(sheets_delay_dictionary.keys())[0]:
                sensory_stim = np.array(
                    [reconstruct_sensory_stimulus(stim) for stim in stimuli]
                ).squeeze()

            # slice psth results in dividing per image
            delay = sheets_delay_dictionary[sheet]
            if time_window == None:
                time_window = stimuli[0].time_per_image

            signals_arr = np.array(
                [
                    np.array(
                        [
                            np.reshape(
                                s.rescale(munits.sp / qt.ms).magnitude,
                                (
                                    int(
                                        s.duration.rescale(qt.ms)
                                        / (
                                            (
                                                stim.time_per_image
                                                + stim.time_per_blank
                                            )
                                            * qt.ms
                                        )
                                    ),
                                    -1,
                                ),
                            )
                            for s in sig
                        ]
                    )
                    for sig, stim in zip(signals, stimuli)
                ]
            ).squeeze()
            # slice according to sheet delay and time window and then sum to get num of spikes
            signals_arr = np.sum(
                signals_arr[:, :, int(delay) : int(delay + time_window)], axis=-1
            )
            signals_arr = np.swapaxes(signals_arr, 0, 1)
            resp_imgfile_list.append(signals_arr)
            dsv.remove_ads_from_datastore()
        stim_list.append(sensory_stim)
        resp_list.append(np.concatenate(resp_imgfile_list, axis=1))

    stim_resp_dict = {
        "stim": np.concatenate(stim_list),
        "resp": np.concatenate(resp_list),
    }
    if output_file == None:
        output_file = "StimRespExport.pickle"
    with open(os.path.join(Global.root_directory, output_file), "wb") as f:
        pickle.dump(stim_resp_dict, f)

    return stim_resp_dict


def Export2Pandas(datastore, time_window, sheets_delay_dictionary=None):
    def reconstruct_sensory_stimuli_list(stim):
        images = []
        with open(stim.images_locations_file, "rb") as f:
            img_paths = pickle.load(f)
        pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_longest",
            whole_pattern_output_fns=[
                mozaik.stimuli.vision.topographica_based.MaximumDynamicRange()
            ],
        )
        for img_path in img_paths:
            img = imagen.image.FileImage(
                filename=img_path,
                x=stim.location_x,
                y=stim.location_y,
                xdensity=stim.density,
                ydensity=stim.density,
                size=stim.size,
                bounds=imagen.image.BoundingBox(
                    points=(
                        (-stim.size_x / 2, -stim.size_y / 2),
                        (stim.size_x / 2, stim.size_y / 2),
                    )
                ),
                scale=2 * stim.background_luminance,
                pattern_sampler=pattern_sampler,
            )()
            images.append(img)
        return images

    keys = ["stim", *sheets_delay_dictionary.keys()]
    dict_stim_resp = {key: [] for key in keys}

    if sheets_delay_dictionary == None:
        sheets_delay_dictionary = dict.fromkeys(datastore.sheets(), 0)
    assert all(sheet in datastore.sheets() for sheet in sheets_delay_dictionary.keys())

    imgfiles = list(
        set(
            [
                MozaikParametrized.idd(s).images_locations_file
                for s in queries.param_filter_query(
                    datastore, st_name="ImagesSequence"
                ).get_stimuli()
            ]
        )
    )

    for imgfile in tqdm(imgfiles):
        for i, sheet in enumerate(sheets_delay_dictionary().keys()):
            delay = sheets_delay_dictionary[sheet]
            dsv = queries.param_filter_query(
                datastore,
                sheet_name=sheet,
                st_name="ImagesSequence",
                st_images_locations_file=imgfile,
            )
            seg = dsv.get_segment()[0]
            stim = MozaikParametrized.idd(dsv.get_stimuli()[0])

            images_locations_file = stim.images_locations_file
            with open(images_locations_file, "rb") as f:
                images_paths = pickle.load(f)
            resp_list = []
            for i, _ in enumerate(images_paths):
                start = delay + i * (stim.time_per_image + stim.time_per_blank)
                stop = start + time_window
                resp = seg.mean_rates(start, stop) * time_window.rescale(qt.s)
                resp_list.append(np.array(resp).round())
            dict_stim_resp[sheet] += resp

        dict_stim_resp["stim"] += reconstruct_sensory_stimuli_list(stim)
