import mozaik.stimuli.vision.topographica_based as topo
import mozaik.experiments.vision as exp
import stimulus_visualization as viz
import dummy_model as dm
from parameters import ParameterSet


def experiment_default_parameters():
    return {
        "relative_luminance": 1,
        "central_rel_lum": 1,
        "orientation": 0,
        "phase": 0,
        "spatial_frequency": 1,
        "size": 2,
        "flash_duration": 10,
        "x": 0,
        "y": 0,
        "rotations": 2,
        "duration": 1000,
        "num_trials": 1,
        "circles": 2,
        "grid": True,
    }


def base_stim_default_parameters():
    return {
        "frame_duration": 1,
        "duration": 200,
        "trial": 1,
        "direct_stimulation_name": "simulation_name",
        "direct_stimulation_parameters": None,
    }


def visual_stim_default_parameters():
    d = base_stim_default_parameters()
    d.update(
        {
            "background_luminance": 50.0,
            "density": 10.0,
            "location_x": 0.0,
            "location_y": 0.0,
            "size_x": 11.0,
            "size_y": 11.0,
        }
    )
    return d

def NaturalImage_stim_default_parameters():
    d = visual_stim_default_parameters()
    d.update(
        {
            "size" : 20.307,
            "image_location" : "/home/luca/Hillier/images/BBC_life_of_animals/image_1001.tif",
            "duration" : 100,
            
        }
    )
    return d


def ImagesSequence_stim_default_parameters():
    d = visual_stim_default_parameters()
    number_of_images = 5
    time_per_image = 100
    time_per_blank = 100
    d.update(
        {
            "size" : 20.307,
            "images_folder" : "/home/luca/Hillier/images/BBC_life_of_animals/",
            "number_of_images" : 5,
            "time_per_image" : time_per_image,
            "time_per_blank" : time_per_blank,
            "experiment_seed" : 500,
            "duration" : number_of_images*(time_per_image+time_per_blank),
        }
    )
    return d


def IS_exp_default_parameters():
    d = {   
            "images_folder" : "/home/luca/Hillier/images/BBC_life_of_animals/",
            "num_images" : 20,
            "num_images_per_stimulus": 10,
            "num_trials" : 1,
            "num_skipped_images" : 1,
            "size" : 20.307,
            "time_per_image" : 100,
            "time_per_blank" : 100,
        }
    return d


def ContinuousGaborMovementAndJump_default_parameters():
    d = visual_stim_default_parameters()
    d.update(
        {
            "orientation": 0,
            "sigma":1,
            "phase": 0,
            "spatial_frequency": 1,
            "size": 2,
            "center_relative_luminance": 1,
            "moving_relative_luminance": 1,
            "x": 0,
            "y": 0,
            "movement_duration": 100,
            "movement_length": 3,
            "movement_angle": 0,
            "moving_gabor_orientation_radial": False,
            "center_flash_duration": 20,
        }
    )
    return d


def SparseNoise_default_parameters():
    d = visual_stim_default_parameters()
    d.update(
        {"experiment_seed": 0, "time_per_image": 20, "blank_time": 20, "grid_size": 11, "grid": True}
    )
    return d


def exp_default_parameters():
    return {"duration": 1000}


def MapSimpleGabor_default_parameters():
    d = exp_default_parameters()
    d.update(
        {
            "relative_luminance": 1.0,
            "central_rel_lum": 0.5,
            "orientation": 0,
            "phase": 0,
            "spatial_frequency": 1,
            "size": 2.0,
            "flash_duration": 20,
            "x": 0,
            "y": 0,
            "rotations": 2,
            "duration": 93,
            "num_trials": 1,
            "circles": 2,
            "grid": False,
        }
    )
    return d


def MapTwoStrokeGabor_default_parameters():
    d = exp_default_parameters()
    d.update(
        {
            "relative_luminance": 1.0,
            "central_rel_lum": 0.5,
            "orientation": 0,
            "phase": 0,
            "spatial_frequency": 1,
            "size": 2.0,
            "flash_duration": 20,
            "stroke_time": 10,
            "x": 0,
            "y": 0,
            "rotations": 6,
            "duration": 50,
            "num_trials": 1,
            "circles": 2,
            "grid": False,
        }
    )
    return d


def demo_show_frame():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    frames = viz.pop_frames(stim, 100)
    viz.show_frame(frames[0], params=params, grid=True)
    viz.show_frames(frames)

def demo_stimulus_NaturalImage():
    params = NaturalImage_stim_default_parameters()
    stim = topo.NaturalImage(**params)
    viz.show_stimulus(stim)

def demo_stimulus_NaturalImagesSequence():
    params = NaturalImagesSequence_stim_default_parameters()
    stim = topo.NaturalImagesSequence(**params)
    viz.show_stimulus(stim)

def demo_stimulus_0():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim)


def demo_stimulus_1():
    params = SparseNoise_default_parameters()
    stim = topo.SparseNoise(**params)
    viz.show_stimulus(stim)


def demo_stimulus_0_duration():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, duration=130)


def demo_stimulus_0_grid():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, grid=2)


def demo_stimulus_0_frame_delay():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, frame_delay=100)


def demo_stimulus_0_animate():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, grid=True, animate=False)


def demo_experiment_0():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = MapTwoStrokeGabor_default_parameters()
    parameters = ParameterSet(params)
    experiment = exp.MapTwoStrokeGabor(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=True)


def demo_experiment_1():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = MapSimpleGabor_default_parameters()
    parameters = ParameterSet(params)
    experiment = exp.MapSimpleGabor(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False, grid=True)


def demo_experiment_NIS():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = default_parameters_NIS_exp()
    parameters = ParameterSet(params)
    experiment = exp.MeasureNaturalImagesSequence(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False, grid=None)


def demo_experiment_ImagesSequence():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = IS_exp_default_parameters()
    parameters = ParameterSet(params)
    experiment = exp.MeasureImagesSequence(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False, grid=None)

def main():

    if True:
        # Try out show_stimulus arguments
       
       
          demo_experiment_ImagesSequence()
    #     demo_stimulus_1()
    #     demo_stimulus_0_duration()
    #     demo_stimulus_0_grid()
    #     demo_stimulus_0_frame_delay()
    #     # demo_stimulus_0_animate() # Commented out as you'd have to click through all frames
    #     demo_stimulus_NaturalImagesSequence()

    # if True:
    #     # Try out visualizing experiments
    #     demo_experiment_0()
    #     demo_experiment_1()

    # if True:
    #     # Try out the underlying show_frame function
    #     demo_show_frame()


main()
