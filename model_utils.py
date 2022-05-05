import torch
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import pickle

"""parser imports & utils"""
from parser import parse_file
from utils import (
    check_for_periph_data,
    filter_to_idxs,
    fill_gaps,
    trim_data,
    flatten_dict,
    singleify,
    smooth_arr,
)
from visualizer import (
    save_figure_to_file,
)

data_dir = "data"
results_dir = "results.model"


def load_data(filename: str) -> Tuple[dict, dict, dict]:
    if "\\" in filename or "." in filename or "/" in filename:
        data = get_model_data(filename)
    else:
        data = get_all_data(filename)
    # don't normalize time or steering/throttle/brake
    data = normalize_batch(
        data,
        exclude=[
            "TimestampCarla_data",
            "UserInputs_Steering",
            "UserInputs_Throttle",
            "UserInputs_Brake",
        ],
    )

    """get data!!!"""
    t = data["TimestampCarla_data"]

    """OUTPUT VARIABLES"""
    Y = {}
    Y["steering"] = data["UserInputs_Steering"]
    Y["throttle"] = data["UserInputs_Throttle"]
    Y["brake"] = data["UserInputs_Brake"]

    feature_names = [
        "EgoVariables_VehicleLoc_0",
        "EgoVariables_VehicleLoc_1",
        "EgoVariables_VehicleVel",
        "EgoVariables_Velocity_0",
        "EgoVariables_Velocity_1",
        "EgoVariables_AngularVelocity_1",  # yaw velocity
        "EyeTracker_LEFTGazeDir_1_s",
        "EyeTracker_LEFTGazeDir_2_s",
        "EyeTracker_RIGHTGazeDir_1_s",
        "EyeTracker_RIGHTGazeDir_2_s",
        "EyeTracker_LEFTPupilDiameter_s",
        "EyeTracker_LEFTPupilPosition_0_s",
        "EyeTracker_LEFTPupilPosition_1_s",
        "EyeTracker_RIGHTPupilDiameter_s",
        "EyeTracker_RIGHTPupilPosition_0_s",
        "EyeTracker_RIGHTPupilPosition_1_s",
        "EgoVariables_CameraLoc_0",
        "EgoVariables_CameraLoc_1",
        "EgoVariables_CameraRot_0",
        "EgoVariables_CameraRot_1",
    ]

    features: Dict[str, List[str]] = {}

    features["steering"] = feature_names + [
        "UserInputs_Throttle",  # other driving inputs
        "UserInputs_Brake",  # other driving inputs
    ]

    features["throttle"] = feature_names + [
        "UserInputs_Steering",  # other driving inputs
        "UserInputs_Brake",  # other driving inputs
    ]

    features["brake"] = feature_names + [
        "UserInputs_Steering",  # other driving inputs
        "UserInputs_Throttle",  # other driving inputs
    ]

    """INPUT VARIABLE"""
    X = {}
    model_types = ["steering", "throttle", "brake"]
    for model_type in model_types:
        X[model_type] = np.array([data[k] for k in features[model_type]]).T

    # Split sampled data into training and test
    p = 0.2  # last 20% of the data is for testing
    m = int(len(t) * (1 - p))  # percentage for training
    train_split = {
        "X": {k: X[k][:m] for k in X.keys()},
        "Y": {k: Y[k][:m] for k in Y.keys()},
    }

    test_split = {
        "X": {k: X[k][m:] for k in X.keys()},
        "Y": {k: Y[k][m:] for k in Y.keys()},
    }

    return train_split, test_split, features, t[:m], t[m:]


def get_all_data(name: str):
    print("loading cumulative data")
    all_data = []
    assert os.path.exists(data_dir)
    _, _, files = list(os.walk(data_dir))[0]
    for f in sorted(files):
        if name in f and "data" in f:
            data = try_load_data(f.replace(".data", ""))
            assert data is not None
            all_data.append(data)
    full_data = {}
    for d in all_data:
        for k in d.keys():
            if k not in full_data:
                full_data[k] = np.array([])
            data = d[k]
            if k == "TimestampCarla_data" and len(full_data[k]) > 0:
                data += full_data[k][-1]  # ensure time is monotonically incr
            full_data[k] = np.concatenate((full_data[k], data))
    return full_data


def get_model_data(filename: str) -> Dict[str, Any]:
    """try to load cached data from previous runs"""
    print(f"loading file data: {filename}")
    data = try_load_data(filename)
    if data is not None:
        return data

    """Have to read the file to create the object"""
    data = parse_file(filename)
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    """sanitize data"""
    if "CustomActor" in data:
        data.pop("CustomActor")  # not using this rn
    data = filter_to_idxs(data, mode="all")
    data["EyeTracker"]["LEFTPupilDiameter"] = fill_gaps(
        np.squeeze(data["EyeTracker"]["LEFTPupilDiameter"]),
        lambda x: x < 1,
        mode="mean",
    )
    data["EyeTracker"]["RIGHTPupilDiameter"] = fill_gaps(
        np.squeeze(data["EyeTracker"]["RIGHTPupilDiameter"]),
        lambda x: x < 1,
        mode="mean",
    )
    # remove all "validity" boolean vectors
    for key in list(data["EyeTracker"].keys()):
        if "Valid" in key:
            data["EyeTracker"].pop(key)

    # compute ego position derivatives
    t = data["TimestampCarla"]["data"] / 1000  # ms to s
    data["TimestampCarla"]["data"] = t
    delta_ts = np.diff(t)  # t is in seconds
    n: int = len(delta_ts)
    assert delta_ts.min() > 0  # should always be monotonically increasing!
    ego_displacement = np.diff(data["EgoVariables"]["VehicleLoc"], axis=0)
    ego_velocity = (ego_displacement.T / delta_ts).T
    ego_velocity = np.concatenate((np.zeros((1, 3)), ego_velocity))  # include 0 @ t=0
    ego_accel = (np.diff(ego_velocity, axis=0).T / delta_ts).T
    ego_accel = np.concatenate((np.zeros((1, 3)), ego_accel))  # include 0 @ t=0
    data["EgoVariables"]["Velocity"] = ego_velocity
    data["EgoVariables"]["Accel"] = ego_accel
    rot3D = data["EgoVariables"]["VehicleRot"]
    angular_disp = np.diff(rot3D, axis=0)
    # fix rollovers for +360
    angular_disp[
        np.squeeze(np.where(np.abs(np.diff(rot3D[:, 1], axis=0)) > 359))
    ] = 0  # TODO
    # pos_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) > 359))
    # angular_disp[pos_roll_idxs][:, 1] = -1 * (360 - angular_disp[pos_roll_idxs][:, 1])
    # neg_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) < -359))
    # angular_disp[neg_roll_idxs][:, 1] = 360 + angular_disp[neg_roll_idxs][:, 1]
    angular_vel = (angular_disp.T / delta_ts).T
    angular_vel = np.concatenate((np.zeros((1, 3)), angular_disp))  # include 0 @ t=0
    data["EgoVariables"]["AngularVelocity"] = angular_vel

    # trim data bounds
    data = trim_data(data, (50, 100))
    data = flatten_dict(data)
    data = singleify(data)  # so individual axes are accessible via _ notation

    # apply data smoothing
    smooth_amnt = 200
    data["EyeTracker_COMBINEDGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_COMBINEDGazeDir_1"], smooth_amnt
    )
    data["EyeTracker_COMBINEDGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_COMBINEDGazeDir_2"], smooth_amnt
    )
    data["EyeTracker_LEFTGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_LEFTGazeDir_1"], smooth_amnt
    )
    data["EyeTracker_LEFTGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_LEFTGazeDir_2"], smooth_amnt
    )
    data["EyeTracker_RIGHTGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_RIGHTGazeDir_1"], smooth_amnt
    )
    data["EyeTracker_RIGHTGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_RIGHTGazeDir_2"], smooth_amnt
    )
    data["EyeTracker_LEFTPupilDiameter_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilDiameter"], smooth_amnt
    )
    data["EyeTracker_LEFTPupilPosition_0_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilPosition_0"], smooth_amnt
    )
    data["EyeTracker_LEFTPupilPosition_1_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilPosition_1"], smooth_amnt
    )
    data["EyeTracker_RIGHTPupilDiameter_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilDiameter"], smooth_amnt
    )
    data["EyeTracker_RIGHTPupilPosition_0_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilPosition_0"], smooth_amnt
    )
    data["EyeTracker_RIGHTPupilPosition_1_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilPosition_1"], smooth_amnt
    )
    cache_data(data, filename)
    return data


def visualize_importance(
    model,
    feature_names,
    input_tensor,
    title="Average Feature Importances",
    axis_title="Features",
):
    print("Visualizing feature importances...")
    # Helper method to print importances and visualize distribution
    ig = IntegratedGradients(model)
    input_tensor.requires_grad_()
    attr, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attr = attr.detach().numpy()
    importances = np.mean(attr, axis=0) / np.abs(np.mean(attr))
    for i in range(len(feature_names)):
        print(f"{feature_names[i]} : {importances[i]:.3f}")
    x_pos = np.arange(len(feature_names))

    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.bar(x_pos, importances, align="center")
    plt.xticks(x_pos, feature_names, wrap=True, rotation=80)
    plt.xlabel(axis_title)
    plt.title(title)
    clean_title = title.replace(" ", "_")
    save_figure_to_file(fig, f"{clean_title}.png")


def try_load_data(filename) -> Optional[Dict[str, Any]]:
    actual_name = filename.split("/")[-1].replace(".txt", "")
    filename = f"{os.path.join(data_dir, actual_name)}.data"
    data = None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data from {filename}")
    else:
        print(f"Did not find data at {filename}")
    return data


def cache_data(data, filename):
    actual_name: str = filename.split("/")[-1].replace(".txt", "")
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{os.path.join(data_dir, actual_name)}.data"
    with open(filename, "wb") as filehandler:
        pickle.dump(data, filehandler)
    print(f"cached data to {filename}")


def normalize_batch(data, exclude: List[str]):
    for k in data.keys():
        if k in exclude:
            continue
        try:
            x = data[k]
            mu = np.mean(x)
            std = np.std(x)
            if std != 0:
                data[k] = (x - mu) / std
        except Exception as e:
            pass
    return data


def seed_everything(seed: int = 99):
    np.random.seed(seed)
    torch.manual_seed(seed)
