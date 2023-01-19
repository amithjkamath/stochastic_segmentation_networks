"""
This is a preprocessing script for the QUBIQ brain-growth data set, to run SSN on it.
"""

import os
import argparse

import SimpleITK as sitk
import pandas as pd
import numpy as np


def get_sampling_mask(t1):
    brain_mask = sitk.GetImageFromArray(
        (sitk.GetArrayFromImage(t1) > 10).astype(np.uint8)
    )
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(
    channel, brain_mask, cutoff_percentiles=(5.0, 95.0), cutoff_below_mean=True
):
    low, high = np.percentile(channel[brain_mask.astype(np.bool)], cutoff_percentiles)
    norm_mask = np.logical_and(
        brain_mask, np.logical_and(channel > low, channel < high)
    )
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def preprocess(input_dir, output_dir):

    output_dataframe = pd.DataFrame()

    train_path = os.path.join(
        input_dir, "training_data_v3_QC", "brain-growth", "Training"
    )
    input_dirs = os.listdir(os.path.join(train_path))
    subdirs = sorted(
        input_dirs, key=lambda x: int(x.strip("case"))
    )  # to handle 1, 10, 11, ... 2, and return 1, 2, 3, ... 10, ...

    id = 1
    for subdir in subdirs:
        seg = sitk.ReadImage(
            os.path.join(train_path, subdir, "task01_seg_majority.nii.gz")
        )
        output_path = os.path.join(output_dir, subdir, "task01_seg.nii.gz")
        output_dataframe.loc[str(id), "seg"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(seg, output_path)

        image = sitk.ReadImage(os.path.join(train_path, subdir, "image.nii.gz"))
        brain_mask = get_sampling_mask(image)
        output_path = os.path.join(output_dir, subdir, "task01_mask.nii.gz")
        output_dataframe.loc[str(id), "sampling_mask"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(brain_mask, output_path)

        for suffix in ["MR"]:
            image = sitk.ReadImage(os.path.join(train_path, subdir, "image.nii.gz"))
            channel_array = sitk.GetArrayFromImage(image)
            normalised_channel_array = z_score_normalisation(
                channel_array, sitk.GetArrayFromImage(brain_mask)
            )
            normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
            normalised_channel.CopyInformation(image)
            output_path = os.path.join(output_dir, subdir, "image.nii.gz")
            output_dataframe.loc[str(id), suffix] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(normalised_channel, output_path)
        id += 1

    output_dataframe.index.name = "id"
    assets_path = os.path.join(output_dir, "assets")
    os.makedirs(assets_path, exist_ok=True)

    train_index = output_dataframe
    train_index.to_csv(os.path.join(assets_path, "data_index_train.csv"))

    output_dataframe = pd.DataFrame()

    valid_path = os.path.join(
        input_dir, "validation_data_qubiq2021_QC", "brain-growth", "Validation"
    )
    input_dirs = os.listdir(os.path.join(valid_path))
    subdirs = sorted(
        input_dirs, key=lambda x: int(x.strip("case"))
    )  # to handle 1, 10, 11, ... 2, and return 1, 2, 3, ... 10, ...

    id = 1
    for subdir in subdirs:
        seg = sitk.ReadImage(
            os.path.join(valid_path, subdir, "task01_seg_majority.nii.gz")
        )
        output_path = os.path.join(output_dir, subdir, "task01_seg.nii.gz")
        output_dataframe.loc[str(id), "seg"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(seg, output_path)

        image = sitk.ReadImage(os.path.join(valid_path, subdir, "image.nii.gz"))
        brain_mask = get_sampling_mask(image)
        output_path = os.path.join(valid_path, subdir, "task01_mask.nii.gz")
        output_dataframe.loc[str(id), "sampling_mask"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(brain_mask, output_path)

        for suffix in ["MR"]:
            image = sitk.ReadImage(os.path.join(valid_path, subdir, "image.nii.gz"))
            channel_array = sitk.GetArrayFromImage(image)
            normalised_channel_array = z_score_normalisation(
                channel_array, sitk.GetArrayFromImage(brain_mask)
            )
            normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
            normalised_channel.CopyInformation(image)
            output_path = os.path.join(output_dir, subdir, "image.nii.gz")
            output_dataframe.loc[str(id), suffix] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(normalised_channel, output_path)
        id += 1

    output_dataframe.index.name = "id"
    valid_index = output_dataframe
    valid_index.to_csv(os.path.join(assets_path, "data_index_valid.csv"))

    test_index = output_dataframe
    test_index.to_csv(os.path.join(assets_path, "data_index_test.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Path to input directory."
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Path to output directory."
    )

    parse_args, unknown = parser.parse_known_args()
    input_dir = parse_args.input_dir
    output_dir = parse_args.output_dir
    preprocess(input_dir, output_dir)
