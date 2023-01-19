import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np


label_config = {
    "BrainStem": 1,
    "Cochlea_L": 2,
    "Cochlea_R": 3,
    "Eye_L": 4,
    "Eye_R": 5,
    "Hippocampus_L": 6,
    "Hippocampus_R": 7,
    "Lacrimal_L": 8,
    "Lacrimal_R": 9,
    "Lens_L": 10,
    "Lens_R": 11,
    "OpticChiasm": 12,
    "OpticNerve_L": 13,
    "OpticNerve_R": 14,
    "Pituitary": 15,
    "Retina_L": 16,
    "Retina_R": 17,
}


def get_brain_mask(t1):
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


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 0] = 0  # Background
    array[array == 1] = 1  # Brainstem

    # array[array == 2] = 0  # Make 2 empty for HippL
    # array[array == 3] = 0  # Make 3 empty for HippR
    # array[array == label_config["Hippocampus_L"]] = 1
    # array[array == label_config["Hippocampus_R"]] = 1

    array[array > 1] = 0  # Make all others empty.
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


def preprocess(expert, input_dir, output_dir):

    output_dataframe = pd.DataFrame()

    input_dirs = os.listdir(os.path.join(input_dir))
    subdirs = sorted(
        input_dirs, key=lambda x: int(x)
    )  # to handle 1, 10, 11, ... 2, and return 1, 2, 3, ... 10, ...
    for subdir in subdirs:
        id_ = subdir + "/" + subdir
        print(id_)
        s_add = "_reg_resampled_final"

        seg = fix_segmentation_labels(
            sitk.ReadImage(
                os.path.join(input_dir, id_) + "_labelmask_" + expert + "_final.nii.gz"
            )
        )
        output_path = os.path.join(output_dir, id_) + f"_seg.nii.gz"
        output_dataframe.loc[id_, "seg"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(seg, output_path)

        t1 = sitk.ReadImage(os.path.join(input_dir, id_) + "_T1w" + s_add + ".nii.gz")
        brain_mask = get_brain_mask(t1)
        output_path = os.path.join(output_dir, id_) + f"_brain_mask.nii.gz"
        output_dataframe.loc[id_, "sampling_mask"] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(brain_mask, output_path)

        for suffix in ["FLAIR", "T1w", "T1c", "T2w"]:
            channel = sitk.ReadImage(
                os.path.join(input_dir, id_) + f"_{suffix:s}" + s_add + ".nii.gz"
            )
            channel_array = sitk.GetArrayFromImage(channel)
            normalised_channel_array = z_score_normalisation(
                channel_array, sitk.GetArrayFromImage(brain_mask)
            )
            normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
            normalised_channel.CopyInformation(channel)
            output_path = os.path.join(output_dir, id_) + f"_{suffix:s}.nii.gz"
            output_dataframe.loc[id_, suffix] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(normalised_channel, output_path)

    output_dataframe.index.name = "id"
    assets_path = os.path.join(output_dir, "assets")
    os.makedirs(assets_path, exist_ok=True)

    train_index = output_dataframe[:18]
    train_index.to_csv(os.path.join(assets_path, "data_index_train.csv"))
    valid_index = output_dataframe[18:21]
    valid_index.to_csv(os.path.join(assets_path, "data_index_valid.csv"))
    test_index = output_dataframe[21:]
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
    expert = "EH"
    preprocess(expert, input_dir, output_dir)
