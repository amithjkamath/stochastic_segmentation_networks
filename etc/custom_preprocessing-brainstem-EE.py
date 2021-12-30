import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np


def get_brain_mask(t1):
    brain_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(t1) > 10).astype(np.uint8))
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(channel, brain_mask, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
    low, high = np.percentile(channel[brain_mask.astype(np.bool)], cutoff_percentiles)
    norm_mask = np.logical_and(brain_mask, np.logical_and(channel > low, channel < high))
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 0] = 0  # Background
    array[array == 1] = 1  # Brainstem
    array[array == 2] = 0  # Make 2 empty to ignore it.
    
    array[array >= 2] = 0  # Make all others empty.
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


def preprocess(input_dir, output_dir):

    output_dataframe = pd.DataFrame()
    for subdir_1 in os.listdir(os.path.join(input_dir)):
        id_ = subdir_1 + '/' + subdir_1
        print(id_)
        s_add = '_reg_resampled_final'

        seg = fix_segmentation_labels(sitk.ReadImage(os.path.join(input_dir, id_) + '_labelmask_EE_final.nii.gz'))
        output_path = os.path.join(output_dir, id_) + f'_seg.nii.gz'
        output_dataframe.loc[id_, 'seg'] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(seg, output_path)

        t1 = sitk.ReadImage(os.path.join(input_dir, id_) + '_T1w' + s_add + '.nii.gz')
        brain_mask = get_brain_mask(t1)
        output_path = os.path.join(output_dir, id_) + f'_brain_mask.nii.gz'
        output_dataframe.loc[id_, 'sampling_mask'] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(brain_mask, output_path)

        for suffix in ['FLAIR', 'T1w', 'T1c', 'T2w']:
            channel = sitk.ReadImage(os.path.join(input_dir, id_) + f'_{suffix:s}' + s_add + '.nii.gz')
            channel_array = sitk.GetArrayFromImage(channel)
            normalised_channel_array = z_score_normalisation(channel_array, sitk.GetArrayFromImage(brain_mask))
            normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
            normalised_channel.CopyInformation(channel)
            output_path = os.path.join(output_dir, id_) + f'_{suffix:s}.nii.gz'
            output_dataframe.loc[id_, suffix] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(normalised_channel, output_path)
    
    output_dataframe.index.name = 'id'
    assets_path = os.path.join(output_dir, 'assets')
    os.makedirs(assets_path, exist_ok=True)
    
    train_index = output_dataframe[:20]
    train_index.to_csv(os.path.join(assets_path, 'data_index_train.csv'))
    valid_index = output_dataframe[20:25]
    valid_index.to_csv(os.path.join(assets_path, 'data_index_valid.csv'))
    test_index = output_dataframe[25:]
    test_index.to_csv(os.path.join(assets_path, 'data_index_test.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Path to input directory.')
    parser.add_argument('--output-dir',
                        required=True,
                        type=str,
                        help='Path to output directory.')

    parse_args, unknown = parser.parse_known_args()
    input_dir  = parse_args.input_dir
    output_dir = parse_args.output_dir

    preprocess(input_dir, output_dir)
