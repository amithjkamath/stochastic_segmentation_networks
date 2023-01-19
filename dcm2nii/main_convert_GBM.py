import os
import json
from time import time
import shutil
import SimpleITK as sitk
from pydicom import dcmread
from pydicom.dataset import Tag
from Conversion import (DicomRTConversionParameter, DicomImageConversionParameter, NiftiToDicomRtConverter,
                        NiftiToDicomConverter, KEY_REFERENCE_IMAGE_PATH, DEFAULT_STRUCTURE_NAME_LIST, TAG_MODALITY,
                        TAG_SOP_INSTANCE_UID, DicomRtToNiftiConverter, DicomToNiftiConverter, DicomDirectoryFilter)

from Conversion.definitions import *


def write_dcm_for_patient(patient: str, output_dir: str, input_dir: str):
    files = os.scandir(input_dir)

    for file in files:
        dataset = dcmread(file.path)
        dataset.decode()
        patient_id = str(dataset.get_item(Tag(TAG_PATIENT_ID)).value)
        if patient_id == patient:
            shutil.copy(file.path, os.path.join(output_dir, file.name))


def generate_series_dicts(series_instance_uids, file_entries):
    series_dicts = []
    for series_instance_uid in series_instance_uids:
        series_dict = {KEY_SERIES_INSTANCE_UID: series_instance_uid, PH_FILES: []}
        is_first_entry = True
        for entry in file_entries:
            if series_instance_uid == entry.get(KEY_SERIES_INSTANCE_UID):
                if is_first_entry:
                    series_dict[KEY_SERIES_DESCRIPTION] = entry.get(KEY_SERIES_DESCRIPTION)
                    series_dict[KEY_SERIES_NUMBER] = entry.get(KEY_SERIES_NUMBER)
                    is_first_entry = False
                series_dict[PH_FILES].append(entry.get(PH_FILES))
        series_dicts.append(series_dict)
    return series_dicts


def generate_conversion_parameters(dicom_base_dir: str , parameter_output_file_path: str) -> None:
    """ Generates the basic conversion parameters and save them in a JSON file.

    Args:
        dicom_base_dir:
        parameter_output_file_path:

    Examples:
        >>> dicom_base_dir = 'D:/DataBackups/2020_08_ISAS_GBM_work/'
        >>> parameter_output_file_path = 'D:/DataBackups/2020_08_ISAS_GBM_work/assignments.json'
        >>> generate_conversion_parameters(dicom_base_dir, parameter_output_file_path)

    """
    subject_paths = [entry for entry in os.scandir(dicom_base_dir) if entry.is_dir()]

    dataset_info = {}
    for i, subject_path in enumerate(subject_paths):
        start = time()
        dir_filter = DicomDirectoryFilter(subject_path.path)
        dir_info = dir_filter.get_info()
        print(f'[{i}/{len(subject_paths)}]\t{subject_path.name}\tProcessing {dir_info[0].get(KEY_PATIENT_ID)}...')

        # Build the basic structure:
        patient_dict = {KEY_PATIENT_ID: dir_info[0].get(KEY_PATIENT_ID),
                        PH_IMAGE: {},
                        PH_REGISTRATION: [],
                        PH_SEGMENTATION: []}

        # Summarize the CT-series
        ct_images = dir_filter.get_entries_by_criterion(KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_CT, dir_info)
        ct_series_instance_uids = dir_filter.get_unique_values(KEY_SERIES_INSTANCE_UID, ct_images)
        if len(ct_series_instance_uids) != 1:
            print(f'{dir_info[0].get(KEY_PATIENT_ID)}: '
                  f'No or multiple CT SeriesInstanceUIDs ({len(ct_series_instance_uids)})!')
        ct_series_dicts = generate_series_dicts(ct_series_instance_uids, ct_images)
        for j, series_dict in enumerate(ct_series_dicts):
            if len(ct_series_dicts) == 1:
                patient_dict[PH_IMAGE].update({'CT': series_dict})
            else:
                patient_dict[PH_IMAGE].update({f'CT_{j}': series_dict})

        # Build the MR-series
        mr_images = dir_filter.get_entries_by_criterion(KEY_SOP_CLASS_UID,
                                                        (DEFAULT_SOP_CLASS_UID_MR, DEFAULT_SOP_CLASS_UID_MR_ENHANCED),
                                                        dir_info)
        mr_series_instance_uids = dir_filter.get_unique_values(KEY_SERIES_INSTANCE_UID, mr_images)
        mr_series_dicts = generate_series_dicts(mr_series_instance_uids, mr_images)
        for i, series_dict in enumerate(mr_series_dicts):
            if len(mr_series_dicts) == 1:
                patient_dict[PH_IMAGE].update({'MR': series_dict})
            else:
                patient_dict[PH_IMAGE].update({f'MR_{i}': series_dict})

        # Build the registration-series
        registrations = dir_filter.get_entries_by_criterion(KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_REGISTRATION,
                                                            dir_info)
        registration_series_instance_uids = dir_filter.get_unique_values(KEY_SERIES_INSTANCE_UID, registrations)
        registration_series_dicts = generate_series_dicts(registration_series_instance_uids, registrations)
        for registration_series_dict in registration_series_dicts:
            patient_dict[PH_REGISTRATION].append(registration_series_dict)

        # Build the segmentation-series
        segmentations = dir_filter.get_entries_by_criterion(KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_RTSS)
        segmentation_series_instance_uids = dir_filter.get_unique_values(KEY_SERIES_INSTANCE_UID, segmentations)
        segmentation_series_dicts = generate_series_dicts(segmentation_series_instance_uids, segmentations)
        for segmentation_series_dict in segmentation_series_dicts:
            patient_dict[PH_SEGMENTATION].append(segmentation_series_dict)

        dataset_info.update({patient_dict.get(KEY_PATIENT_ID): patient_dict})
        print(f'Time for subject: {(time() - start):.2f}sec')

    with open(parameter_output_file_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)


if __name__ == '__main__':

    # dicom_base_dir = 'D:/DataBackups/2021_09_ISAS_METIS/'
    # parameter_output_file_path = 'D:/DataBackups/2021_09_ISAS_METIS/assignments.json'
    # generate_conversion_parameters(dicom_base_dir, parameter_output_file_path)

    # dicom_base_dir = 'D:/DataBackups/2020_08_ISAS_GBM_work/'
    # output_json = 'D:/DataBackups/2020_08_ISAS_GBM_work/assignments_unfilled.json'
    # generate_conversion_parameters(dicom_base_dir, output_json)

    # input_json = 'D:/DataBackups/2020_08_ISAS_GBM_work/assignments.json'
    # input_corr = os.path.join(os.getcwd(), 'assignments_old.txt')
    # output_json = 'D:/DataBackups/2020_08_ISAS_GBM_work/assignments_filled_full.json'
    #
    # # Read the assignments
    # with open(input_json) as f:
    #     assignments = json.load(f)
    #
    # correlation = []
    # with open(input_corr) as f:
    #     corr_read_data = f.readlines()
    #
    # corr_read_data = [entry for entry in corr_read_data if entry != '\n']
    # for corr_data in corr_read_data:
    #     # print(corr_data)
    #     main_chunks = corr_data.split(', ')
    #     first_part_chunks = main_chunks[0].split('_')
    #     second_part_chunks = main_chunks[1].split('_')
    #     patient_name = '_'.join(first_part_chunks[1:4])
    #     if patient_name == 'ISAS_METIS_004':
    #         print('STOP')
    #     series_id = int(first_part_chunks[4])
    #     series_name_export = '_'.join(first_part_chunks[5:])[:-7]
    #     modality = second_part_chunks[-1].split('.')[0]
    #     entry_data = {KEY_PATIENT_ID: patient_name,
    #                   KEY_SERIES_DESCRIPTION: series_name_export,
    #                   KEY_SERIES_NUMBER: series_id,
    #                   PH_MODALITY: modality}
    #     correlation.append(entry_data)
    #
    # # Try to assign the modalities
    # new_dict = {}
    # counter = 0
    # for patient, patient_data in assignments.items():
    #     new_image_dict = {}
    #     for modality_name, patient_image in patient_data[PH_IMAGE].items():
    #         requested_series_number = patient_image.get(KEY_SERIES_NUMBER)
    #         requested_series_description = patient_image.get(KEY_SERIES_DESCRIPTION)
    #         requested_series_description = requested_series_description.replace(' ', '_')
    #         requested_series_description = requested_series_description.replace(';', '')
    #         requested_series_description = requested_series_description.replace(':', '')
    #         requested_series_description = requested_series_description.replace('ä', 'a')
    #         requested_series_description = requested_series_description.replace('/', '-')
    #
    #         found_entry = False
    #         for entry in correlation:
    #             matching_criteria = (entry.get(KEY_PATIENT_ID) == patient,
    #                                  entry.get(KEY_SERIES_DESCRIPTION) == requested_series_description,
    #                                  entry.get(KEY_SERIES_NUMBER) == requested_series_number)
    #             if all(matching_criteria):
    #                 data_old = assignments[patient][PH_IMAGE][modality_name]
    #                 new_image_dict.update({entry.get(PH_MODALITY): data_old})
    #                 found_entry = True
    #         if not found_entry:
    #             counter += 1
    #             print(f'No matching partner found for {patient} and {modality_name}')
    #             new_image_dict.update({modality_name: assignments[patient][PH_IMAGE][modality_name]})
    #
    #     new_dict.update({patient: {KEY_PATIENT_ID: patient,
    #                                PH_IMAGE: new_image_dict,
    #                                PH_REGISTRATION: patient_data.get(PH_REGISTRATION),
    #                                PH_SEGMENTATION: patient_data.get(PH_SEGMENTATION)}})
    #
    # print(f'Number of failures: {counter}')
    #
    # with open(output_json, 'w') as f:
    #     json.dump(new_dict, f, indent=4)
    #
    # for patient, data in new_dict.items():
    #     image_keys = sorted(list(data.get(PH_IMAGE).keys()))
    #     print(f'{patient}:\t {image_keys}')


    # ===========================
    # CONVERT THE DICOMS TO NIFTI
    # ===========================

    # Default settings
    conversion_parameters_file_path = 'D:/DataBackups/2020_08_ISAS_GBM_work/assignments.json'
    dicom_base_dir = 'D:/DataBackups/2020_08_ISAS_GBM_work/'
    nifti_output_dir = 'D:/trash/20210913_ISAS_OAR_curation/'
    reference_image_modality = 'T1c'

    # Load the conversion data
    with open(conversion_parameters_file_path, 'r') as f:
        conversion_parameters = json.load(f)  # type: dict

    # Convert the subjects
    for patient_id, patient_data in conversion_parameters.items():
        print(f'Converting patient {patient_id}...')
        patient_dir_path = os.path.join(nifti_output_dir, patient_id)
        if not os.path.exists(patient_dir_path):
            os.mkdir(patient_dir_path)

        # Get the registration files
        registration_files = []
        for registration_entry in patient_data.get(PH_REGISTRATION):
            registration_files.extend(registration_entry.get(PH_FILES))
        registration_files = tuple(registration_files)

        # Convert the dicom images and apply the registrations
        # for modality, image_data in patient_data.get(PH_IMAGE).items():
        #     print(f'Converting {modality} image...')
        #
        #     # Generate the converter parameters
        #     additional_parameters = {KEY_REFERENCE_IMAGE_PATH: image_data.get(PH_FILES)[0], }
        #     converter_parameters = DicomImageConversionParameter(**additional_parameters)
        #     converter_parameters.finalize()
        #
        #     # Convert the data
        #     image_converter = DicomToNiftiConverter(tuple(image_data.get(PH_FILES)), modality,
        #                                             os.path.dirname(image_data.get(PH_FILES)[0]))
        #     matching_registration_files = image_converter.get_matching_registration(registration_files)
        #     image_converter.set_registrations(matching_registration_files)
        #     image = image_converter.convert()
        #     file_name = image_converter.get_file_name()
        #
        #     # Save the image
        #     sitk.WriteImage(image, os.path.join(patient_dir_path, file_name))

        # Convert the segmentations
        reference_image_file = patient_data.get(PH_IMAGE).get(reference_image_modality).get(PH_FILES)[0]
        for segmentation_entry in patient_data.get(PH_SEGMENTATION):
            for segmentation_file in segmentation_entry.get(PH_FILES):
                print(f'Converting segmentations...')

                segmentation_converter = DicomRtToNiftiConverter(segmentation_file, reference_image_file)
                segmentation_images, segmentation_roi_names = segmentation_converter.convert()
                for image, roi_name in zip(segmentation_images, segmentation_roi_names):
                    file_name = f'seg_{patient_id}_{roi_name}.nii.gz'
                    sitk.WriteImage(image, os.path.join(patient_dir_path, file_name))

    print('Finished')

