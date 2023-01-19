import os
from pydicom.dataset import Tag
from dicom2nii import (
    DicomRTConversionParameter,
    DicomImageConversionParameter,
    NiftiToDicomRtConverter,
    NiftiToDicomConverter,
    KEY_REFERENCE_IMAGE_PATH,
    DEFAULT_STRUCTURE_NAME_LIST,
    TAG_MODALITY,
    TAG_SOP_INSTANCE_UID,
    DicomRtToNiftiConverter,
    DicomToNiftiConverter,
    DicomDirectoryFilter,
)

from dicom2nii.definitions import *


def testing_basic_funcs():
    # Default settings
    output_base_path = "D:\\trash\\RTSS_test\\ISAS_GBM_001\\"

    base_nii_image_directory_path = (
        "D:\\trash\\20210525_ISAS_OAR_curation_3DKP_256_V3\\ISAS_GBM_001\\"
    )
    nii_image_file_paths = (
        os.path.join(base_nii_image_directory_path, "img_ISAS_GBM_001_T1c.nii.gz"),
        os.path.join(base_nii_image_directory_path, "img_ISAS_GBM_001_T1w.nii.gz"),
        os.path.join(base_nii_image_directory_path, "img_ISAS_GBM_001_T2w.nii.gz"),
        os.path.join(base_nii_image_directory_path, "img_ISAS_GBM_001_FLAIR.nii.gz"),
        os.path.join(base_nii_image_directory_path, "img_ISAS_GBM_001_CT.nii.gz"),
    )

    base_dcm_image_reference_directory_path = (
        "D:\\DataBackups\\2020_08_ISAS_GBM_work\\ISAS_GBM_001\\"
    )
    dcm_image_reference_paths = (
        os.path.join(
            base_dcm_image_reference_directory_path,
            "MR.1.3.6.1.4.1.5962.99.1.1856959841.1925150802.1556635153761.238.0.dcm",
        ),
        os.path.join(
            base_dcm_image_reference_directory_path,
            "MR.1.3.6.1.4.1.5962.99.1.661595793.191880616.1559734757009.10.0.dcm",
        ),
        os.path.join(
            base_dcm_image_reference_directory_path,
            "MR.1.3.6.1.4.1.5962.99.1.661595793.191880616.1559734757009.213.0.dcm",
        ),
        os.path.join(
            base_dcm_image_reference_directory_path,
            "MR.1.3.6.1.4.1.5962.99.1.661595793.191880616.1559734757009.168.0.dcm",
        ),
        os.path.join(
            base_dcm_image_reference_directory_path,
            "CT.1.3.6.1.4.1.5962.99.1.2628079426.196750453.1557406273346.1014.0.dcm",
        ),
    )

    base_nii_label_directory_path = "D:\\TTIE\\Experiments\\Exp_GBM_00_METIS_00_METPO_20_2021-07-30_06-22\\testvolumes"
    nii_label_file_paths = (
        os.path.join(
            base_nii_label_directory_path, "test_sample_ISAS_GBM_001_prediction.nii.gz"
        ),
    )

    # Nifti image to DICOM image conversion
    if len(nii_image_file_paths) != len(dcm_image_reference_paths):
        raise ValueError(
            f"The number of Nifti image file paths must be equal to the number of "
            f"DICOM image reference paths!"
        )

    ref_image_file_paths = []
    for image_idx, (nii_image_file_path, dicom_image_reference_path) in enumerate(
        zip(nii_image_file_paths, dcm_image_reference_paths)
    ):
        # Build the parameters for the image conversion
        keyword_parameters_image = {
            KEY_REFERENCE_IMAGE_PATH: dicom_image_reference_path
        }
        conversion_parameters_image = DicomImageConversionParameter(
            **keyword_parameters_image
        )
        conversion_parameters_image.finalize()

        # Perform the image conversion
        converter_image = NiftiToDicomConverter(
            nii_image_file_path, conversion_parameters_image
        )
        image_datasets = converter_image.convert()
        image_output_paths = []
        for image_dataset in image_datasets:
            image_output_base_path = os.path.join(output_base_path, str(image_idx))
            if not os.path.exists(image_output_base_path):
                os.mkdir(image_output_base_path)
            image_output_name = (
                f"{image_dataset.get(Tag(TAG_MODALITY)).value}."
                f"{image_dataset.get(Tag(TAG_SOP_INSTANCE_UID)).value}.dcm"
            )
            image_output_path = os.path.join(image_output_base_path, image_output_name)
            converter_image.save_dataset(image_dataset, image_output_path)
            image_output_paths.append(image_output_path)
        ref_image_file_paths.append(image_output_paths[0])
    ref_image_file_path = ref_image_file_paths[0]

    # Build the parameters for the conversion
    keyword_parameters_rt = {KEY_REFERENCE_IMAGE_PATH: ref_image_file_path}
    conversion_parameters_rt = DicomRTConversionParameter(**keyword_parameters_rt)
    conversion_parameters_rt.finalize()

    # Perform the RT conversion
    converter = NiftiToDicomRtConverter(
        nii_label_file_paths, conversion_parameters_rt, DEFAULT_STRUCTURE_NAME_LIST
    )
    rtss_dataset = converter.convert()
    rt_output_name = f"RT.{rtss_dataset.get(Tag(TAG_SOP_INSTANCE_UID)).value}.dcm"
    rt_output_path = os.path.join(output_base_path, rt_output_name)
    converter.save_dataset(rtss_dataset, rt_output_path)


if __name__ == "__main__":

    testing_basic_funcs()

    # Testing
    base_input_directory = "D:/DataBackups/2020_08_ISAS_GBM_work/ISAS_GBM_001"
    directory_filter = DicomDirectoryFilter(base_input_directory)
    dicom_infos = directory_filter.get_info()

    dicom_image_infos = directory_filter.get_entries_by_criterion(
        KEY_SOP_CLASS_UID,
        (
            DEFAULT_SOP_CLASS_UID_MR,
            DEFAULT_SOP_CLASS_UID_MR_ENHANCED,
            DEFAULT_SOP_CLASS_UID_CT,
        ),
    )

    dicom_reg_info = directory_filter.get_entries_by_criterion(
        KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_REGISTRATION
    )
    dicom_image_series_instance_uids = directory_filter.get_unique_values(
        KEY_SERIES_INSTANCE_UID, dicom_image_infos
    )

    # Convert the CT file
    ct_files = directory_filter.get_entries_by_criterion(
        KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_CT
    )
    ct_series_instance_uids = directory_filter.get_unique_values(
        KEY_SERIES_INSTANCE_UID, ct_files
    )
    if len(ct_series_instance_uids) != 1:
        print("No or multiple CT-scans detected for patient ISAS_GBM_001!")
    for ct_series_instance_uid in ct_series_instance_uids:
        ct_files = directory_filter.get_entries_by_criterion(
            KEY_SERIES_INSTANCE_UID, ct_series_instance_uid
        )
        converter = DicomToNiftiConverter(ct_files, "CT")
        ct_image = converter.convert()
        filename = converter.get_file_name()
        # SAVE THE IMAGE

    print("Finished")
