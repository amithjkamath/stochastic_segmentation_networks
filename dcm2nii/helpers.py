import os
from typing import List, Dict, Any, Optional
from glob import glob
from pydicom import dcmread
from pydicom.dataset import Tag
import SimpleITK as sitk
import numpy as np
import itk

from dicom2nii import (
    KEY_SERIES_DESCRIPTION,
    TAG_SERIES_DESCRIPTION,
    KEY_SOP_INSTANCE_UID,
    TAG_SOP_INSTANCE_UID,
    KEY_STUDY_INSTANCE_UID,
    TAG_STUDY_INSTANCE_UID,
    KEY_SERIES_NUMBER,
    TAG_SERIES_NUMBER,
    KEY_PATIENT_ID,
    TAG_PATIENT_ID,
    PH_FILES,
    KEY_SERIES_INSTANCE_UID,
    TAG_SERIES_INSTANCE_UID,
    KEY_SOP_CLASS_UID,
    TAG_SOP_CLASS_UID,
)


def reorient_image_to_rai(
    image: sitk.Image, orientation_name: str = "RAS"
) -> sitk.Image:
    if orientation_name == "LPS":
        orientation = (3 << 0) + (4 << 8) + (9 << 16)
    elif orientation_name == "RAI":
        orientation = (2 << 0) + (5 << 8) + (8 << 16)
    elif orientation_name == "RAS":
        orientation = (2 << 0) + (5 << 8) + (9 << 16)
    elif orientation_name == "LAI":
        orientation = (3 << 0) + (5 << 8) + (8 << 16)

    itk_image = itk.GetImageFromArray(
        sitk.GetArrayFromImage(image),
        is_vector=image.GetNumberOfComponentsPerPixel() > 1,
    )
    itk_image.SetOrigin(image.GetOrigin())
    itk_image.SetSpacing(image.GetSpacing())
    itk_image.SetDirection(
        itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [3] * 2))
    )

    itk_image_type = itk.Image[itk.template(itk_image)[1]]
    orientation_filter = itk.OrientImageFilter[itk_image_type, itk_image_type].New()
    orientation_filter.SetUseImageDirection(True)
    orientation_filter.SetDesiredCoordinateOrientation(orientation)
    orientation_filter.SetInput(itk_image)
    orientation_filter.Update()
    itk_image_reoriented = orientation_filter.GetOutput()

    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image_reoriented),
        isVector=itk_image_reoriented.GetNumberOfComponentsPerPixel() > 1,
    )
    sitk_image.SetOrigin(tuple(itk_image_reoriented.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image_reoriented.GetSpacing()))
    sitk_image.SetDirection(
        itk.GetArrayFromMatrix(itk_image_reoriented.GetDirection()).flatten()
    )
    return sitk_image


class DicomDirectoryFilter:
    def __init__(self, directory_path: str) -> None:
        super().__init__()
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(
                f"The directory path {directory_path} does not specify a directory!"
            )
        self.directory_path = os.path.normpath(directory_path)
        self.info = list()
        self.execute()

    def get_unique_info_by_name(self, name: str) -> List[str]:
        if not self.info:
            return list()
        else:
            if name not in self.info[0].keys():
                raise ValueError(f"The name {name} is not contained in the entries!")
            data = [entry.get(name) for entry in self.info]
            return list(set(data))

    def get_entries_by_criterion(
        self,
        name: str,
        match_criterion: Any,
        info: Optional[List[Dict[str, Any]]] = None,
    ):
        internal_info = info if info is not None else self.info
        if not internal_info:
            return internal_info
        else:
            output_entries = []
            for entry in internal_info:
                value = entry.get(name, None)
                if isinstance(match_criterion, tuple) or isinstance(
                    match_criterion, list
                ):
                    output_entries.append(entry) if value in match_criterion else None
                else:
                    output_entries.append(entry) if value == match_criterion else None
            return output_entries

    def get_unique_values(self, name: str, info: Optional[List[Dict[str, Any]]] = None):
        internal_info = info if info is not None else self.info
        if not internal_info:
            return internal_info
        else:
            output_values = []
            for entry in internal_info:
                value = entry.get(name, None)
                output_values.append(value) if value is not None else None
            return list(set(output_values))

    def get_info(self) -> List[Dict[str, Any]]:
        return self.info

    def execute(self) -> List[Dict[str, Any]]:
        file_paths = glob(self.directory_path + "/*.dcm", recursive=True)
        for files_path in file_paths:
            dataset = dcmread(files_path)
            dataset.decode()

            study_instance_uid = str(
                dataset.get_item(Tag(TAG_STUDY_INSTANCE_UID)).value
            )
            sop_instance_uid = str(dataset.get_item(Tag(TAG_SOP_INSTANCE_UID)).value)
            sop_class_uid = str(dataset.get_item(Tag(TAG_SOP_CLASS_UID)).value)
            series_instance_uid = str(
                dataset.get_item(Tag(TAG_SERIES_INSTANCE_UID)).value
            )
            series_description = dataset.get_item(Tag(TAG_SERIES_DESCRIPTION))
            series_description = (
                str(series_description.value)
                if series_description is not None
                else "Unnamed_Series"
            )
            series_number = dataset.get_item(Tag(TAG_SERIES_NUMBER))
            series_number = (
                int(series_number.value) if series_number is not None else ""
            )
            patient_id = dataset.get_item(Tag(TAG_PATIENT_ID))
            patient_id = str(patient_id.value) if patient_id is not None else ""

            info_entry = {
                KEY_STUDY_INSTANCE_UID: study_instance_uid,
                KEY_SOP_INSTANCE_UID: sop_instance_uid,
                KEY_SOP_CLASS_UID: sop_class_uid,
                KEY_SERIES_INSTANCE_UID: series_instance_uid,
                KEY_SERIES_DESCRIPTION: series_description,
                KEY_SERIES_NUMBER: series_number,
                KEY_PATIENT_ID: patient_id,
                PH_FILES: files_path,
            }

            self.info.append(info_entry)
        return self.info
