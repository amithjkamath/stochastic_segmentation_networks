from abc import (ABC, abstractmethod)
from datetime import datetime
from pydicom.uid import generate_uid
from .definitions import *


class ConversionParameter(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def _validate(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError


class BaseDicomConversionParameter(ConversionParameter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # DICOM related parameters
        self._time_now = datetime.now()
        self.content_date = self._time_now.strftime('%Y%m%d')
        self.content_time = self._time_now.strftime('%H%M%S.%f')

        self.sop_class_uid = kwargs.get(KEY_SOP_CLASS_UID, None)
        self.sop_instance_uid_prefix = kwargs.get(KEY_SOP_INSTANCE_UID_PREFIX, DEFAULT_SOP_INSTANCE_UID_PREFIX)
        self.sop_instance_uid = kwargs.get(KEY_SOP_INSTANCE_UID, None)
        self.study_instance_uid_prefix = kwargs.get(KEY_STUDY_INSTANCE_UID_PREFIX, DEFAULT_STUDY_INSTANCE_UID_PREFIX)
        self.study_instance_uid = kwargs.get(KEY_STUDY_INSTANCE_UID, None)
        self.series_instance_uid_prefix = kwargs.get(KEY_SERIES_INSTANCE_UID_PREFIX, DEFAULT_SERIES_INSTANCE_UID_PREFIX)
        self.series_instance_uid = kwargs.get(KEY_SERIES_INSTANCE_UID, None)
        self.series_description = kwargs.get(KEY_SERIES_DESCRIPTION, '')
        self.series_number = kwargs.get(KEY_SERIES_NUMBER, DEFAULT_SERIES_NUMBER)
        self.modality = kwargs.get(KEY_MODALITY, None)
        self.study_id = kwargs.get(KEY_STUDY_ID, DEFAULT_STUDY_ID)
        self.accession_number = kwargs.get(KEY_ACCESSION_NUMBER, DEFAULT_ACCESSION_NUMBER)
        self.manufacturer = kwargs.get(KEY_MANUFACTURER, DEFAULT_MANUFACTURER)
        self.manufacturer_model_name = kwargs.get(KEY_MANUFACTURER_MODEL_NAME, DEFAULT_MANUFACTURER_MODEL_NAME)
        self.station_name = kwargs.get(KEY_STATION_NAME, DEFAULT_STATION_NAME)
        self.software_version = kwargs.get(KEY_SOFTWARE_VERSIONS, DEFAULT_SOFTWARE_VERSION)
        self.device_serial_number = kwargs.get(KEY_DEVICE_SERIAL_NUMBER, DEFAULT_DEVICE_SERIAL_NUMBER)
        self.implementation_class_uid = None

        # DICOM unrelated parameters
        self.reference_dicom_path = kwargs.get(KEY_REFERENCE_IMAGE_PATH, None)

    @abstractmethod
    def _validate(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError


class DicomImageConversionParameter(BaseDicomConversionParameter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # DICOM related parameters
        # NONE

        # DICOM unrelated parameters
        # NONE

    def _validate(self) -> bool:
        entries_to_check_on_none = (self.sop_instance_uid,
                                    self.series_instance_uid,
                                    self.reference_dicom_path)
        return all(entry is not None for entry in entries_to_check_on_none)

    def finalize(self) -> None:
        # Set or generate the SOPInstanceUID
        self.sop_instance_uid = generate_uid(prefix=self.sop_instance_uid_prefix) \
            if self.sop_instance_uid is None else None

        # Set the SeriesInstanceUID
        self.series_instance_uid = generate_uid(prefix=self.series_instance_uid_prefix) \
            if self.series_instance_uid is None else None

        if not self._validate():
            raise ValueError(f'The parameter set is not valid!')


class DicomRTConversionParameter(BaseDicomConversionParameter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # DICOM related parameters
        self.sop_class_uid = kwargs.get(KEY_SOP_CLASS_UID, DEFAULT_SOP_CLASS_UID_RTSS)
        self.modality = kwargs.get(KEY_MODALITY, DEFAULT_MODALITY_RTSTRUCT)
        self.instance_number = kwargs.get(KEY_INSTANCE_NUMBER, DEFAULT_INSTANCE_NUMBER)
        self.operators_name = kwargs.get(KEY_OPERATORS_NAME, DEFAULT_OPERATORS_NAME)
        self.approval_status = kwargs.get(KEY_APPROVAL_STATUS, DEFAULT_APPROVAL_STATUS)
        self.structure_set_label = kwargs.get(KEY_STRUCTURE_SET_LABEL, DEFAULT_STRUCTURE_SET_LABEL)
        self.contour_geometric_type = kwargs.get(KEY_CONTOUR_GEOMETRIC_TYPE, DEFAULT_CONTOUR_GEOMETRIC_TYPE)
        self.roi_generation_algorithm = kwargs.get(KEY_ROI_GENERATION_ALGORITHM, DEFAULT_ROI_GENERATION_ALGORITHM)
        self.coding_scheme_uid = kwargs.get(KEY_CODING_SCHEME_UID, DEFAULT_CODING_SCHEME_UID)
        self.coding_scheme_designator = kwargs.get(KEY_CODING_SCHEME_DESIGNATOR, DEFAULT_CODING_SCHEME_DESIGNATOR)

        # DICOM unrelated parameters
        self.structure_info = kwargs.get(KEY_STRUCTURE_INFO, DEFAULT_STRUCTURE_INFO)

    def _validate(self) -> bool:
        entries_to_check_on_none = (self.sop_class_uid,
                                    self.sop_instance_uid,
                                    self.study_instance_uid,
                                    self.series_instance_uid,
                                    self.implementation_class_uid,
                                    self.reference_dicom_path)
        return all(entry is not None for entry in entries_to_check_on_none)

    def finalize(self) -> None:
        # Set or generate the SOPInstanceUID
        self.sop_instance_uid = generate_uid(prefix=self.sop_instance_uid_prefix) \
            if self.sop_instance_uid is None else None

        # Set the StudyInstanceUID
        self.study_instance_uid = generate_uid(prefix=self.study_instance_uid_prefix) \
            if self.study_instance_uid is None else None

        # Set the SeriesInstanceUID
        self.series_instance_uid = generate_uid(prefix=self.series_instance_uid_prefix) \
            if self.series_instance_uid is None else None

        # Set the ImplementationClassUID
        self.implementation_class_uid = generate_uid(prefix=self.study_instance_uid_prefix) \
            if self.implementation_class_uid is None else None

        if not self._validate():
            raise ValueError(f'The parameter set is not valid!')
