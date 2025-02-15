# Definitions

# Converter Definitions
PH_NAME = 'name'
PH_IMAGE = 'image'
PH_COLOR = 'color'
PH_ORIGIN = 'origin'
PH_DIRECTION = 'direction'
PH_SPACING = 'spacing'
PH_FILES = 'files'
PH_STUDY_DATA = 'study_data'
PH_T1C = 'T1c'
PH_T1w = 'T1w'
PH_T2w = 'T2w'
PH_FLAIR = 'FLAIR'
PH_CT = 'CT'
PH_PATIENT_ID = 'patient_id'
PH_REGISTRATION = 'registration'
PH_SEGMENTATION = 'segmentation'
PH_MODALITY = 'modality'

# DICOM Parameters
KEY_SOP_CLASS_UID = 'SOPClassUID'
TAG_SOP_CLASS_UID = (0x0008, 0x0016)
VR_SOP_CLASS_UID = 'UI'

KEY_SOP_INSTANCE_UID = 'SOPInstanceUID'
TAG_SOP_INSTANCE_UID = (0x0008, 0x0018)
VR_SOP_INSTANCE_UID = 'UI'
KEY_SOP_INSTANCE_UID_PREFIX = 'SOPInstanceUID_prefix'

KEY_STUDY_INSTANCE_UID = 'StudyInstanceUID'
TAG_STUDY_INSTANCE_UID = (0x0020, 0x000d)
VR_STUDY_INSTANCE_UID = 'UI'
KEY_STUDY_INSTANCE_UID_PREFIX = 'StudyInstanceUID_prefix'

KEY_SERIES_INSTANCE_UID = 'SeriesInstanceUID'
TAG_SERIES_INSTANCE_UID = (0x0020, 0x000e)
VR_SERIES_INSTANCE_UID = 'UI'
KEY_SERIES_INSTANCE_UID_PREFIX = 'SeriesInstanceUID_prefix'

KEY_FRAME_OF_REFERENCE_UID = 'FrameOfReferenceUID'
TAG_FRAME_OF_REFERENCE_UID = (0x0020, 0x0052)
VR_FRAME_OF_REFERENCE_UID = 'UI'

KEY_SPECIFIC_CHARACTER_SET = 'SpecificCharacterSet'
TAG_SPECIFIC_CHARACTER_SET = (0x0008, 0x0005)
VR_SPECIFIC_CHARACTER_SET = 'CS'

KEY_MODALITY = 'Modality'
TAG_MODALITY = (0x0008, 0x0060)
VR_MODALITY = 'CS'

KEY_MANUFACTURER = 'Manufacturer'
TAG_MANUFACTURER = (0x0008, 0x0070)
VR_MANUFACTURER = 'LO'

KEY_MANUFACTURER_MODEL_NAME = 'ManufacturerModelName'
TAG_MANUFACTURER_MODEL_NAME = (0x0008, 0x1090)
VR_MANUFACTURER_MODEL_NAME = 'LO'

KEY_SERIES_DESCRIPTION = 'SeriesDescription'
TAG_SERIES_DESCRIPTION = (0x0008, 0x103e)
VR_SERIES_DESCRIPTION = 'LO'

KEY_SERIES_NUMBER = 'SeriesNumber'
TAG_SERIES_NUMBER = (0x0020, 0x0011)
VR_SERIES_NUMBER = 'IS'

KEY_INSTANCE_NUMBER = 'InstanceNumber'
TAG_INSTANCE_NUMBER = (0x0020, 0x0013)
VR_INSTANCE_NUMBER = 'IS'

KEY_SOFTWARE_VERSIONS = 'SoftwareVersions'
TAG_SOFTWARE_VERSIONS = (0x0018, 0x1020)
VR_SOFTWARE_VERSIONS = 'LO'

KEY_OPERATORS_NAME = 'OperatorsName'
TAG_OPERATORS_NAME = (0x0008, 0x1070)
VR_OPERATORS_NAME = 'PN'

KEY_ACCESSION_NUMBER = 'AccessionNumber'
TAG_ACCESSION_NUMBER = (0x0008, 0x0050)
VR_ACCESSION_NUMBER = 'SH'

KEY_APPROVAL_STATUS = 'ApprovalStatus'
TAG_APPROVAL_STATUS = (0x300e, 0x0002)
VR_APPROVAL_STATUS = 'CS'

KEY_CONTENT_DATE = 'ContentDate'
TAG_CONTENT_DATE = (0x0008, 0x0023)
VR_CONTENT_DATE = 'DA'

KEY_CONTENT_TIME = 'ContentTime'
TAG_CONTENT_TIME = (0x0008, 0x0033)
VR_CONTENT_TIME = 'TM'

KEY_INSTANCE_CREATION_DATE = 'InstanceCreationDate'
TAG_INSTANCE_CREATION_DATE = (0x0008, 0x0012)
VR_INSTANCE_CREATION_DATE = 'DA'

KEY_INSTANCE_CREATION_TIME = 'InstanceCreationTime'
TAG_INSTANCE_CREATION_TIME = (0x0008, 0x0013)
VR_INSTANCE_CREATION_TIME = 'DA'

KEY_SERIES_DATE = 'SeriesDate'
TAG_SERIES_DATE = (0x0008, 0x0021)
VR_SERIES_DATE = 'DA'

KEY_SERIES_TIME = 'SeriesTime'
TAG_SERIES_TIME = (0x0008, 0x0031)
VR_SERIES_TIME = 'TM'

KEY_IMAGE_POSITION_PATIENT = 'ImagePositionPatient'
TAG_IMAGE_POSITION_PATIENT = (0x0020, 0x0032)
VR_IMAGE_POSITION_PATIENT = 'DS'

KEY_STUDY_DATE = 'StudyDate'
TAG_STUDY_DATE = (0x0008, 0x0020)
VR_STUDY_DATE = 'DA'

KEY_STUDY_TIME = 'StudyTime'
TAG_STUDY_TIME = (0x0008, 0x0030)
VR_STUDY_TIME = 'TM'

KEY_STATION_NAME = 'StationName'
TAG_STATION_NAME = (0x0008, 0x1010)
VR_STATION_NAME = 'SH'

KEY_STUDY_DESCRIPTION = 'StudyDescription'
TAG_STUDY_DESCRIPTION = (0x0008, 0x1030)
VR_STUDY_DESCRIPTION = 'LO'

KEY_PATIENT_NAME = 'PatientName'
TAG_PATIENT_NAME = (0x0010, 0x0010)
VR_PATIENT_NAME = 'PN'

KEY_PATIENT_ID = 'PatientID'
TAG_PATIENT_ID = (0x0010, 0x0020)
VR_PATIENT_ID = 'LO'

KEY_PATIENT_BIRTH_DATE = 'PatientBirthDate'
TAG_PATIENT_BIRTH_DATE = (0x0010, 0x0030)
VR_PATIENT_BIRTH_DATE = 'DA'

KEY_PATIENT_SEX = 'PatientSex'
TAG_PATIENT_SEX = (0x0010, 0x0040)
VR_PATIENT_SEX = 'CS'

KEY_DEVICE_SERIAL_NUMBER = 'DeviceSerialNumber'
TAG_DEVICE_SERIAL_NUMBER = (0x0018, 0x1000)
VR_DEVICE_SERIAL_NUMBER = 'LO'

KEY_STUDY_ID = 'StudyID'
TAG_STUDY_ID = (0x0020, 0x0010)
VR_STUDY_ID = 'SH'

KEY_IMAGE_ORIENTATION_PATIENT = 'ImageOrientationPatient'
TAG_IMAGE_ORIENTATION_PATIENT = (0x0020, 0x0037)
VR_IMAGE_ORIENTATION_PATIENT = 'DS'

# DICOM Image Parameters
KEY_REFERENCED_IMAGE_SEQUENCE = 'ReferencedImageSequence'
TAG_REFERENCED_IMAGE_SEQUENCE = (0x0008, 0x1140)
VR_REFERENCED_IMAGE_SEQUENCE = 'SQ'

KEY_SLICE_LOCATION = 'SliceLocation'
TAG_SLICE_LOCATION = (0x0020, 0x1041)
VR_SLICE_LOCATION = 'DS'

KEY_ROWS = 'Rows'
TAG_ROWS = (0x0028, 0x0010)
VR_ROWS = 'US'

KEY_COLUMNS = 'Columns'
TAG_COLUMNS = (0x0028, 0x0011)
VR_COLUMNS = 'US'

KEY_PIXEL_SPACING = 'PixelSpacing'
TAG_PIXEL_SPACING = (0x0028, 0x0030)
VR_PIXEL_SPACING = 'DS'

KEY_PIXEL_DATA = 'PixelData'
TAG_PIXEL_DATA = (0x7fe0, 0x0010)
VR_PIXEL_DATA = 'OW'

KEY_WINDOW_CENTER = 'WindowCenter'
TAG_WINDOW_CENTER = (0x0028, 0x1050)
VR_WINDOW_CENTER = 'DS'

KEY_WINDOW_WIDTH = 'WindowWidth'
TAG_WINDOW_WIDTH = (0x0028, 0x1051)
VR_WINDOW_WIDTH = 'DS'

# DICOM RTSS Parameters
KEY_STRUCTURE_SET_DATE = 'StructureSetDate'
TAG_STRUCTURE_SET_DATE = (0x3006, 0x0008)
VR_STRUCTURE_SET_DATE = 'DA'

KEY_STRUCTURE_SET_TIME = 'StructureSetTime'
TAG_STRUCTURE_SET_TIME = (0x3006, 0x0009)
VR_STRUCTURE_SET_TIME = 'TM'

KEY_REFERENCED_SOP_CLASS_UID = 'ReferencedSOPClassUID'
TAG_REFERENCED_SOP_CLASS_UID = (0x0008, 0x1150)
VR_REFERENCED_SOP_CLASS_UID = 'UI'

KEY_REFERENCED_SOP_INSTANCE_UID = 'ReferencedSOPInstanceUID'
TAG_REFERENCED_SOP_INSTANCE_UID = (0x0008, 0x1155)
VR_REFERENCED_SOP_INSTANCE_UID = 'UI'

KEY_CONTOUR_GEOMETRIC_TYPE = 'ContourGeometricType'
TAG_CONTOUR_GEOMETRIC_TYPE = (0x3006, 0x0042)
VR_CONTOUR_GEOMETRIC_TYPE = 'CS'

KEY_CONTOUR_IMAGE_SEQUENCE = 'ContourImageSequence'
TAG_CONTOUR_IMAGE_SEQUENCE = (0x3006, 0x0016)
VR_CONTOUR_IMAGE_SEQUENCE = 'SQ'

KEY_NUMBER_OF_CONTOUR_POINTS = 'NumberOfContourPoints'
TAG_NUMBER_OF_CONTOUR_POINTS = (0x3006, 0x0046)
VR_NUMBER_OF_CONTOUR_POINTS = 'IS'

KEY_CONTOUR_DATA = 'ContourData'
TAG_CONTOUR_DATA = (0x3006, 0x0050)
VR_CONTOUR_DATA = 'DS'

KEY_ROI_DISPLAY_COLOR = 'ROIDisplayColor'
TAG_ROI_DISPLAY_COLOR = (0x3006, 0x002a)
VR_ROI_DISPLAY_COLOR = 'IS'

KEY_REFERENCED_ROI_NUMBER = 'ReferencedROINumber'
TAG_REFERENCED_ROI_NUMBER = (0x3006, 0x0084)
VR_REFERENCED_ROI_NUMBER = 'IS'

KEY_CONTOUR_SEQUENCE = 'ContourSequence'
TAG_CONTOUR_SEQUENCE = (0x3006, 0x0040)
VR_CONTOUR_SEQUENCE = 'SQ'

KEY_ROI_CONTOUR_SEQUENCE = 'ROIContourSequence'
TAG_ROI_CONTOUR_SEQUENCE = (0x3006, 0x0039)
VR_ROI_CONTOUR_SEQUENCE = 'SQ'

KEY_STRUCTURE_SET_ROI_SEQUENCE = 'StructureSetROISequence'
TAG_STRUCTURE_SET_ROI_SEQUENCE = (0x3006, 0x0020)
VR_STRUCTURE_SET_ROI_SEQUENCE = 'SQ'

KEY_ROI_NUMBER = 'ROINumber'
TAG_ROI_NUMBER = (0x3006, 0x0022)
VR_ROI_NUMBER = 'IS'

KEY_REFERENCED_FRAME_OF_REFERENCE_UID = 'ReferencedFrameOfReferenceUID'
TAG_REFERENCED_FRAME_OF_REFERENCE_UID = (0x3006, 0x0024)
VR_REFERENCED_FRAME_OF_REFERENCE_UID = 'UI'

KEY_ROI_NAME = 'ROIName'
TAG_ROI_NAME = (0x3006, 0x0026)
VR_ROI_NAME = 'LO'

KEY_ROI_GENERATION_ALGORITHM = 'ROIGenerationAlgorithm'
TAG_ROI_GENERATION_ALGORITHM = (0x3006, 0x0036)
VR_ROI_GENERATION_ALGORITHM = 'CS'

KEY_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE = 'ReferencedFrameOfReferenceSequence'
TAG_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE = (0x3006, 0x0010)
VR_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE = 'SQ'

KEY_RT_REFERENCED_STUDY_SEQUENCE = 'RTReferencedStudySequence'
TAG_RT_REFERENCED_STUDY_SEQUENCE = (0x3006, 0x0012)
VR_RT_REFERENCED_STUDY_SEQUENCE = 'SQ'

KEY_RT_REFERENCED_SERIES_SEQUENCE = 'RTReferencedSeriesSequence'
TAG_RT_REFERENCED_SERIES_SEQUENCE = (0x3006, 0x0014)
VR_RT_REFERENCED_SERIES_SEQUENCE = 'SQ'

KEY_CODING_SCHEME_IDENTIFICATION_SEQUENCE = 'CodingSchemeIdentificationSequence'
TAG_CODING_SCHEME_IDENTIFICATION_SEQUENCE = (0x0008, 0x0110)
VR_CODING_SCHEME_IDENTIFICATION_SEQUENCE = 'SQ'

KEY_CODING_SCHEME_DESIGNATOR = 'CodingSchemeDesignator'
TAG_CODING_SCHEME_DESIGNATOR = (0x0008, 0x0102)
VR_CODING_SCHEME_DESIGNATOR = 'SH'

KEY_CODING_SCHEME_UID = 'CodingSchemeUID'
TAG_CODING_SCHEME_UID = (0x0008, 0x010c)
VR_CODING_SCHEME_UID = 'UI'

KEY_CODING_SCHEME_NAME = 'CodingSchemeName'
TAG_CODING_SCHEME_NAME = (0x0008, 0x0115)
VR_CODING_SCHEME_NAME = 'ST'

KEY_CODING_SCHEME_RESPONSIBLE_ORGANISATION = 'CodingSchemeResponsibleOrganization'
TAG_CODING_SCHEME_RESPONSIBLE_ORGANISATION = (0x0008, 0x0116)
VR_CODING_SCHEME_RESPONSIBLE_ORGANISATION = 'ST'

KEY_CONTEXT_GROUP_IDENTIFICATION_SEQUENCE = 'ContextGroupIdentificationSequence'
TAG_CONTEXT_GROUP_IDENTIFICATION_SEQUENCE = (0x0008, 0x0123)
VR_CONTEXT_GROUP_IDENTIFICATION_SEQUENCE = 'SQ'

KEY_MAPPING_RESOURCE = 'MappingResource'
TAG_MAPPING_RESOURCE = (0x0008, 0x0105)
VR_MAPPING_RESOURCE = 'CS'

KEY_CONTEXT_GROUP_VERSION = 'ContextGroupVersion'
TAG_CONTEXT_GROUP_VERSION = (0x0008, 0x0106)
VR_CONTEXT_GROUP_VERSION = 'DT'

KEY_CONTEXT_IDENTIFIER = 'ContextIdentifier'
TAG_CONTEXT_IDENTIFIER = (0x0008, 0x010f)
VR_CONTEXT_IDENTIFIER = 'CS'

KEY_CONTEXT_UID = 'ContextUID'
TAG_CONTEXT_UID = (0x0008, 0x0117)
VR_CONTEXT_UID = 'UI'

KEY_MAPPING_RESOURCE_IDENTIFICATION_SEQUENCE = 'MappingResourceIdentificationSequence'
TAG_MAPPING_RESOURCE_IDENTIFICATION_SEQUENCE = (0x0008, 0x0124)
VR_MAPPING_RESOURCE_IDENTIFICATION_SEQUENCE = 'SQ'

KEY_MAPPING_RESOURCE_UID = 'MappingResourceUID'
TAG_MAPPING_RESOURCE_UID = (0x0008, 0x0118)
VR_MAPPING_RESOURCE_UID = 'UI'

KEY_MAPPING_RESOURCE_NAME = 'MappingResourceName'
TAG_MAPPING_RESOURCE_NAME = (0x0008, 0x0122)
VR_MAPPING_RESOURCE_NAME = 'LO'

KEY_STRUCTURE_SET_LABEL = 'StructureSetLabel'
TAG_STRUCTURE_SET_LABEL = (0x3006, 0x0002)
VR_STRUCTURE_SET_LABEL = 'SH'

KEY_RT_ROI_OBSERVATION_SEQUENCE = 'RTROIObservationsSequence'
TAG_RT_ROI_OBSERVATION_SEQUENCE = (0x3006, 0x0080)
VR_RT_ROI_OBSERVATION_SEQUENCE = 'SQ'

KEY_OBSERVATION_NUMBER = 'ObservationNumber'
TAG_OBSERVATION_NUMBER = (0x3006, 0x0082)
VR_OBSERVATION_NUMBER = 'IS'

KEY_ROI_OBSERVATION_LABEL = 'ROIObservationLabel'
TAG_ROI_OBSERVATION_LABEL = (0x3006, 0x0085)
VR_ROI_OBSERVATION_LABEL = 'SH'

KEY_RT_ROI_IDENTIFICATION_CODE_SEQUENCE = 'RTROIIdentificationCodeSequence'
TAG_RT_ROI_IDENTIFICATION_CODE_SEQUENCE = (0x3006, 0x0086)
VR_RT_ROI_IDENTIFICATION_CODE_SEQUENCE = 'SQ'

KEY_RT_ROI_INTERPRETED_TYPE = 'RTROIInterpretedType'
TAG_RT_ROI_INTERPRETED_TYPE = (0x3006, 0x00a4)
VR_RT_ROI_INTERPRETED_TYPE = 'CS'

KEY_ROI_INTERPRETER = 'ROIInterpreter'
TAG_ROI_INTERPRETER = (0x3006, 0x00a6)
VR_ROI_INTERPRETER = 'PN'

KEY_CODE_VALUE = 'CodeValue'
TAG_CODE_VALUE = (0x0008, 0x0100)
VR_CODE_VALUE = 'SH'

KEY_CODING_SCHEME_VERSION = 'CodingSchemeVersion'
TAG_CODING_SCHEME_VERSION = (0x0008, 0x0103)
VR_CODING_SCHEME_VERSION = 'SH'

KEY_CODE_MEANING = 'CodeMeaning'
TAG_CODE_MEANING = (0x0008, 0x0104)
VR_CODE_MEANING = 'LO'

# DICOM Registration Parameters
KEY_REGISTRATION_SEQUENCE = 'RegistrationSequence'
TAG_REGISTRATION_SEQUENCE = (0x0070, 0x0308)
VR_REGISTRATION_SEQUENCE = 'SQ'

KEY_MATRIX_REGISTRATION_SEQUENCE = 'MatrixRegistrationSequence'
TAG_MATRIX_REGISTRATION_SEQUENCE = (0x0070, 0x0309)
VR_MATRIX_REGISTRATION_SEQUENCE = 'SQ'

KEY_MATRIX_SEQUENCE = 'MatrixSequence'
TAG_MATRIX_SEQUENCE = (0x0070, 0x030a)
VR_MATRIX_SEQUENCE = 'SQ'

KEY_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE = 'FrameOfReferenceTransformationMatrixType'
TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE = (0x0070, 0x030c)
VR_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE = 'CS'

KEY_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX = 'FrameOfReferenceTransformationMatrix'
TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX = (0x3006, 0x00c6)
VR_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX = 'DS'

KEY_STUDIES_CONTAINING_OTHER_REFERENCED_INSTANCES_SEQUENCE = 'StudiesContainingOtherReferencedInstancesSequence'
TAG_STUDIES_CONTAINING_OTHER_REFERENCED_INSTANCES_SEQUENCE = (0x0008, 0x1200)
VR_STUDIES_CONTAINING_OTHER_REFERENCED_INSTANCES_SEQUENCE = 'SQ'

KEY_REFERENCED_SERIES_SEQUENCE = 'ReferencedSeriesSequence'
TAG_REFERENCED_SERIES_SEQUENCE = (0x0008, 0x1115)
VR_REFERENCED_SERIES_SEQUENCE = 'SQ'

# Parameters not related to DICOM
KEY_REFERENCE_IMAGE_PATH = 'reference_image_path'
KEY_STRUCTURE_INFO = 'structure_info'

# Default Parameter Values
DEFAULT_SOP_CLASS_UID_RTSS = '1.2.840.10008.5.1.4.1.1.481.3'
DEFAULT_SOP_CLASS_UID_CT = '1.2.840.10008.5.1.4.1.1.2'
DEFAULT_SOP_CLASS_UID_MR = '1.2.840.10008.5.1.4.1.1.4'
DEFAULT_SOP_CLASS_UID_MR_ENHANCED = '1.2.840.10008.5.1.4.1.1.4.1'
DEFAULT_SOP_CLASS_UID_MR_SPECTROSCOPY = '1.2.840.10008.5.1.4.1.1.4.2'
DEFAULT_SOP_CLASS_UID_REGISTRATION = '1.2.840.10008.5.1.4.1.1.66.1'
DEFAULT_SOP_INSTANCE_UID_PREFIX = '1.2.246.352.205.'
DEFAULT_STUDY_INSTANCE_UID_PREFIX = '1.3.6.1.4.1.5264.'  # UID for Varian Medical Systems
DEFAULT_SERIES_INSTANCE_UID_PREFIX = '1.2.246.352.205.'
DEFAULT_MODALITY_RTSTRUCT = 'RTSTRUCT'
DEFAULT_MANUFACTURER = 'Varian Medical Systems'
DEFAULT_MANUFACTURER_MODEL_NAME = 'ISAS InnoSuisse Auto Segmentation Project'
DEFAULT_SERIES_DESCRIPTION = 'Automatic contours generated by AI model'
DEFAULT_SERIES_NUMBER = '99'
DEFAULT_INSTANCE_NUMBER = '99'
DEFAULT_SOFTWARE_VERSION = 'X0.0.1'
DEFAULT_OPERATORS_NAME = 'AI Algorithm'
DEFAULT_ACCESSION_NUMBER = ''
DEFAULT_APPROVAL_STATUS = 'UNAPPROVED'
DEFAULT_STATION_NAME = 'ISAS AutoSegmentation Algorithm'
DEFAULT_DEVICE_SERIAL_NUMBER = '1'
DEFAULT_STUDY_ID = 'S'
DEFAULT_STRUCTURE_SET_LABEL = 'Autogenerated StructureSet 3'  # TODO Change this back
DEFAULT_CONTOUR_GEOMETRIC_TYPE = 'CONTOUR_PLANAR'
DEFAULT_MAPPING_RESOURCE = '99VMS'
DEFAULT_CODING_SCHEME_DESIGNATOR = 'FMA'
DEFAULT_CODING_SCHEME_VERSION = '3.2'
DEFAULT_CONTEXT_IDENTIFIER = 'VMS011'
DEFAULT_CONTEXT_GROUP_VERSION = '20161209'
DEFAULT_CONTEXT_UID = '1.2.246.352.7.2.11'
DEFAULT_MAPPING_RESOURCE_UID = '1.2.246.352.7.1.1'
DEFAULT_RT_ROI_INTERPRETED_TYPE = 'ORGAN'
DEFAULT_ROI_INTERPRETER = ''
DEFAULT_ROI_GENERATION_ALGORITHM = 'AUTOMATIC'
DEFAULT_CODING_SCHEME_UID = '2.16.840.1.113883.6.119'
DEFAULT_STRUCTURE_INFO = {
    "Retina_R": {
        KEY_CODE_VALUE: '58302',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right retina',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Retina_L": {
        KEY_CODE_VALUE: '58303',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left retina',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Pituitary": {
        KEY_CODE_VALUE: '13889',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Pituitary gland',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "OpticNerve_R": {
        KEY_CODE_VALUE: '50875',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right optic nerve',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "OpticNerve_L": {
        KEY_CODE_VALUE: '50878',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left optic nerve',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "OpticChiasm": {
        KEY_CODE_VALUE: '62045',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Optic chiasm',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Lens_R": {
        KEY_CODE_VALUE: '58242',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right lens',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Lens_L": {
        KEY_CODE_VALUE: '58243',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left lens',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Lacrimal_R": {
        KEY_CODE_VALUE: '59102',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right lacrimal gland',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Lacrimal_L": {
        KEY_CODE_VALUE: '59103',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left lacrimal gland',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Hippocampus_R": {
        KEY_CODE_VALUE: '275022',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right hippocampus',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Hippocampus_L": {
        KEY_CODE_VALUE: '275024',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left hippocampus',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Eye_R": {
        KEY_CODE_VALUE: '12514',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right eyeball',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Eye_L": {
        KEY_CODE_VALUE: '12515',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left eyeball',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Cochlea_R": {
        KEY_CODE_VALUE: '60202',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Right cochlea',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Cochlea_L": {
        KEY_CODE_VALUE: '60203',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Left cochlea',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Brainstem": {
        KEY_CODE_VALUE: '79876',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Brainstem',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: DEFAULT_RT_ROI_INTERPRETED_TYPE},
    "Resection_Cavity": {
        KEY_CODE_VALUE: 'CTV_Intermediate',
        KEY_CODING_SCHEME_DESIGNATOR: '99VMS_STRUCTCODE',
        KEY_CODING_SCHEME_VERSION: '1.0',
        KEY_CODE_MEANING: 'Target Volume Intermediate Risk',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: 'CTV'},
    "Brain": {
        KEY_CODE_VALUE: '50801',
        KEY_CODING_SCHEME_DESIGNATOR: DEFAULT_CODING_SCHEME_DESIGNATOR,
        KEY_CODING_SCHEME_VERSION: DEFAULT_CODING_SCHEME_VERSION,
        KEY_CODE_MEANING: 'Brain',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: 'PTV'},
    "OpticTract_L": {
        KEY_CODE_VALUE: 'NormalTissue',
        KEY_CODING_SCHEME_DESIGNATOR: '99VMS_STRUCTCODE',
        KEY_CODING_SCHEME_VERSION: '1.0',
        KEY_CODE_MEANING: 'Undefined Normal Tissue',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: 'AVOIDANCE'},
    "OpticTract_R": {
        KEY_CODE_VALUE: 'NormalTissue',
        KEY_CODING_SCHEME_DESIGNATOR: '99VMS_STRUCTCODE',
        KEY_CODING_SCHEME_VERSION: '1.0',
        KEY_CODE_MEANING: 'Undefined Normal Tissue',
        KEY_MAPPING_RESOURCE: DEFAULT_MAPPING_RESOURCE,
        KEY_CONTEXT_GROUP_VERSION: DEFAULT_CONTEXT_GROUP_VERSION,
        KEY_CONTEXT_IDENTIFIER: DEFAULT_CONTEXT_IDENTIFIER,
        KEY_CONTEXT_UID: DEFAULT_CONTEXT_UID,
        KEY_MAPPING_RESOURCE_UID: DEFAULT_MAPPING_RESOURCE_UID,
        KEY_MAPPING_RESOURCE_NAME: DEFAULT_MANUFACTURER,
        KEY_RT_ROI_INTERPRETED_TYPE: 'AVOIDANCE'}}

DEFAULT_STRUCTURE_NAME_LIST = {"Brainstem": 1,
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
                               "OpticTract_L": 18,
                               "OpticTract_R": 19}
