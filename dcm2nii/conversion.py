from glob import glob
import math
from vtk.util.numpy_support import vtk_to_numpy
from colorsys import hls_to_rgb
from pydicom import errors, dcmwrite
from .generation import *


class BaseConverter(ABC):
    def __init__(self, input_file_paths: Union[str, Tuple[str, ...]]) -> None:
        super().__init__()
        self.input_file_paths = self._check_file_path(input_file_paths)

    @staticmethod
    def _check_file_path(file_paths: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
        if isinstance(file_paths, str):
            if not os.path.isfile(file_paths):
                raise FileNotFoundError(f"The file {file_paths} is not existing!")
            return (file_paths,)
        if isinstance(file_paths, tuple):
            for path in file_paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"The file {path} is not existing!")
            return file_paths

    @staticmethod
    def load_image(path: str, must_be_dicom: bool = False) -> sitk.Image:
        internal_path = os.path.normpath(path)
        if not os.path.exists(internal_path):
            raise FileNotFoundError(f"The file {internal_path} is not existing!")
        if must_be_dicom and not internal_path.endswith(".dcm"):
            raise ValueError(f"The file {internal_path} is not a DICOM file!")

        if internal_path.endswith(".dcm") or internal_path.endswith(".nii.gz"):
            reader = sitk.ImageFileReader()
            reader.SetFileName(internal_path)
        else:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(internal_path))
            reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        return reader.Execute()

    @staticmethod
    def load_images(paths: Tuple[str, ...]) -> Tuple[sitk.Image, ...]:
        images = []
        for path in paths:
            images.append(BaseConverter.load_image(path))
        return tuple(images)

    @staticmethod
    def get_numpy_from_sitk_image(
        image: sitk.Image,
    ) -> Tuple[np.ndarray, Tuple, Tuple, Tuple]:
        np_image = sitk.GetArrayFromImage(image)
        origin = tuple(image.GetOrigin())
        spacing = tuple(image.GetSpacing())
        direction = tuple(image.GetDirection())
        return np_image, origin, spacing, direction

    @staticmethod
    def get_numpy_from_sitk_images(
        images: Tuple[sitk.Image, ...]
    ) -> Tuple[Tuple[np.ndarray, Tuple, Tuple, Tuple], ...]:
        output = []
        for image in images:
            output.append(BaseConverter.get_numpy_from_sitk_image(image))
        return tuple(output)

    @abstractmethod
    def convert(self) -> Any:
        raise NotImplemented


class BaseDicomConverter(BaseConverter):
    def __init__(self, input_file_paths: Union[str, Tuple[str, ...]]) -> None:
        super().__init__(input_file_paths)

    @staticmethod
    def save_dataset(
        dataset: Dataset,
        file_path: str,
        is_little_endian: bool = True,
        is_implicit_vr: bool = True,
    ) -> None:
        dataset.is_little_endian = is_little_endian
        dataset.is_implicit_VR = is_implicit_vr
        dcmwrite(file_path, dataset)

    @abstractmethod
    def convert(self) -> Dataset:
        raise NotImplementedError


class BaseNiftiConverter(BaseConverter):
    def __init__(self, input_file_paths: Union[str, Tuple[str, ...]]) -> None:
        super().__init__(input_file_paths)

    @staticmethod
    def save_nifti(image: sitk.Image, file_path: str) -> None:
        sitk.WriteImage(image, file_path)

    @abstractmethod
    def convert(self) -> Any:
        raise NotImplementedError


class NiftiToDicomConverter(BaseDicomConverter):
    def __init__(
        self,
        input_file_paths: Union[str, Tuple[str, ...]],
        params: DicomImageConversionParameter,
    ) -> None:
        super().__init__(input_file_paths)
        self.parameters = params

    def convert(self) -> Tuple[Dataset, ...]:
        # Get the information from the reference image
        reference_dataset = BaseDicomGenerator.load_dicom_metadata(
            self.parameters.reference_dicom_path
        )

        # Load the nifti image file
        if isinstance(self.input_file_paths, tuple):
            if len(self.input_file_paths) > 1:
                raise ValueError(
                    f"The {self.__class__.__name__} received multiple file paths but can convert "
                    f"only a single Nifti file!"
                )
        image = BaseDicomConverter.load_image(self.input_file_paths[0])

        img_generator = BaseDicomImageGenerator(image, self.parameters)
        base_datasets = img_generator.execute(reference_dataset)

        datasets = []
        for dataset in base_datasets:

            # Generate the Referenced Image Sequence
            referenced_sop_class_uids = (dataset.get(Tag(TAG_SOP_CLASS_UID)).value,)
            referenced_sop_instance_uids = (f"{self.parameters.sop_instance_uid}.1.0",)
            ris_generator = ReferencedImageSequenceGenerator(
                referenced_sop_class_uids, referenced_sop_instance_uids
            )
            dataset = ris_generator.execute(dataset)
            datasets.append(dataset)

        return tuple(datasets)


class NiftiToDicomRtConverter(BaseDicomConverter):
    def __init__(
        self,
        input_file_paths: Union[str, Tuple[str, ...]],
        params: DicomRTConversionParameter,
        structure_names: Dict[str, int] = DEFAULT_STRUCTURE_NAME_LIST,
    ) -> None:
        super().__init__(input_file_paths)
        self.structure_names = structure_names
        self.parameters = params

    def _get_dicom_slice_information(
        self, reference_dataset: Dataset
    ) -> Tuple[np.ndarray, Tuple, Tuple, Tuple]:
        reference_file_directory = os.path.dirname(self.parameters.reference_dicom_path)
        dicom_file_paths = [
            os.path.normpath(file) for file in glob(reference_file_directory + "/*.dcm")
        ]

        reference_series_instance_uid = reference_dataset.get_item(
            Tag(TAG_SERIES_INSTANCE_UID)
        ).value

        file_paths = list()
        positions = list()
        uids = list()
        slice_ids = list()
        slice_locations = list()
        for dicom_file_path in dicom_file_paths:
            try:
                candidate_dataset = dcmread(dicom_file_path, stop_before_pixels=True)
                candidate_dataset.decode()
                candidate_series_instance_uid = candidate_dataset.get_item(
                    Tag(TAG_SERIES_INSTANCE_UID)
                ).value

                if candidate_series_instance_uid != reference_series_instance_uid:
                    continue

                file_paths.append(dicom_file_path)
                positions.append(
                    candidate_dataset.get_item(Tag(TAG_IMAGE_POSITION_PATIENT)).value
                )
                slice_ids.append(
                    candidate_dataset.get_item(Tag(TAG_INSTANCE_NUMBER)).value
                )
                uids.append(candidate_dataset.get_item(Tag(TAG_SOP_INSTANCE_UID)).value)
                slice_locations.append(
                    candidate_dataset.get_item(Tag(TAG_SLICE_LOCATION)).value
                )

            except errors.InvalidDicomError:
                print(f"The file {dicom_file_path} is not a valid DICOM file!")

        if len(uids) == 0:
            raise ValueError(
                f"There are no other valid DICOM files provided "
                f"in the same folder as the reference DICOM image!"
            )

        slice_locations = np.array(slice_locations)
        sorted_indices = np.argsort(slice_locations, axis=0)
        sorted_positions = list()
        sorted_uids = list()
        sorted_file_paths = list()
        sorted_slice_ids = list()
        for idx in sorted_indices:
            sorted_positions.append(positions[idx])
            sorted_uids.append(uids[idx])
            sorted_file_paths.append(file_paths[idx])
            sorted_slice_ids.append(slice_ids[idx])

        sorted_positions = np.array(sorted_positions)
        sorted_uids = tuple(sorted_uids)
        sorted_file_paths = tuple(sorted_file_paths)
        sorted_slice_ids = tuple(sorted_slice_ids)

        return sorted_positions, sorted_uids, sorted_file_paths, sorted_slice_ids

    @staticmethod
    def _get_colors(num_colors: int):
        colors = list()
        for i in np.arange(0.0, 360.0, 360.0 / num_colors):
            hue = i / 360.0
            lightness = (50 + np.random.rand() * 10) / 100.0
            saturation = (90 + np.random.rand() * 10) / 100.0
            colors.append(
                [np.ceil(255 * i) for i in hls_to_rgb(hue, lightness, saturation)]
            )
        return colors

    @staticmethod
    def vector_angle(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        return np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        )

    @staticmethod
    def is_in(value: float, sequence: tuple, eps: float = 1e-3):
        indicator = [entry - eps <= value <= entry + eps for entry in sequence]
        return any(indicator)

    def _prepare_image_data(
        self, reference_image: sitk.Image, orientation: str
    ) -> Dict[int, Dict[str, Any]]:
        image_data = dict()
        for idx, path in enumerate(self.input_file_paths):
            image = BaseConverter.load_image(path)
            image = sitk.DICOMOrient(image, orientation)

            # TODO(Elias): Delete this if the other solution works
            # Check if flipping is necessary
            # direction_reference_image = reference_image.GetDirection()
            # direction_image = image.GetDirection()
            # direction_vector_angles = (self.vector_angle(direction_reference_image[0:3], direction_image[0:3]),
            #                            self.vector_angle(direction_reference_image[3:6], direction_image[3:6]),
            #                            self.vector_angle(direction_reference_image[6:9], direction_image[6:9]))
            # flip_indicator_direction_vector = sitk.VectorBool()
            # for angle in direction_vector_angles:
            #     if angle == -1.:
            #         flip_indicator_direction_vector.push_back(True)
            #     else:
            #         flip_indicator_direction_vector.push_back(False)
            # flip_indicator_direction = [angle == -1. for angle in direction_vector_angles]
            # if any(flip_indicator_direction):
            #     image = sitk.Flip(image, flip_indicator_direction_vector)

            # Shift the origins if they are just the image information long
            physical_size_image = tuple(
                np.array(image.GetSpacing())
                * (np.array(image.GetSize()) - np.array([1, 1, 1]))
            )
            origin_difference = tuple(
                np.array(image.GetOrigin()) - np.array(reference_image.GetOrigin())
            )
            if any(self.is_in(i, physical_size_image) for i in origin_difference):
                transform = sitk.TranslationTransform(image.GetDimension())
                offset = tuple(
                    np.array(image.GetOrigin()) - np.array(reference_image.GetOrigin())
                )
                transform.SetOffset(offset)
            else:
                transform = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)

            # Adjust the remaining geometrical differences if any
            if not all(
                (
                    image.GetOrigin() == reference_image.GetOrigin(),
                    image.GetSpacing() == reference_image.GetSpacing(),
                    image.GetDirection() == reference_image.GetDirection(),
                )
            ):
                resample = sitk.ResampleImageFilter()
                resample.SetOutputSpacing(reference_image.GetSpacing())
                resample.SetSize(reference_image.GetSize())
                resample.SetOutputPixelType(image.GetPixelIDValue())
                resample.SetOutputDirection(reference_image.GetDirection())
                resample.SetOutputOrigin(reference_image.GetOrigin())
                resample.SetTransform(transform)
                resample.SetDefaultPixelValue(0)
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
                image = resample.Execute(image)

            np_image = sitk.GetArrayFromImage(image).astype(np.int8)
            origin = tuple(image.GetOrigin())
            spacing = tuple(image.GetSpacing())
            direction = tuple(image.GetDirection())
            direction = [direction[i] for i in [0, 4, -1]]

            unique_label_ids = np.unique(np_image)
            unique_label_ids = [idx for idx in unique_label_ids if idx != 0]

            for lbl_id in unique_label_ids:
                # TODO(elias): Adjust here for a better mechanism
                name = list(self.structure_names.keys())[
                    list(self.structure_names.values()).index(lbl_id)
                ]
                if name == "Eye_L":
                    mask_image = (np_image == lbl_id).astype(np.int8)
                    mask_image += (
                        np_image == self.structure_names.get("Lens_L")
                    ).astype(np.int8)
                    mask_image += (
                        np_image == self.structure_names.get("Retina_L")
                    ).astype(np.int8)
                    mask_image = (mask_image >= 1).astype(np.int8)
                elif name == "Eye_R":
                    mask_image = (np_image == lbl_id).astype(np.int8)
                    mask_image += (
                        np_image == self.structure_names.get("Lens_R")
                    ).astype(np.int8)
                    mask_image += (
                        np_image == self.structure_names.get("Retina_R")
                    ).astype(np.int8)
                    mask_image = (mask_image >= 1).astype(np.int8)
                else:
                    mask_image = (np_image == lbl_id).astype(np.int8)

                label_data = {
                    PH_NAME: name,
                    PH_IMAGE: mask_image,
                    PH_ORIGIN: origin,
                    PH_SPACING: spacing,
                    PH_DIRECTION: direction,
                }
                if lbl_id in image_data.keys():
                    raise ValueError(f"An image with label {lbl_id} already exists!")
                image_data.update({lbl_id: label_data})

        colors = self._get_colors(len(self.structure_names))
        for lbl_id, data in image_data.items():
            data[PH_COLOR] = colors[lbl_id - 1]

        return image_data

    @staticmethod
    def get_dataset_direction(dataset: Dataset) -> Tuple[float, ...]:
        """Generates the full direction cosine vector from the DICOM meta data.

        Args:
            dataset: The DICOM dataset containing the orientation data (DICOM attribute: (0020,0037)).

        Returns:
            tuple: The full direction vector with the 9 direction cosines.
        """
        dataset_orientation = np.array(
            dataset.get_item(Tag(TAG_IMAGE_ORIENTATION_PATIENT)).value
        )
        orientation_vector_0 = dataset_orientation[:3]
        orientation_vector_1 = dataset_orientation[3:]
        orientation_vector_2 = np.cross(orientation_vector_0, orientation_vector_1)
        full_dataset_orientation = tuple(
            np.concatenate(
                (orientation_vector_2, orientation_vector_0, orientation_vector_1),
                axis=0,
            )
        )
        return full_dataset_orientation

    @staticmethod
    def reorient_image(
        image: sitk.Image, direction: Tuple[float, ...]
    ) -> Tuple[sitk.Image, str]:
        """Performs a reorientation of the input image to the direction specified.

        Args:
            image: The image to reorient.
            direction: The desired direction of the output image.

        Returns:
            sitk.Image: The reoriented image.
        """
        orientation_filter = sitk.DICOMOrientImageFilter()
        orientation_name = orientation_filter.GetOrientationFromDirectionCosines(
            direction
        )
        orientation_filter.SetDesiredCoordinateOrientation(orientation_name)
        reoriented_image = orientation_filter.Execute(image)
        return reoriented_image, orientation_name

    @staticmethod
    def load_image_series(file_paths: Tuple[str, ...]) -> sitk.Image:
        """Represents the loading of a series of sorted files as an image.

        Args:
            file_paths (tuple): The sorted file path to the files containing image information.

        Returns:
            sitk.Image: The image specified in the file paths.
        """
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        reader.LoadPrivateTagsOn()
        reader.MetaDataDictionaryArrayUpdateOn()
        return reader.Execute()

    def convert(self) -> Dataset:
        # Get the DICOM reference dataset and its properties
        reference_dataset = BaseDicomGenerator.load_dicom_metadata(
            self.parameters.reference_dicom_path
        )
        (
            ref_pos,
            ref_uids,
            ref_file_paths,
            ref_slice_ids,
        ) = self._get_dicom_slice_information(reference_dataset)

        # Get the DICOM reference file
        reference_image = self.load_image_series(ref_file_paths)

        # Adjust the image orientation of the reference image from the dataset
        reference_direction = self.get_dataset_direction(reference_dataset)
        reference_image, reference_orientation = self.reorient_image(
            reference_image, reference_direction
        )

        # Get the structures information
        structure_info = self._prepare_image_data(
            reference_image, reference_orientation
        )

        # Generate the basic RTSS dataset
        rtss_generator = BaseDicomRTGenerator(reference_dataset, self.parameters)
        new_dataset = rtss_generator.execute()

        # Generate the ROI Contour Sequence
        rcs_generator = ROIContourSequenceGenerator(
            structure_info, ref_uids, reference_dataset, ref_pos, self.parameters
        )
        new_dataset = rcs_generator.execute(new_dataset)

        # Generate the Structure Set ROI Sequence
        ssrs_generator = StructureSetROISequenceGeneratorBase(
            structure_info, reference_dataset, self.parameters
        )
        new_dataset = ssrs_generator.execute(new_dataset)

        # Generate the Coding Scheme Identification Sequence
        csis_generator = CodingSchemeIdentificationSequenceGenerator()
        new_dataset = csis_generator.execute(new_dataset)

        # Generate the Context Group Identification Sequence
        cgis_generator = ContextGroupIdentificationSequenceGenerator()
        new_dataset = cgis_generator.execute(new_dataset)

        # Generate the Mapping Resource Identification Sequence
        mris_generator = MappingResourceIdentificationSequenceGenerator()
        new_dataset = mris_generator.execute(new_dataset)

        # Generate the Referenced Frame Of Reference Sequence
        rfrs_generator = ReferencedFrameOfReferenceSequenceGeneratorBase(
            reference_dataset, ref_uids, self.parameters
        )
        new_dataset = rfrs_generator.execute(new_dataset)

        # Generate the RT ROI Observations Sequence
        rros_generator = RTROIObservationsSequenceGenerator(
            structure_info, self.parameters
        )
        new_dataset = rros_generator.execute(new_dataset)

        return new_dataset


class DicomToNiftiConverter(BaseNiftiConverter):
    def __init__(
        self,
        input_file_paths: Union[str, Tuple[str, ...]],
        full_image_identifier: str,
        registration_reference_file_dir: str,
    ) -> None:
        super().__init__(input_file_paths)
        self.reference_dataset = dcmread(input_file_paths[0])
        self.reference_dataset.decode()
        self.full_image_identifier = full_image_identifier
        self.registration_reference_file_dir = registration_reference_file_dir
        self.patient_name = self.reference_dataset.get(Tag(TAG_PATIENT_NAME)).value
        self.transform_matrices = []
        self.reg_other_reference_frame_of_reference_uid = []
        self.reg_referenced_series_instance_uid = []
        self.registration_files = []

    def get_matching_registration(
        self, registration_file_paths: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        image_series_instance_uid = str(
            self.reference_dataset.get_item(Tag(TAG_SERIES_INSTANCE_UID)).value
        )
        image_frame_of_reference_uid = str(
            self.reference_dataset.get_item(Tag(TAG_FRAME_OF_REFERENCE_UID)).value
        )
        matching_files = []
        for file_path in registration_file_paths:
            dataset = dcmread(file_path, stop_before_pixels=True)
            dataset.decode()

            if (
                str(dataset.get_item(Tag(TAG_SOP_CLASS_UID)).value)
                != DEFAULT_SOP_CLASS_UID_REGISTRATION
            ):
                continue

            matching_series_instance_uid = False
            matching_frame_of_reference_uid = False

            # Check Referenced Series Sequence
            referenced_series_sequence = dataset.get_item(
                Tag(TAG_REFERENCED_SERIES_SEQUENCE)
            ).value
            for referenced_series_item in referenced_series_sequence:
                referenced_series_instance_uid = str(
                    referenced_series_item.get_item(Tag(TAG_SERIES_INSTANCE_UID)).value
                )
                if referenced_series_instance_uid == image_series_instance_uid:
                    matching_series_instance_uid = True

            # Checking Studies Containing Other Referenced Instances Sequence
            other_studies_sequence = dataset.get_item(
                Tag(TAG_STUDIES_CONTAINING_OTHER_REFERENCED_INSTANCES_SEQUENCE)
            )
            if other_studies_sequence is not None:
                for other_studies_item in other_studies_sequence.value:
                    referenced_series_sequence = other_studies_item.get_item(
                        Tag(TAG_REFERENCED_SERIES_SEQUENCE)
                    ).value
                    for referenced_series_item in referenced_series_sequence:
                        referenced_series_instance_uid = (
                            referenced_series_item.get_item(
                                Tag(TAG_SERIES_INSTANCE_UID)
                            ).value
                        )
                        if referenced_series_instance_uid == image_series_instance_uid:
                            matching_series_instance_uid = True

            # Checking Registration Sequence
            registration_sequence = dataset.get_item(
                Tag(TAG_REGISTRATION_SEQUENCE)
            ).value
            for registration_item in registration_sequence:
                frame_of_reference_uid = str(
                    registration_item.get_item(Tag(TAG_FRAME_OF_REFERENCE_UID)).value
                )
                matrix_registration_sequence = registration_item.get_item(
                    Tag(TAG_MATRIX_REGISTRATION_SEQUENCE)
                ).value
                for matrix_registration_item in matrix_registration_sequence:
                    matrix_sequence = matrix_registration_item.get_item(
                        Tag(TAG_MATRIX_SEQUENCE)
                    ).value
                    for matrix_item in matrix_sequence:
                        transformation_type = matrix_item.get_item(
                            Tag(TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE)
                        ).value
                        transformation_matrix = tuple(
                            matrix_item.get_item(
                                Tag(TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX)
                            ).value
                        )
                        criteria = (
                            transformation_type == "RIGID",
                            transformation_matrix
                            != (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
                            frame_of_reference_uid == image_frame_of_reference_uid,
                        )
                        if all(criteria):
                            matching_frame_of_reference_uid = True

            # Assign the file if it fulfills the criteria
            matching_files.append(
                file_path
            ) if matching_series_instance_uid and matching_frame_of_reference_uid else None

        return tuple(set(matching_files))

    def set_registrations(self, registration_file_paths: Tuple[str, ...]) -> None:
        image_frame_of_reference_uid = str(
            self.reference_dataset.get_item(Tag(TAG_FRAME_OF_REFERENCE_UID)).value
        )
        for file_path in registration_file_paths:
            dataset = dcmread(file_path)
            dataset.decode()

            if (
                str(dataset.get_item(Tag(TAG_SOP_CLASS_UID)).value)
                != DEFAULT_SOP_CLASS_UID_REGISTRATION
            ):
                raise ValueError(
                    f"The registration file {file_path} is not a valid registration file!"
                )

            is_valid_registration_file = False
            registration_sequence = dataset.get_item(
                Tag(TAG_REGISTRATION_SEQUENCE)
            ).value
            for item in registration_sequence:
                frame_of_reference_uid = item.get_item(
                    Tag(TAG_FRAME_OF_REFERENCE_UID)
                ).value
                if frame_of_reference_uid != image_frame_of_reference_uid:
                    continue
                is_valid_registration_file = True

                matrix_registration_sequence = item.get_item(
                    Tag(TAG_MATRIX_REGISTRATION_SEQUENCE)
                ).value
                for mrs_item in matrix_registration_sequence:
                    matrix_sequence = mrs_item.get_item(Tag(TAG_MATRIX_SEQUENCE)).value
                    for ms_item in matrix_sequence:
                        transform_type = ms_item.get_item(
                            Tag(TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE)
                        ).value
                        if transform_type != "RIGID":
                            raise ValueError(
                                "The mechanism is only implemented for rigid transformations!"
                            )
                        transform_matrix = np.array(
                            ms_item.get_item(
                                Tag(TAG_FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX)
                            ).value
                        )
                        if all(
                            np.equal(
                                transform_matrix,
                                np.array(
                                    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
                                ),
                            )
                        ):
                            is_valid_registration_file = False
                        else:
                            self.transform_matrices.append(tuple(transform_matrix))

            if is_valid_registration_file:
                for item in registration_sequence:
                    frame_of_reference_uid = item.get_item(
                        Tag(TAG_FRAME_OF_REFERENCE_UID)
                    ).value
                    if frame_of_reference_uid != image_frame_of_reference_uid:
                        self.reg_other_reference_frame_of_reference_uid.append(
                            str(frame_of_reference_uid)
                        )
                studies_containing_other_referenced_instances_sequence = (
                    dataset.get_item(
                        Tag(TAG_STUDIES_CONTAINING_OTHER_REFERENCED_INSTANCES_SEQUENCE)
                    )
                )
                if studies_containing_other_referenced_instances_sequence is not None:
                    for (
                        other_sequence_item
                    ) in studies_containing_other_referenced_instances_sequence.value:
                        referenced_series_sequence = other_sequence_item.get_item(
                            Tag(TAG_REFERENCED_SERIES_SEQUENCE)
                        ).value
                        for referenced_serie in referenced_series_sequence:
                            self.reg_referenced_series_instance_uid.append(
                                str(
                                    referenced_serie.get_item(
                                        Tag(TAG_SERIES_INSTANCE_UID)
                                    ).value
                                )
                            )
                self.registration_files.append(file_path)
        error_criteria = (
            len(self.reg_other_reference_frame_of_reference_uid)
            != len(self.transform_matrices),
            len(self.reg_referenced_series_instance_uid)
            != len(self.transform_matrices),
        )
        if any(error_criteria):
            raise ValueError(
                "There is something wrong with the definition of the registration"
            )

    def _apply_transformations(self, image: sitk.Image) -> sitk.Image:
        internal_image = sitk.Image(image)
        for matrix, referenced_other_frame_of_reference, registration_file in zip(
            self.transform_matrices,
            self.reg_other_reference_frame_of_reference_uid,
            self.registration_files,
        ):

            # Get the image statistics
            statistics_filter = sitk.StatisticsImageFilter()
            statistics_filter.Execute(internal_image)
            minimum_intensity = statistics_filter.GetMinimum()

            # Build the transformation
            internal_matrix = np.array(matrix).reshape(4, 4)
            rotation_matrix = internal_matrix[:3, :3]
            transform = sitk.AffineTransform(3)
            rotation_vector = sitk.VectorDouble()
            [
                rotation_vector.push_back(float(entry))
                for entry in rotation_matrix.flatten()
            ]
            transform.SetMatrix(rotation_vector)
            translation_vector = sitk.VectorDouble()
            [
                translation_vector.push_back(float(entry))
                for entry in internal_matrix[:3, 3].flatten()
            ]
            transform.SetTranslation(translation_vector)

            # Build the output direction
            output_direction = (
                np.dot(rotation_matrix, np.array(image.GetDirection()).reshape(3, 3))
                .flatten()
                .tolist()
            )

            # Resample the image
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetOutputDirection(output_direction)
            resample_filter.SetOutputOrigin(transform.TransformPoint(image.GetOrigin()))
            resample_filter.SetOutputSpacing(image.GetSpacing())
            resample_filter.SetSize(image.GetSize())
            resample_filter.SetInterpolator(sitk.sitkBSpline)
            resample_filter.SetOutputPixelType(image.GetPixelIDValue())
            resample_filter.SetDefaultPixelValue(minimum_intensity)
            resample_filter.SetTransform(transform.GetInverse())
            internal_image = resample_filter.Execute(internal_image)
        return internal_image

    def get_file_name(self) -> str:
        return f"img_{self.patient_name}_{self.full_image_identifier}.nii.gz"

    def convert(self) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(
            os.path.dirname(self.input_file_paths[0]),
            self.reference_dataset.get_item(Tag(TAG_SERIES_INSTANCE_UID)).value,
        )
        reader.SetFileNames(files)
        reader.LoadPrivateTagsOn()
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()

        if self.transform_matrices:
            image = self._apply_transformations(image)
        return image


# TODO Correct and finalize this stuff such that a complete image is generated
class DicomRtToNiftiConverter(BaseNiftiConverter):
    def __init__(
        self, input_file_paths: Union[str, Tuple[str, ...]], reference_dicom_image: str
    ) -> None:
        super().__init__(input_file_paths)
        if len(self.input_file_paths) > 1:
            raise ValueError(
                f"There should be at maximum one input file path, but {len(self.input_file_paths)} paths"
                f"are provided!"
            )
        self.rtss_dataset = dcmread(self.input_file_paths[0], force=True)
        self.rtss_dataset.decode()

        if (
            self.rtss_dataset.get_item(Tag(TAG_SOP_CLASS_UID)).value
            != DEFAULT_SOP_CLASS_UID_RTSS
        ):
            raise ValueError(
                f"The file {self.input_file_paths[0]} is not a DICOM RTSS file!"
            )

        self.reference_image_file = reference_dicom_image
        self.reference_image_dataset = dcmread(
            reference_dicom_image, force=True, stop_before_pixels=True
        )
        self.reference_image_dataset.decode()

        reader = sitk.ImageSeriesReader()
        image_files = reader.GetGDCMSeriesFileNames(
            os.path.dirname(reference_dicom_image),
            self.reference_image_dataset.get_item(Tag(TAG_SERIES_INSTANCE_UID)).value,
        )
        reader.SetFileNames(image_files)
        reader.LoadPrivateTagsOn()
        reader.MetaDataDictionaryArrayUpdateOn()
        self.reference_image = reader.Execute()

    @staticmethod
    def get_contour_sequence_data(dataset):
        contour_sequence = dataset.get_item(Tag(TAG_CONTOUR_SEQUENCE)).value
        number_lines = len(contour_sequence)
        count = 0
        starts = []

        for contour_sequence_item in contour_sequence:
            contour_data = contour_sequence_item.get_item(Tag(TAG_CONTOUR_DATA)).value
            starts.append(int(count / 3))
            count += len(contour_data)

        number_points = int(count / 3)
        return number_points, number_lines, starts

    def get_poly_lines_from_roi_contour_sequence_v2(self):
        roi_contour_sequence = self.rtss_dataset.get_item(
            Tag(TAG_ROI_CONTOUR_SEQUENCE)
        ).value
        poly_data_lines = []
        referenced_roi_numbers = []
        rotation_matrix = np.linalg.inv(
            np.array(self.reference_image.GetDirection()).reshape(3, 3)
        )

        for roi_contour_item in roi_contour_sequence:
            referenced_roi_number = int(
                roi_contour_item.get(Tag(TAG_REFERENCED_ROI_NUMBER)).value
            )
            referenced_roi_numbers.append(referenced_roi_number)
            n_pts, n_lines, starts = self.get_contour_sequence_data(roi_contour_item)

            contour_sequence = roi_contour_item.get_item(
                Tag(TAG_CONTOUR_SEQUENCE)
            ).value
            for contour_item in contour_sequence:
                contour_data = contour_item.get_item(Tag(TAG_CONTOUR_DATA)).value
                num_points = len(contour_data)

    def get_poly_lines_from_roi_contour_sequence(self):
        roi_contour_sequence = self.rtss_dataset.get_item(
            Tag(TAG_ROI_CONTOUR_SEQUENCE)
        ).value

        poly_data_lines = []
        referenced_roi_numbers = []

        rotation_matrix = np.linalg.inv(
            np.array(self.reference_image.GetDirection()).reshape(3, 3)
        )

        for item in roi_contour_sequence:
            referenced_roi_number = int(item.get(Tag(TAG_REFERENCED_ROI_NUMBER)).value)
            referenced_roi_numbers.append(referenced_roi_number)
            n_pts, n_lines, starts = self.get_contour_sequence_data(item)

            points = vtk.vtkPoints()
            for c in item.ContourSequence:
                n = len(c.ContourData)
                control_pts = []
                for i in range(0, n, 3):
                    pt = np.array(
                        [
                            float(c.ContourData[i]),
                            float(c.ContourData[i + 1]),
                            float(c.ContourData[i + 2]),
                        ]
                    )
                    pt = np.dot(rotation_matrix, pt)
                    control_pts.append(pt)
                    # points.InsertNextPoint(pt)
                control_pts = np.stack(control_pts)
                print(
                    f"Contour Data:"
                    f"\t\tz-Min:{np.min(control_pts[:,2]):.3f}"
                    f"\t\tz-Max:{np.max(control_pts[:, 2]):.3f}"
                    f"\t\tSpan:{np.abs(np.max(control_pts[:, 2])-np.min(control_pts[:, 2])):.3f}"
                )
                mean_z = np.round(np.mean(control_pts[:, 2]), 2)
                for pt in control_pts:
                    pt[2] = mean_z
                    points.InsertNextPoint(pt)

            lines = vtk.vtkCellArray()

            i = 0
            for c in item.ContourSequence:
                n = int(len(c.ContourData) / 3)
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(n + 1)
                for j in range(n):
                    line.GetPointIds().SetId(j, j + starts[i])
                line.GetPointIds().SetId(n, starts[i])
                lines.InsertNextCell(line)
                i += 1

            linesPolyData = vtk.vtkPolyData()
            linesPolyData.SetPoints(points)
            linesPolyData.SetLines(lines)
            poly_data_lines.append(linesPolyData)

            # for contour_item in contour_sequence:
            #     number_of_contour_points = int(contour_item.get(Tag(TAG_NUMBER_OF_CONTOUR_POINTS)).value)
            #     contour_data = tuple([float(entry) for entry in contour_item.get(Tag(TAG_CONTOUR_DATA)).value])
            #     num_contour_data = int(len(contour_data) / 3)
            #     polyline = vtk.vtkPolyLine()
            #     polyline.GetPointIds().SetNumberOfIds(num_contour_data + 1)
            #     for contour_point_idx in range(0, len(contour_data), 3):
            #         polyline = vtk.vtkPolyLine()

        return tuple(poly_data_lines), tuple(referenced_roi_numbers)

    def _get_roi_names_to_roi_numbers(self):
        structure_set_roi_sequence = self.rtss_dataset.get(
            Tag(TAG_STRUCTURE_SET_ROI_SEQUENCE)
        ).value
        roi_assignments = {}
        for entry in structure_set_roi_sequence:
            roi_number = int(entry.get(Tag(TAG_ROI_NUMBER)).value)
            roi_name = str(entry.get(Tag(TAG_ROI_NAME)).value)
            roi_assignments.update({roi_number: roi_name})
        return roi_assignments

    def get_file_name(self, roi_name: str):
        return f"seg_{self.rtss_dataset.get(Tag(TAG_PATIENT_NAME)).value}_{roi_name}.nii.gz"

    def convert(self) -> Tuple[Tuple[sitk.Image, ...], Tuple[str, ...]]:
        # spacing = self.reference_image.GetSpacing()
        spacing = (1.0, 1.0, 1.0)
        roi_name_assignments = self._get_roi_names_to_roi_numbers()
        (
            polylines,
            referenced_roi_numbers,
        ) = self.get_poly_lines_from_roi_contour_sequence()

        images = []
        roi_names = []

        for polyline, roi_number in zip(polylines, referenced_roi_numbers):
            stencil = vtk.vtkPolyDataToImageStencil()
            stencil.SetInputData(polyline)

            bounds = polyline.GetBounds()
            extended_bounds = []
            print("Polyline bounds: ", bounds)
            for i, bound in enumerate(bounds):
                if (i % 2) == 0:
                    extended_bounds.append(math.floor(bound) - 1)
                else:
                    extended_bounds.append(math.ceil(bound) + 1)

            print("Volume bounds: ", extended_bounds)
            # extended_bounds = (-500, 500, -500, 500, -500, 500)
            stencil.SetOutputWholeExtent(*extended_bounds)
            stencil.SetOutputSpacing(*spacing)
            stencil.SetOutputOrigin(bounds[::2])
            stencil.Update()

            # TODO(Elias): This is a test
            # white_image = vtk.vtkImageData()
            # imgstenc = vtk.vtkImageStencil()
            # imgstenc.SetInputData(white_image)
            # imgstenc.SetStencilData(pol2stenc.GetOutput())
            # imgstenc.ReverseStencilOff()
            # imgstenc.SetBackgroundValue(0.0)
            # imgstenc.Update()

            sten2img = vtk.vtkImageStencilToImage()
            sten2img.SetInputConnection(stencil.GetOutputPort())
            sten2img.SetOutsideValue(0)
            sten2img.SetInsideValue(1)
            sten2img.Update()
            vtkimg = sten2img.GetOutput()

            bmin = extended_bounds[::2]
            print("Volume min: ", bmin)
            vtkimg.SetOrigin(bmin)
            images.append(vtkimg)
            roi_names.append(roi_name_assignments.get(roi_number))

        sitk_images = []
        for image_vtk in images:
            point_data = image_vtk.GetPointData()
            array = vtk_to_numpy(point_data.GetScalars())
            array = array.reshape(-1)
            is_vector = point_data.GetScalars().GetNumberOfComponents() != 1
            dims = list(image_vtk.GetDimensions())
            if is_vector and dims[-1] == 1:
                # 2D
                dims = dims[:2]
                dims.reverse()
                dims.append(point_data.GetScalars().GetNumberOfComponents())
            else:
                dims.reverse()
            array.shape = tuple(dims)
            image = itk.image_view_from_array(array, is_vector)

            dim = image.GetImageDimension()
            spacing = [1.0] * dim
            spacing[:dim] = image_vtk.GetSpacing()[:dim]
            image.SetSpacing(spacing)
            origin = [0.0] * dim
            origin[:dim] = image_vtk.GetOrigin()[:dim]
            image.SetOrigin(origin)
            # Todo: Add Direction with VTK 9

            image_sitk = sitk.GetImageFromArray(
                itk.GetArrayFromImage(image), isVector=False
            )
            image_sitk.SetOrigin(tuple(image.GetOrigin()))
            image_sitk.SetSpacing(tuple(image.GetSpacing()))
            image_sitk.SetDirection(
                itk.GetArrayFromMatrix(image.GetDirection()).flatten()
            )
            sitk_images.append(image_sitk)

        return tuple(sitk_images), tuple(roi_names)
