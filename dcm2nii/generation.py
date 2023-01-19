import itertools
import os
from typing import (Tuple, Dict, Any, Union)
import SimpleITK as sitk
import itk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from skimage import measure
from copy import deepcopy
from skimage.util import img_as_uint
from pydicom import dcmread
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat
from pydicom.sequence import Sequence
from pydicom.dataset import (FileMetaDataset, Tag, Dataset)
from pydicom.uid import (ImplicitVRLittleEndian)

from .parameters import *


class Generator(ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_data_element(dataset: Dataset, key: str, vr: str, value: Any) -> None:
        dataset.add_new(Tag(key), vr, value)

    @staticmethod
    def add_data_elements(dataset: Dataset, keys: Tuple[str, ...], vrs: Tuple[str, ...], values: Tuple[Any, ...]) \
            -> None:
        default_length = len(keys)
        if not all(len(entry) == default_length for entry in (vrs, values)):
            raise ValueError(f'The number of keys, value representations, and values must be equal!')
        for key, vr, value in zip(keys, vrs, values):
            Generator.add_data_element(dataset, key, vr, value)

    @staticmethod
    def get_dataset_with_data_element(key: str, vr: str, value: Any) -> Dataset:
        ds = Dataset()
        Generator.add_data_element(ds, key, vr, value)
        return ds

    @staticmethod
    def get_dataset_with_data_elements(keys: Tuple[str, ...], vrs: Tuple[str, ...], values: Tuple[Any, ...]) \
            -> Dataset:
        ds = Dataset()
        Generator.add_data_elements(ds, keys, vrs, values)
        return ds

    @abstractmethod
    def execute(self, dataset: Union[Dataset, None] = None) -> Union[Dataset, Tuple[Dataset, ...]]:
        raise NotImplementedError


class BaseDicomGenerator(Generator):

    def __init__(self, params: BaseDicomConversionParameter):
        super().__init__()
        if not os.path.exists(params.reference_dicom_path):
            raise FileNotFoundError(f'The reference file {params.reference_dicom_path} does not exist!')
        if not params.reference_dicom_path.endswith('.dcm'):
            raise ValueError(f'The reference file {params.reference_dicom_path} is not a DICOM file!')
        self.parameters = params

    @staticmethod
    def _load_image(file_path: str, must_be_dicom: bool = False) -> sitk.Image:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'The file {file_path} is not existing!')
        if must_be_dicom and not file_path.endswith(".dcm"):
            raise ValueError(f'The file {file_path} is not a DICOM file!')

        if file_path.endswith('.dcm'):
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(file_path))
            reader.MetaDataDictionaryArrayUpdateOn()
        else:
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)

        reader.LoadPrivateTagsOn()
        reader.SetOutputPixelType(sitk.sitkInt16)
        return reader.Execute()

    @staticmethod
    def load_dicom_metadata(file_path: str) -> Dataset:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'The file {file_path} is not existing!')
        if not file_path.endswith('.dcm'):
            raise ValueError(f'The file {file_path} is not a DICOM file!')
        dataset = dcmread(file_path)
        dataset.decode()
        return dataset

    def _generate_meta_dataset(self) -> FileMetaDataset:
        meta = FileMetaDataset()
        meta.FileMetaInformationVersion = b'\x00\x01'
        meta.FileMetaInformationGroupLength = 204
        meta.MediaStorageSOPClassUID = self.parameters.sop_class_uid
        meta.MediaStorageSOPInstanceUID = self.parameters.sop_instance_uid
        meta.ImplementationClassUID = self.parameters.implementation_class_uid
        return meta

    @abstractmethod
    def execute(self, dataset: Union[Dataset, None] = None) -> Union[Dataset, Tuple[Dataset, ...]]:
        raise NotImplementedError


class BaseDicomImageGenerator(BaseDicomGenerator):

    def __init__(self, image: sitk.Image, params: DicomImageConversionParameter):
        super().__init__(params)
        self.image = image
        self.parameters = params

    @staticmethod
    def _orient_image_to_rai(image: sitk.Image) -> sitk.Image:
        ITK_COORDINATE_ORIENTATION_RAI = (2 << 0) + (5 << 8) + (8 << 16)

        # Cast from SimpleITK to ITK
        image_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(image), is_vector=False)
        image_itk.SetOrigin(image.GetOrigin())
        image_itk.SetSpacing(image.GetSpacing())
        image_itk.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [3] * 2)))
        itk_image_type = itk.Image[itk.template(image_itk)[1]]

        # Reorientation of the image
        orientation_filter = itk.OrientImageFilter[itk_image_type, itk_image_type].New()
        orientation_filter.SetUseImageDirection(True)
        orientation_filter.SetDesiredCoordinateOrientation(ITK_COORDINATE_ORIENTATION_RAI)
        orientation_filter.SetInput(image_itk)
        orientation_filter.Update()
        image_itk_reoriented = orientation_filter.GetOutput()

        # Cast back from ITK to SimpleITK
        image_sitk = sitk.GetImageFromArray(itk.GetArrayFromImage(image_itk_reoriented), isVector=False)
        image_sitk.SetOrigin(tuple(image_itk_reoriented.GetOrigin()))
        image_sitk.SetSpacing(tuple(image_itk_reoriented.GetSpacing()))
        image_sitk.SetDirection(itk.GetArrayFromMatrix(image_itk_reoriented.GetDirection()).flatten())
        return image_sitk

    @staticmethod
    def _build_slice_datasets(dataset: Dataset, image: sitk.Image, base_sop_instance_uid: str) -> Tuple[Dataset]:
        axis_idx = 0
        slice_coordinate_index_1 = 1
        slice_coordinate_index_2 = 2

        internal_image = sitk.Image(image)
        internal_image = BaseDicomImageGenerator._orient_image_to_rai(internal_image)
        np_image = sitk.GetArrayFromImage(internal_image)
        np_image /= np.max(np_image)
        # TODO get here a better adjustment of the grayscale values
        np_image *= 0.05
        np_image = img_as_uint(np_image)

        window_width = int(np.ceil(np.percentile(np_image, 99.8)) * 1.5)
        window_center = int(np.ceil(np.mean(np_image)))

        base_direction = tuple(internal_image.GetDirection())
        image_orientation_patient_sitk = [base_direction[idx] for idx in (0, 3, 6, 1, 4, 7)]
        rows = np_image.shape[slice_coordinate_index_1]
        columns = np_image.shape[slice_coordinate_index_2]
        base_slice_location = internal_image.GetOrigin()[axis_idx]
        base_slice_location_shift = internal_image.GetSpacing()[axis_idx]
        raw_spacing = internal_image.GetSpacing()
        spacing = np.array(raw_spacing).take(indices=(slice_coordinate_index_1, slice_coordinate_index_2),
                                             axis=0)

        Generator.add_data_elements(dataset,
                                    keys=(TAG_IMAGE_ORIENTATION_PATIENT,
                                          TAG_ROWS,
                                          TAG_COLUMNS,
                                          TAG_PIXEL_SPACING,
                                          TAG_WINDOW_CENTER,
                                          TAG_WINDOW_WIDTH),
                                    vrs=(VR_IMAGE_ORIENTATION_PATIENT,
                                         VR_ROWS,
                                         VR_COLUMNS,
                                         VR_PIXEL_SPACING,
                                         VR_WINDOW_CENTER,
                                         VR_WINDOW_WIDTH),
                                    values=(MultiValue(DSfloat, image_orientation_patient_sitk),
                                            rows,
                                            columns,
                                            MultiValue(DSfloat, spacing),
                                            window_center,
                                            window_width))

        datasets = []
        for slice_idx in range(np_image.shape[axis_idx]):
            slice_dataset = deepcopy(dataset)
            np_image_slice = np_image.take(indices=slice_idx, axis=axis_idx)

            sop_instance_uid = f'{base_sop_instance_uid}.{slice_idx + 100}.0'
            image_position_patient = internal_image.TransformIndexToPhysicalPoint([0, 0, slice_idx])
            Generator.add_data_elements(slice_dataset,
                                        keys=(TAG_SOP_INSTANCE_UID,
                                              TAG_INSTANCE_NUMBER,
                                              TAG_IMAGE_POSITION_PATIENT,
                                              TAG_SLICE_LOCATION,
                                              TAG_PIXEL_DATA),
                                        vrs=(VR_SOP_INSTANCE_UID,
                                             VR_INSTANCE_NUMBER,
                                             VR_IMAGE_POSITION_PATIENT,
                                             VR_SLICE_LOCATION,
                                             VR_PIXEL_DATA),
                                        values=(sop_instance_uid,
                                                slice_idx + 1,
                                                MultiValue(DSfloat, image_position_patient),
                                                base_slice_location + base_slice_location_shift * slice_idx,
                                                np_image_slice.tobytes()))
            datasets.append(slice_dataset)
        return tuple(datasets)

    def execute(self, dataset: Union[Dataset, None] = None) -> Tuple[Dataset]:
        dataset = Dataset() if dataset is None else dataset

        Generator.add_data_elements(dataset,
                                    keys=(TAG_CONTENT_DATE,
                                          TAG_CONTENT_TIME,
                                          TAG_SERIES_INSTANCE_UID),
                                    vrs=(VR_CONTENT_DATE,
                                         VR_CONTENT_TIME,
                                         VR_SERIES_INSTANCE_UID),
                                    values=(self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.series_instance_uid))

        datasets = self._build_slice_datasets(dataset, self.image, self.parameters.sop_instance_uid)
        return datasets


class ReferencedImageSequenceGenerator(Generator):

    def __init__(self, referenced_sop_class_uids: Tuple[str, ...], referenced_sop_instance_uids: Tuple[str, ...]) \
            -> None:
        super().__init__()
        if len(referenced_sop_class_uids) != len(referenced_sop_instance_uids):
            raise ValueError(f'The number of entries for referenced_sop_class_uids must be equal to the number of '
                             f'entries for referenced_sop_instance_uids!')
        self.referenced_sop_class_uids = referenced_sop_class_uids
        self.referenced_sop_instance_uids = referenced_sop_instance_uids

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        sequence = Sequence()
        for sop_class_uid, sop_instance_uid in zip(self.referenced_sop_class_uids, self.referenced_sop_instance_uids):
            ds_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_REFERENCED_SOP_CLASS_UID,
                      TAG_REFERENCED_SOP_INSTANCE_UID),
                vrs=(VR_REFERENCED_SOP_CLASS_UID,
                     VR_REFERENCED_SOP_INSTANCE_UID),
                values=(sop_class_uid,
                        sop_instance_uid))
            sequence.append(ds_item)
        Generator.add_data_element(dataset, TAG_REFERENCED_IMAGE_SEQUENCE, VR_REFERENCED_IMAGE_SEQUENCE, sequence)
        return dataset


class BaseDicomRTGenerator(BaseDicomGenerator):

    def __init__(self, reference_dataset: Dataset, params: DicomRTConversionParameter):
        super().__init__(params)
        self.parameters = params
        self.reference_dataset = reference_dataset

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        ds_meta = self._generate_meta_dataset()
        ds = Dataset()
        ds.set_original_encoding(is_implicit_vr=True, is_little_endian=True, character_encoding='ISO_IR 192')
        ds.file_meta = ds_meta
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        # Assign the data to the DICOM tags
        Generator.add_data_elements(ds,
                                    keys=(TAG_SOP_CLASS_UID,
                                          TAG_SOP_INSTANCE_UID,
                                          TAG_SERIES_INSTANCE_UID,
                                          TAG_STUDY_INSTANCE_UID,
                                          TAG_SPECIFIC_CHARACTER_SET,
                                          TAG_MODALITY,
                                          TAG_MANUFACTURER,
                                          TAG_MANUFACTURER_MODEL_NAME,
                                          TAG_SERIES_DESCRIPTION,
                                          TAG_SERIES_NUMBER,
                                          TAG_SOFTWARE_VERSIONS,
                                          TAG_OPERATORS_NAME,
                                          TAG_ACCESSION_NUMBER,
                                          TAG_APPROVAL_STATUS,
                                          TAG_CONTENT_DATE,
                                          TAG_CONTENT_TIME,
                                          TAG_INSTANCE_CREATION_DATE,
                                          TAG_INSTANCE_CREATION_TIME,
                                          TAG_SERIES_DATE,
                                          TAG_SERIES_TIME,
                                          TAG_STRUCTURE_SET_DATE,
                                          TAG_STRUCTURE_SET_TIME,
                                          TAG_STUDY_DATE,
                                          TAG_STUDY_TIME,
                                          TAG_STATION_NAME,
                                          TAG_STUDY_DESCRIPTION,
                                          TAG_PATIENT_NAME,
                                          TAG_PATIENT_ID,
                                          TAG_PATIENT_SEX,
                                          TAG_PATIENT_BIRTH_DATE,
                                          TAG_DEVICE_SERIAL_NUMBER,
                                          TAG_STUDY_ID,
                                          TAG_STRUCTURE_SET_LABEL),
                                    vrs=(VR_SOP_CLASS_UID,
                                         VR_SOP_INSTANCE_UID,
                                         VR_SERIES_INSTANCE_UID,
                                         VR_STUDY_INSTANCE_UID,
                                         VR_SPECIFIC_CHARACTER_SET,
                                         VR_MODALITY,
                                         VR_MANUFACTURER,
                                         VR_MANUFACTURER_MODEL_NAME,
                                         VR_SERIES_DESCRIPTION,
                                         VR_SERIES_NUMBER,
                                         VR_SOFTWARE_VERSIONS,
                                         VR_OPERATORS_NAME,
                                         VR_ACCESSION_NUMBER,
                                         VR_APPROVAL_STATUS,
                                         VR_CONTENT_DATE,
                                         VR_CONTENT_TIME,
                                         VR_INSTANCE_CREATION_DATE,
                                         VR_INSTANCE_CREATION_TIME,
                                         VR_SERIES_DATE,
                                         VR_SERIES_TIME,
                                         VR_STRUCTURE_SET_DATE,
                                         VR_STRUCTURE_SET_TIME,
                                         VR_STUDY_DATE,
                                         VR_STUDY_TIME,
                                         VR_STATION_NAME,
                                         VR_STUDY_DESCRIPTION,
                                         VR_PATIENT_NAME,
                                         VR_PATIENT_ID,
                                         VR_PATIENT_SEX,
                                         VR_PATIENT_BIRTH_DATE,
                                         VR_DEVICE_SERIAL_NUMBER,
                                         VR_STUDY_ID,
                                         VR_STRUCTURE_SET_LABEL),
                                    values=(self.parameters.sop_class_uid,
                                            self.parameters.sop_instance_uid,
                                            self.parameters.series_instance_uid,
                                            self.reference_dataset.get(KEY_STUDY_INSTANCE_UID),
                                            self.reference_dataset.get(KEY_SPECIFIC_CHARACTER_SET),
                                            self.parameters.modality,
                                            self.parameters.manufacturer,
                                            self.parameters.manufacturer_model_name,
                                            self.parameters.series_description,
                                            self.parameters.series_number,
                                            self.parameters.software_version,
                                            self.parameters.operators_name,
                                            self.parameters.accession_number,
                                            self.parameters.approval_status,
                                            self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.content_date,
                                            self.parameters.content_time,
                                            self.parameters.station_name,
                                            self.reference_dataset.get(KEY_STUDY_DESCRIPTION),
                                            self.reference_dataset.get(KEY_PATIENT_NAME),
                                            self.reference_dataset.get(KEY_PATIENT_ID),
                                            self.reference_dataset.get(KEY_PATIENT_SEX),
                                            self.reference_dataset.get(KEY_PATIENT_BIRTH_DATE),
                                            self.parameters.device_serial_number,
                                            self.parameters.study_id,
                                            self.parameters.structure_set_label))
        return ds


class ROIContourSequenceGenerator(Generator):

    def __init__(self, structure_info: Dict[int, Dict[str, Any]], reference_uids: Tuple[str],
                 reference_dataset: Dataset, positions: np.ndarray, params: DicomRTConversionParameter,
                 smoothing: bool = True):
        super().__init__()
        self.structure_info = structure_info
        self.reference_uids = reference_uids
        self.reference_dataset = reference_dataset
        self.parameters = params
        self.minimal_number_points = 10
        self.reference_positions = positions
        self.smoothing = smoothing

    def _get_contour_data(self, mask_slice: np.ndarray, origin: Tuple[float, float, float],
                          spacing: Tuple[float, float, float], direction: Tuple[float, float, float],
                          slice_idx: int) -> Tuple[Tuple[float, ...], int]:
        internal_mask_slice = deepcopy(mask_slice)
        raw_points = tuple(measure.find_contours(internal_mask_slice, 0.99)[0])
        if len(raw_points) < self.minimal_number_points:
            return (), 0
        points = [(np.round(point[1] * spacing[0] * direction[0] + origin[0], 3),
                   np.round(point[0] * spacing[1] * direction[1] + origin[1], 3),
                   np.round(slice_idx * spacing[2] * direction[2] + origin[2], 3)) for point in raw_points]
        return tuple(itertools.chain.from_iterable(points)), len(raw_points)

    @staticmethod
    def show_image(image: vtk.vtkImageData, idx: int, axis: int = 0):
        # TODO: Simplify the imports -> vtk.util.numpy_support does not work
        np_image = vtk_to_numpy(image.GetPointData().GetScalars())
        from matplotlib import pyplot as plt
        plt.imshow(np_image.take(indices=idx, axis=axis))
        plt.show()

    @staticmethod
    def show_poly_data(polydata: vtk.vtkPolyData):
        colors = vtk.vtkNamedColors()

        # create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.SetWindowName('No Name')
        renWin.AddRenderer(ren)

        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        # actor
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper)
        actor1.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

        # assign actor to the renderer
        ren.AddActor(actor1)
        ren.SetBackground(colors.GetColor3d('MidnightBlue'))

        # enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()

    def _get_smoothed_contour_data(self, mask: np.ndarray, origin: Tuple[float, float, float],
                                   spacing: Tuple[float, float, float], direction: Tuple[float, float, float]):
        # TODO Implement this based on the idea of Varian
        # TODO Something is wrong with the locations
        internal_mask = deepcopy(mask)
        internal_mask = internal_mask.astype(np.uint8)
        input_mask_offset = np.array(origin)
        data_string = internal_mask.tostring()

        mask_import = vtk.vtkImageImport()
        mask_import.CopyImportVoidPointer(data_string, len(data_string))
        mask_import.SetDataScalarTypeToUnsignedChar()
        mask_import.SetNumberOfScalarComponents(1)
        mask_import.SetWholeExtent(0, internal_mask.shape[2] - 1,
                                   0, internal_mask.shape[1] - 1,
                                   0, internal_mask.shape[0] - 1)
        mask_import.SetDataExtentToWholeExtent()
        mask_import.SetDataSpacing(spacing[0], spacing[1], spacing[2])
        mask_import.SetDataOrigin(0, 0, 0)
        mask_import.Update()

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(mask_import.GetOutput())
        # dmc.SetInputConnection(mask_import.GetOutputPort())
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()

        # self.show_poly_data(dmc.GetOutput())

        decimate1 = vtk.vtkDecimatePro()
        # decimate1.SetInputConnection(dmc.GetOutputPort())
        decimate1.SetInputData(dmc.GetOutput())
        decimate1.SetTargetReduction(0.200)
        decimate1.BoundaryVertexDeletionOff()
        decimate1.Update()

        smoother1 = vtk.vtkSmoothPolyDataFilter()
        # smoother1.SetInputConnection(decimate1.GetOutputPort())
        smoother1.SetInputData(decimate1.GetOutput())
        smoother1.SetNumberOfIterations(int(500 / 2))  # TODO: Make this a setting
        smoother1.SetOutputPointsPrecision(0)  # use single precision float
        # smoother1.FeatureEdgeSmoothingOn()
        smoother1.Update()

        decimate2 = vtk.vtkDecimatePro()
        # decimate2.SetInputConnection(smoother1.GetOutputPort())
        decimate2.SetInputData(smoother1.GetOutput())
        decimate2.SetTargetReduction(0.300)
        decimate2.BoundaryVertexDeletionOff()
        decimate2.Update()

        smoother2 = vtk.vtkSmoothPolyDataFilter()
        # smoother2.SetInputConnection(decimate2.GetOutputPort())
        smoother2.SetInputData(decimate2.GetOutput())
        smoother2.SetNumberOfIterations(int(500 / 2))  # TODO: Make this a setting
        smoother2.SetOutputPointsPrecision(0)  # use single precision float
        # smoother2.FeatureEdgeSmoothingOn()
        smoother2.Update()

        mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(smoother2.GetOutputPort())
        mapper.SetInputData(smoother2.GetOutput())
        mapper.Update()
        structure_bounding_box = mapper.GetBounds()

        plane_source = vtk.vtkPlaneSource()
        plane = vtk.vtkPlane()
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetOutputPointsPrecision(0)  # single precision
        # cutter.SetInputConnection(smoother2.GetOutputPort())
        cutter.SetInputData(smoother2.GetOutput())
        cutter.Update()

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        # connectivity.SetInputConnection(cutter.GetOutputPort())
        connectivity.SetInputData(cutter.GetOutput())
        connectivity.SetOutputPointsPrecision(0)
        connectivity.ScalarConnectivityOff()
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.Update()

        stripper = vtk.vtkStripper()
        # stripper.SetInputConnection(connectivity.GetOutputPort())
        stripper.SetInputData(connectivity.GetOutput())
        stripper.SetMaximumLength(10000)
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()

        # normalize image orientation
        # TODO HERE IS A MASSIVE ERROR!!!!
        slice_pos = np.copy(self.reference_positions)
        slice_pos[:, 0] = direction[0] * self.reference_positions[:, 0]
        slice_pos[:, 1] = direction[1] * self.reference_positions[:, 1]
        slice_pos[:, 2] = direction[2] * self.reference_positions[:, 2]

        # z offset from segmentation mask
        nii_z_offset = input_mask_offset[2]

        # get z indices of slices with structure-specific contours
        z_indices = np.where((slice_pos[:, 2] >= structure_bounding_box[4] + nii_z_offset) &
                             (slice_pos[:, 2] <= structure_bounding_box[5] + nii_z_offset))

        # process all slices
        ss = []
        for z in z_indices[0]:
            # cut structure with current plane
            plane_source.SetCenter(0, 0, slice_pos[z][2] - nii_z_offset)
            plane_source.SetPoint1(500, 0, slice_pos[z][2] - nii_z_offset)
            plane_source.SetPoint2(0, 500, slice_pos[z][2] - nii_z_offset)
            plane_source.SetOrigin(0, 0, slice_pos[z][2] - nii_z_offset)
            plane_source.SetNormal(0, 0, 1)
            plane_source.Update()
            plane.SetOrigin(plane_source.GetOrigin())
            plane.SetNormal(plane_source.GetNormal())
            smoother1.Update()
            smoother2.Update()
            cutter.Update()

            # extract contour regions
            connectivity.InitializeSpecifiedRegionList()
            connectivity.AddSpecifiedRegion(0)
            connectivity.Modified()
            connectivity.Update()
            num_regions = connectivity.GetNumberOfExtractedRegions()

            if num_regions > 0:
                # regions = []

                # process all contour regions
                for i in range(0, num_regions):
                    connectivity.SetExtractionModeToSpecifiedRegions()
                    connectivity.InitializeSpecifiedRegionList()
                    connectivity.AddSpecifiedRegion(i)
                    connectivity.Modified()
                    connectivity.Update()
                    stripper.Update()

                    # process points in specific region
                    p = stripper.GetOutput().GetPoints()
                    if p is not None:
                        regions = []
                        p_d = p.GetData()
                        p_np = vtk_to_numpy(p_d)
                        # p_np = numpy_support.vtk_to_numpy(p_d)
                        lines = stripper.GetOutput().GetLines()

                        d = lines.GetData()
                        d_np = vtk_to_numpy(d)
                        # d_np = numpy_support.vtk_to_numpy(d)
                        d_np = np.delete(d_np, np.arange(0, d_np.size, 3))
                        pp = p_np[d_np, 0:3]

                        # downsampling points by 3x,
                        # TODO: make this a function of average contour point-to-point distance;
                        #  make sure there are at least 3 rows
                        # print(pp.shape)
                        if pp.shape[0] > 100:
                            pp = pp[::3, :]

                        # incorporate slice position and offset into point coordinates
                        pp[:, 0] += slice_pos[z][0]
                        pp[:, 1] += slice_pos[z][1]
                        pp[:, 2] += nii_z_offset

                        # normalize orientation
                        pp[:, 0] = direction[0] * pp[:, 0]
                        pp[:, 1] = direction[1] * pp[:, 1]
                        pp[:, 2] = direction[2] * pp[:, 2]

                        # update the regions and structure set lists
                        regions.append(pp)
                        flatten_pixels = list(itertools.chain.from_iterable(regions[0]))

                        current_slice = {'regions': regions,
                                         'pixels': flatten_pixels,
                                         'uid': self.reference_uids[z],
                                         'position': slice_pos[z]}

                        # current_slice = Slice()
                        # # current_slice.dcm_fname = fnames_list[z]
                        # # current_slice.pos = slice_pos[z]
                        # current_slice.regions = regions
                        # current_slice.pixels = flatten_pixels
                        # current_slice.uid = self.reference_uids[z]
                        ss.append(current_slice)
        return ss

    def _build_smoothed_contour_sequence(self, data: Dict[str, Any], referenced_sop_class_uid: str,
                                         contour_geometric_type: str) -> Sequence:
        sequence = Sequence()
        mask = data.get('image')  # type: np.ndarray

        contour_data_dict = self._get_smoothed_contour_data(mask, data.get('origin'), data.get('spacing'),
                                                            data.get('direction'))

        for entry in contour_data_dict:
            ds_contour_image_sequence = Generator.get_dataset_with_data_elements(
                (TAG_REFERENCED_SOP_CLASS_UID, TAG_REFERENCED_SOP_INSTANCE_UID),
                (VR_REFERENCED_SOP_CLASS_UID, VR_REFERENCED_SOP_INSTANCE_UID),
                (referenced_sop_class_uid, entry.get('uid')))

            ds_contour_sequence_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_CONTOUR_IMAGE_SEQUENCE,
                      TAG_CONTOUR_GEOMETRIC_TYPE,
                      TAG_NUMBER_OF_CONTOUR_POINTS,
                      TAG_CONTOUR_DATA),
                vrs=(VR_CONTOUR_IMAGE_SEQUENCE,
                     VR_CONTOUR_GEOMETRIC_TYPE,
                     VR_NUMBER_OF_CONTOUR_POINTS,
                     VR_CONTOUR_DATA),
                values=(Sequence([ds_contour_image_sequence]),
                        contour_geometric_type,
                        int(len(entry.get('pixels')) / 3),
                        MultiValue(DSfloat, tuple(entry.get('pixels')))))
            sequence.append(ds_contour_sequence_item)
        return sequence

    def _build_contour_sequence(self, data: Dict[str, Any], referenced_sop_class_uid: str,
                                contour_geometric_type: str) -> Sequence:
        sequence = Sequence()
        mask = data.get('image')  # type: np.ndarray
        axis_index = 0

        for slice_idx in range(mask.shape[axis_index]):
            if np.sum(mask.take(indices=slice_idx, axis=axis_index)) == 0:
                continue
            contour_data, num_contour_points = self._get_contour_data(mask.take(indices=slice_idx, axis=axis_index),
                                                                      data.get('origin'),
                                                                      data.get('spacing'),
                                                                      data.get('direction'),
                                                                      slice_idx)
            if num_contour_points == 0:
                continue

            ds_contour_image_sequence = Generator.get_dataset_with_data_elements(
                (TAG_REFERENCED_SOP_CLASS_UID, TAG_REFERENCED_SOP_INSTANCE_UID),
                (VR_REFERENCED_SOP_CLASS_UID, VR_REFERENCED_SOP_INSTANCE_UID),
                (referenced_sop_class_uid, self.reference_uids[slice_idx]))

            ds_contour_sequence_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_CONTOUR_IMAGE_SEQUENCE,
                      TAG_CONTOUR_GEOMETRIC_TYPE,
                      TAG_NUMBER_OF_CONTOUR_POINTS,
                      TAG_CONTOUR_DATA),
                vrs=(VR_CONTOUR_IMAGE_SEQUENCE,
                     VR_CONTOUR_GEOMETRIC_TYPE,
                     VR_NUMBER_OF_CONTOUR_POINTS,
                     VR_CONTOUR_DATA),
                values=(Sequence([ds_contour_image_sequence]),
                        contour_geometric_type,
                        num_contour_points,
                        MultiValue(DSfloat, contour_data)))
            sequence.append(ds_contour_sequence_item)
        return sequence

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        roi_contour_sequence = Sequence()
        contour_geometric_type = self.parameters.contour_geometric_type
        referenced_sop_class_uid = self.reference_dataset.get(Tag(TAG_SOP_CLASS_UID)).value

        for label, data in self.structure_info.items():
            if self.smoothing:
                contour_sequence = self._build_smoothed_contour_sequence(data, referenced_sop_class_uid,
                                                                         contour_geometric_type)
            else:
                contour_sequence = self._build_contour_sequence(data, referenced_sop_class_uid, contour_geometric_type)
            ds_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_ROI_DISPLAY_COLOR,
                      TAG_REFERENCED_ROI_NUMBER,
                      TAG_CONTOUR_SEQUENCE),
                vrs=(VR_ROI_DISPLAY_COLOR,
                     VR_REFERENCED_ROI_NUMBER,
                     VR_CONTOUR_SEQUENCE),
                values=(data.get('color'),
                        label,
                        contour_sequence))
            roi_contour_sequence.append(ds_item)
        Generator.add_data_element(dataset, TAG_ROI_CONTOUR_SEQUENCE, VR_ROI_CONTOUR_SEQUENCE, roi_contour_sequence)
        return dataset


class StructureSetROISequenceGeneratorBase(Generator):

    def __init__(self, structure_info: Dict[int, Dict[str, Any]], reference_dataset: Dataset,
                 params: DicomRTConversionParameter):
        super().__init__()
        self.structure_info = structure_info
        self.reference_dataset = reference_dataset
        self.parameters = params

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        sequence = Sequence()
        referenced_frame_of_reference_uid = self.reference_dataset.get(Tag(KEY_FRAME_OF_REFERENCE_UID)).value

        for label, data in self.structure_info.items():
            ds_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_ROI_NUMBER,
                      TAG_REFERENCED_FRAME_OF_REFERENCE_UID,
                      TAG_ROI_NAME,
                      TAG_ROI_GENERATION_ALGORITHM),
                vrs=(VR_ROI_NUMBER,
                     VR_REFERENCED_FRAME_OF_REFERENCE_UID,
                     VR_ROI_NAME,
                     VR_ROI_GENERATION_ALGORITHM),
                values=(label,
                        referenced_frame_of_reference_uid,
                        data.get('name'),
                        self.parameters.roi_generation_algorithm))
            sequence.append(ds_item)
        Generator.add_data_element(dataset, TAG_STRUCTURE_SET_ROI_SEQUENCE, VR_STRUCTURE_SET_ROI_SEQUENCE, sequence)
        return dataset


class CodingSchemeIdentificationSequenceGenerator(Generator):

    def __init__(self):
        super().__init__()

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        sequence = Sequence()

        ds_item_1 = Generator.get_dataset_with_data_elements(
            keys=(TAG_CODING_SCHEME_DESIGNATOR, TAG_CODING_SCHEME_UID),
            vrs=(VR_CODING_SCHEME_DESIGNATOR, VR_CODING_SCHEME_UID),
            values=('FMA', '2.16.840.1.113883.6.119'))
        sequence.append(ds_item_1)

        ds_item_2 = Generator.get_dataset_with_data_elements(
            keys=(TAG_CODING_SCHEME_DESIGNATOR,
                  TAG_CODING_SCHEME_UID,
                  TAG_CODING_SCHEME_NAME,
                  TAG_CODING_SCHEME_RESPONSIBLE_ORGANISATION),
            vrs=(VR_CODING_SCHEME_DESIGNATOR,
                 VR_CODING_SCHEME_UID,
                 VR_CODING_SCHEME_NAME,
                 VR_CODING_SCHEME_RESPONSIBLE_ORGANISATION),
            values=('99VMS_STRUCTCODE',
                    '1.2.246.352.7.3.10',
                    'Structure Codes',
                    'Varian Medical Systems'))
        sequence.append(ds_item_2)

        Generator.add_data_element(dataset, TAG_CODING_SCHEME_IDENTIFICATION_SEQUENCE,
                                   VR_CODING_SCHEME_IDENTIFICATION_SEQUENCE, sequence)
        return dataset


class ContextGroupIdentificationSequenceGenerator(Generator):

    def __init__(self):
        super().__init__()

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset

        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_MAPPING_RESOURCE,
                  TAG_CONTEXT_GROUP_VERSION,
                  TAG_CONTEXT_IDENTIFIER,
                  TAG_CONTEXT_UID),
            vrs=(VR_MAPPING_RESOURCE,
                 VR_CONTEXT_GROUP_VERSION,
                 VR_CONTEXT_IDENTIFIER,
                 VR_CONTEXT_UID),
            values=(DEFAULT_MAPPING_RESOURCE,
                    DEFAULT_CONTEXT_GROUP_VERSION,
                    DEFAULT_CONTEXT_IDENTIFIER,
                    DEFAULT_CONTEXT_UID))
        Generator.add_data_element(dataset, TAG_CONTEXT_GROUP_IDENTIFICATION_SEQUENCE,
                                   VR_CONTEXT_GROUP_IDENTIFICATION_SEQUENCE, Sequence([ds_item]))
        return dataset


class MappingResourceIdentificationSequenceGenerator(Generator):

    def __init__(self):
        super().__init__()

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_MAPPING_RESOURCE,
                  TAG_MAPPING_RESOURCE_UID,
                  TAG_MAPPING_RESOURCE_NAME),
            vrs=(VR_MAPPING_RESOURCE,
                 VR_MAPPING_RESOURCE_UID,
                 VR_MAPPING_RESOURCE_NAME),
            values=('99VMS',
                    '1.2.246.352.7.1.1',
                    'Varian Medical Systems'))
        Generator.add_data_element(dataset, TAG_MAPPING_RESOURCE_IDENTIFICATION_SEQUENCE,
                                   VR_MAPPING_RESOURCE_IDENTIFICATION_SEQUENCE, Sequence([ds_item]))
        return dataset


class ReferencedFrameOfReferenceSequenceGeneratorBase(BaseDicomGenerator):

    def __init__(self, reference_dataset: Dataset, reference_sop_instance_uids: Tuple[str],
                 params: DicomRTConversionParameter):
        super().__init__(params)
        self.reference_dataset = reference_dataset
        self.reference_sop_instance_uids = reference_sop_instance_uids
        self.parameters = params

    def _build_contour_image_sequence(self, referenced_sop_class_uid: str) -> Sequence:
        contour_image_sequence = Sequence()
        for entry in self.reference_sop_instance_uids:
            ds_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_REFERENCED_SOP_CLASS_UID,
                      TAG_REFERENCED_SOP_INSTANCE_UID),
                vrs=(VR_REFERENCED_SOP_CLASS_UID,
                     VR_REFERENCED_SOP_INSTANCE_UID),
                values=(referenced_sop_class_uid,
                        entry))
            contour_image_sequence.append(ds_item)
        return contour_image_sequence

    def _build_rt_referenced_series_sequence(self, reference_dataset: Dataset) -> Sequence:
        sequence = self._build_contour_image_sequence(reference_dataset.get(Tag(TAG_SOP_CLASS_UID)).value)
        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_SERIES_INSTANCE_UID,
                  TAG_CONTOUR_IMAGE_SEQUENCE),
            vrs=(VR_SERIES_INSTANCE_UID,
                 VR_CONTOUR_IMAGE_SEQUENCE),
            values=(reference_dataset.get(Tag(TAG_SERIES_INSTANCE_UID)).value,
                    sequence))
        return Sequence([ds_item])

    def _build_rt_referenced_study_sequence(self, reference_dataset: Dataset) -> Sequence:
        rt_referenced_series_sequence = self._build_rt_referenced_series_sequence(reference_dataset)
        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_REFERENCED_SOP_CLASS_UID,
                  TAG_REFERENCED_SOP_INSTANCE_UID,
                  TAG_RT_REFERENCED_SERIES_SEQUENCE),
            vrs=(VR_REFERENCED_SOP_CLASS_UID,
                 VR_REFERENCED_SOP_INSTANCE_UID,
                 VR_RT_REFERENCED_SERIES_SEQUENCE),
            values=(DEFAULT_SOP_CLASS_UID_RTSS,
                    reference_dataset.get(Tag(TAG_STUDY_INSTANCE_UID)).value,
                    rt_referenced_series_sequence))
        return Sequence([ds_item])

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        rt_referenced_study_sequence = self._build_rt_referenced_study_sequence(self.reference_dataset)

        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_FRAME_OF_REFERENCE_UID,
                  TAG_RT_REFERENCED_STUDY_SEQUENCE),
            vrs=(VR_FRAME_OF_REFERENCE_UID,
                 VR_RT_REFERENCED_STUDY_SEQUENCE),
            values=(self.reference_dataset.get(Tag(TAG_FRAME_OF_REFERENCE_UID)).value,
                    rt_referenced_study_sequence))
        Generator.add_data_element(dataset,
                                   TAG_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
                                   VR_REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
                                   Sequence([ds_item]))
        return dataset


class RTROIObservationsSequenceGenerator(Generator):

    def __init__(self, structure_info: Dict[int, Dict[str, Any]], params: DicomRTConversionParameter):
        super().__init__()
        self.structure_info = structure_info
        self.parameters = params

    def _build_rt_roi_identification_code_sequence(self, label_info: Dict[str, Any]) -> Sequence:
        info = self.parameters.structure_info.get(label_info.get('name'))

        ds_item = Generator.get_dataset_with_data_elements(
            keys=(TAG_CODE_VALUE,
                  TAG_CODING_SCHEME_DESIGNATOR,
                  TAG_CODING_SCHEME_VERSION,
                  TAG_CODE_MEANING,
                  TAG_MAPPING_RESOURCE,
                  TAG_CONTEXT_GROUP_VERSION,
                  TAG_CONTEXT_IDENTIFIER,
                  TAG_CONTEXT_UID,
                  TAG_MAPPING_RESOURCE_UID,
                  TAG_MAPPING_RESOURCE_NAME),
            vrs=(VR_CODE_VALUE,
                 VR_CODING_SCHEME_DESIGNATOR,
                 VR_CODING_SCHEME_VERSION,
                 VR_CODE_MEANING,
                 VR_MAPPING_RESOURCE,
                 VR_CONTEXT_GROUP_VERSION,
                 VR_CONTEXT_IDENTIFIER,
                 VR_CONTEXT_UID,
                 VR_MAPPING_RESOURCE_UID,
                 VR_MAPPING_RESOURCE_NAME),
            values=(info.get(KEY_CODE_VALUE),
                    info.get(KEY_CODING_SCHEME_DESIGNATOR),
                    info.get(KEY_CODING_SCHEME_VERSION),
                    info.get(KEY_CODE_MEANING),
                    info.get(KEY_MAPPING_RESOURCE),
                    info.get(KEY_CONTEXT_GROUP_VERSION),
                    info.get(KEY_CONTEXT_IDENTIFIER),
                    info.get(KEY_CONTEXT_UID),
                    info.get(KEY_MAPPING_RESOURCE_UID),
                    info.get(KEY_MAPPING_RESOURCE_NAME)))
        return Sequence([ds_item])

    def execute(self, dataset: Union[Dataset, None] = None) -> Dataset:
        dataset = Dataset() if dataset is None else dataset
        sequence = Sequence()
        for label_idx, label_info in self.structure_info.items():
            rt_roi_identification_code_sequence = self._build_rt_roi_identification_code_sequence(label_info)

            ds_item = Generator.get_dataset_with_data_elements(
                keys=(TAG_OBSERVATION_NUMBER,
                      TAG_REFERENCED_ROI_NUMBER,
                      TAG_ROI_OBSERVATION_LABEL,
                      TAG_RT_ROI_IDENTIFICATION_CODE_SEQUENCE,
                      TAG_RT_ROI_INTERPRETED_TYPE,
                      TAG_ROI_INTERPRETER),
                vrs=(VR_OBSERVATION_NUMBER,
                     VR_REFERENCED_ROI_NUMBER,
                     VR_ROI_OBSERVATION_LABEL,
                     VR_RT_ROI_IDENTIFICATION_CODE_SEQUENCE,
                     VR_RT_ROI_INTERPRETED_TYPE,
                     VR_ROI_INTERPRETER),
                values=(0,
                        label_idx,
                        label_info.get('name'),
                        rt_roi_identification_code_sequence,
                        DEFAULT_RT_ROI_INTERPRETED_TYPE,
                        DEFAULT_ROI_INTERPRETER))
            sequence.append(ds_item)
        Generator.add_data_element(dataset, TAG_RT_ROI_OBSERVATION_SEQUENCE, VR_RT_ROI_OBSERVATION_SEQUENCE, sequence)
        return dataset
