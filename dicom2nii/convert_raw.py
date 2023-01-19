import os
import enum
import SimpleITK as sitk
import numpy as np
import shutil
from typing import Union, Callable, Tuple, List

import pymia.data as pymia_data
import pymia.data.conversion as conv
import pymia.data.creation as pymia_crt
import pymia.data.loading as pymia_load
import pymia.data.transformation as pymia_tfm
import pymia.data.creation.fileloader as file_load


class FileTypes(enum.Enum):
    CT = 1  # The CT
    LB = 2  # The mask / label


class LoadData(file_load.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            Tuple[np.ndarray, Union[conv.ImageProperties, None]]:

        if category == 'images':
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


class Subject(pymia_data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject=subject,
                         images={FileTypes.CT.name: files[FileTypes.CT]},
                         labels={FileTypes.LB.name: files[FileTypes.LB]})
        self.subject_path = files.get(subject, '')


class DataSetFilePathGenerator(pymia_load.FilePathGenerator):
    """Represents a brain image file path generator.

    The generator is used to convert a human readable image identifier to an image file path,
    which allows to load the image.
    """

    def __init__(self):
        """Initializes a new instance of the DataSetFilePathGenerator class."""
        pass

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for an image.
        Args:
            id_ (str): The image identification.
            root_dir (str): The image' root directory.
            file_key (object): A human readable identifier used to identify the image.
            file_extension (str): The image' file extension.
        Returns:
            str: The images' full file path.
        """
        add_file_extension = True

        files_in_dir = os.listdir(root_dir)

        if file_key == FileTypes.CT:

            for file in files_in_dir:
                if '_CT' in file:
                    file_name = file

        elif file_key == FileTypes.LB:

            for file in files_in_dir:
                if '_LB' in file:
                    file_name = file

        else:
            raise ValueError('Unknown key')

        # file_name = id_ + file_name + file_extension if add_file_extension else file_name
        return os.path.join(root_dir, file_name)


class DirectoryFilter(pymia_load.DirectoryFilter):
    """Represents a data directory filter."""

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: List[str]) -> List[str]:
        """Filters a list of directories.
        Args:
            dirs (List[str]): A list of directories.
        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return sorted(dirs)


class RawConverter:

    def __init__(self, data_root_path_imgs: str, data_root_path_masks: str,
                 target_root_path_imgs: str, target_root_path_masks: str, extension: str = '.nii.gz') -> None:
        super().__init__()

        self.data_root_path_imgs = os.path.expanduser(data_root_path_imgs)
        self.data_root_path_masks = os.path.expanduser(data_root_path_masks)
        self.target_root_path_imgs = os.path.expanduser(target_root_path_imgs)
        self.target_root_path_masks = os.path.expanduser(target_root_path_masks)
        self.ext = extension

        # Check target path
        if not os.path.exists(self.target_root_path_imgs):
            os.mkdir(self.target_root_path_imgs)
        if not os.path.exists(self.target_root_path_masks):
            os.mkdir(self.target_root_path_masks)

    def get_image_filenames(self, crit1: str = '.mhd', crit2: str = '.raw') -> Tuple[list, list, list]:

        # Get all filenames
        filenames = os.listdir(self.data_root_path_imgs)

        # Filter the filenames according to the criteria
        filenames_crit1 = list()
        filenames_crit2 = list()

        for filename in filenames:

            _, extension = os.path.splitext(filename)

            if extension == crit1:
                filenames_crit1.append(filename)
            elif extension == crit2:
                filenames_crit2.append(filename)

        return filenames, filenames_crit1, filenames_crit2

    def get_mask_filenames(self, crit1: str = '.mhd', crit2: str = '.zraw') -> Tuple[list, list, list]:

        # Get all filenames
        filenames = os.listdir(self.data_root_path_masks)

        # Filter the filenames according to the criteria
        filenames_crit1 = list()
        filenames_crit2 = list()

        for filename in filenames:

            _, extension = os.path.splitext(filename)

            if extension == crit1:
                filenames_crit1.append(filename)
            elif extension == crit2:
                filenames_crit2.append(filename)

        return filenames, filenames_crit1, filenames_crit2

    def convert_image(self, filename: str):
        # Get the basename of the file
        basename, _ = os.path.splitext(filename)

        # Report the progress
        print('Converting the image:\t %s ...' % basename)

        # Read the image
        image = sitk.ReadImage(os.path.join(self.data_root_path_imgs, filename))

        # Write the image with the correct extension
        sitk.WriteImage(image, os.path.join(self.target_root_path_imgs, basename) + '_CT' + self.ext)

    def convert_mask(self, filename: str):
        # Get the basename of the file
        basename, _ = os.path.splitext(filename)

        # Report the progress
        print('Converting the mask:\t %s ...' % basename)

        # Read the mask
        mask = sitk.ReadImage(os.path.join(self.data_root_path_masks, filename))

        # Write the mask with the correct extension
        sitk.WriteImage(mask, os.path.join(self.target_root_path_masks, basename) + '_LB' + self.ext)

    def sort_files(self):

        # Report progress
        print('Sorting of the files...')

        # Get the list of all image and mask filenames
        filenames_ct = list()
        filenames_lb = list()

        filenames = os.listdir(self.target_root_path_imgs)

        for filename in filenames:

            if '_CT' in filename:
                filenames_ct.append(filename)
            elif '_LB' in filename:
                filenames_lb.append(filename)

        # Sort the entries to have equal pairs
        filenames_ct.sort()
        filenames_lb.sort()

        assert len(filenames_lb) == len(filenames_ct), 'There must be an equal number of masks as images!'

        for i in range(len(filenames_ct)):

            # Generate a new directory
            dir_name = 'patient_' + str(i)
            dir_path = os.path.join(self.target_root_path_imgs, dir_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # Check the filenames
            name_ct = filenames_ct[i].split(sep='_')[0]
            name_lb = filenames_lb[i].split(sep='_')[0]

            if name_ct == name_lb:

                # Move the appropriate files
                path_ct = os.path.join(self.target_root_path_imgs, filenames_ct[i])
                path_lb = os.path.join(self.target_root_path_imgs, filenames_lb[i])

                shutil.move(path_ct, os.path.join(dir_path, filenames_ct[i]))
                shutil.move(path_lb, os.path.join(dir_path, filenames_lb[i]))

            else:
                print('CT image %s could not be assigned to a label' % filenames_ct[i])

    def convert(self):
        # Get the list of all image and mask filenames
        filenames_images, filenames_images_mhd, filenames_images_raw = self.get_image_filenames()
        filenames_masks, filenames_masks_mhd, filenames_masks_raw = self.get_mask_filenames()

        # Convert the images to nifti
        for filename in filenames_images_mhd:
            self.convert_image(filename)

        for filename in filenames_masks_mhd:
            self.convert_mask(filename)

        # Sort the files
        self.sort_files()


def main(hdf_file: str, data_dir: str, num_patients: int):

    # Adjust / expand the path to the data
    # (Simplification of the directory path for the user)
    data_dir = os.path.expanduser(data_dir)
    hdf_file = os.path.expanduser(hdf_file)

    # Define the keys
    keys = [FileTypes.CT, FileTypes.LB]
    crawler = pymia_load.FileSystemDataCrawler(data_dir,
                                               keys,
                                               DataSetFilePathGenerator(),
                                               DirectoryFilter(),
                                               '.nii.gz')
    # Initialize the subjects and add them to a subjects list
    subjects = [Subject(id_, file_dict) for id_, file_dict in crawler.data.items()]

    # Restrict the subjects list
    subjects = subjects[:num_patients]

    # Check if the HDF5 file is existing and remove it, when existing
    if os.path.exists(hdf_file):
        os.remove(hdf_file)

    # Creating the dataset
    with pymia_crt.get_writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)

        # normalize the images and unsqueeze the labels and mask.
        # Unsqueeze is needed due to the convention to have the number of channels as last dimension.
        transform = pymia_tfm.ComposeTransform([pymia_tfm.IntensityNormalization(loop_axis=0, entries=('images',)),
                                                pymia_tfm.IntensityRescale(lower=0, upper=1, loop_axis=0, entries=('images',)),
                                                pymia_tfm.UnSqueeze(entries=('labels',))
                                                ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


if __name__ == '__main__':

    target_root_path = '../CapsNet_Slice/dataset/LUNA16/imgs/'
    dataset_path = '../CapsNet_Slice/dataset/dataset_luna16.h5'
    number_of_patients_to_process = 15

    # Define a new rawConverter
    converter = RawConverter(data_root_path_imgs='../NetworkZoo/SegCaps/dataset/imgs/',
                             data_root_path_masks='../NetworkZoo/SegCaps/dataset/masks/',
                             target_root_path_imgs=target_root_path,
                             target_root_path_masks=target_root_path)

    # Convert the images and masks to nifti
    # converter.convert()

    # Perform the computation of the dataset
    print('Creating the dataset...')
    main(dataset_path, target_root_path, number_of_patients_to_process)



