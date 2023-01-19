import argparse
import enum
import os
import typing

import SimpleITK as sitk
import numpy as np

import pymia.data as pymia_data
import pymia.data.conversion as conv
import pymia.data.creation as pymia_crt
import pymia.data.loading as pymia_load
import pymia.data.transformation as pymia_tfm
import pymia.data.creation.fileloader as file_load

from NetworkZoo.NetworkManager.Utilities import N4ITKFieldBiasCorrection

class FileTypes(enum.Enum):
    T1c = 1  # The T1-weighted image with contrast agent
    T1w = 2  # The T1-weighted image
    T2 = 3  # The T2-weighted image
    FL = 4  # The FLAIR image
    LB = 5  # The labels


class LoadData(file_load.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:

        if category == 'images':
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


class Subject(pymia_data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject=subject,
                         images={FileTypes.T1c.name: files[FileTypes.T1c],
                                 FileTypes.T1w.name: files[FileTypes.T1w],
                                 FileTypes.T2.name: files[FileTypes.T2],
                                 FileTypes.FL.name: files[FileTypes.FL]},
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

        if file_key == FileTypes.T1c:
            file_name = '_T1c_reg_resampled_final'
        elif file_key == FileTypes.T1w:
            file_name = '_T1w_reg_resampled_final'
        elif file_key == FileTypes.T2:
            file_name = '_T2w_reg_resampled_final'
        elif file_key == FileTypes.FL:
            file_name = '_FLAIR_reg_resampled_final'
        elif file_key == FileTypes.LB:
            file_name = '_labelmask_all_final'
        else:
            raise ValueError('Unknown key')

        file_name = id_ + file_name + file_extension if add_file_extension else file_name
        return os.path.join(root_dir, file_name)


class DirectoryFilter(pymia_load.DirectoryFilter):
    """Represents a data directory filter."""

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: typing.List[str]) -> typing.List[str]:
        """Filters a list of directories.
        Args:
            dirs (List[str]): A list of directories.
        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return sorted(dirs)


def main(hdf_file: str, data_dir: str):

    # Adjust / expand the path to the data
    data_dir = os.path.expanduser(data_dir)

    # Check the data directory
    try:
        content_data_dir = os.listdir(data_dir)
        if len(content_data_dir) == 0:
            raise ValueError('The data directory does not contain any directories')
        if not os.path.isdir(data_dir):
            raise ValueError('The data directory does not exist')
    except ValueError as e:
        print(e)

    # Check if the HDF5 file is existing and remove it, when existing
    if os.path.exists(hdf_file):
        print('The old HDF5 file will be removed / overwritten!')
        os.remove(hdf_file)

    # Define the keys of the different images and the labels
    keys = [FileTypes.T1c, FileTypes.T1w, FileTypes.T2, FileTypes.FL, FileTypes.LB]

    # Initialize the crawler
    crawler = pymia_load.FileSystemDataCrawler(data_dir,
                                               keys,
                                               DataSetFilePathGenerator(),
                                               DirectoryFilter(),
                                               '.nii.gz')

    # Initialize the subjects and add them to a subjects list
    subjects = [Subject(id_, file_dict) for id_, file_dict in crawler.data.items()]

    # Combine the data and save it as HDF5
    with pymia_crt.get_writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)

        # Normalize the images and unsqueeze the labels and mask
        transform = pymia_tfm.ComposeTransform(
            [N4ITKFieldBiasCorrection(),
             pymia_tfm.IntensityNormalization(loop_axis=None, entries=('images',)),
             pymia_tfm.UnSqueeze(entries=('labels',))
             ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


if __name__ == '__main__':
    'This script generates a new dataset including a N4ITK normalization of each image'

    parser = argparse.ArgumentParser(description='Dataset creation')

    # Add the path to the parser, where the HDF5 file should be stored
    parser.add_argument(
        '--hdf_file',
        type=str,
        default='dataset/dataset_2019_N4ITK.h5',
        help='Path to the dataset file.'
    )

    # Add the path to the parser, where to find the unprocessed data
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data_store/ground_truth/',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)