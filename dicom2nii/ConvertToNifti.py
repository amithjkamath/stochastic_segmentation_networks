import os
from typing import Union, Callable
from distutils.dir_util import copy_tree
import numpy as np
import numpy.ma as ma
import shutil
import re
import enum
import json
import random
import SimpleITK as sitk
import pymia.filtering.registration as reg
import subprocess
import itk

# import matplotlib.pyplot as plt


class LabelType(enum.Enum):
    """
    Enumerate to represent the different label structures
    """

    Long = 1  # The name of the organ, its direction and the operator
    Short = 2  # The name of the organ and the direction but no operator
    ExtraShort = 3  # The name of the organ but no direction or operator


class DicomNiftiConverter:
    def __init__(
        self,
        slicer_path: str,
        python_script_path: str,
        input_folder_path: Union[str, None],
        preprocessed_folder_path: str,
        output_folder_path: str,
        label_config_path: str,
        label_hierarchy: list,
        persons: list,
        ext: str,
        output_size: list = (256, 256, 192),
    ) -> None:
        super().__init__()

        # Set the path to the slicer application
        self.slicer_path = os.path.expanduser(slicer_path)

        # Set the path to the python script
        self.python_script_path = os.path.expanduser(python_script_path)

        # Set the path to the input folder
        self.input_folder_path = os.path.expanduser(input_folder_path)

        # Set the path to the output folder
        self.output_folder_path = os.path.expanduser(output_folder_path)

        # Set the preprocessed folder path
        self.preprocessed_folder_path = os.path.expanduser(preprocessed_folder_path)

        # Number of patients
        self.num_patients = 30

        # Define the operators
        self.persons = persons

        # Define the extension
        self.ext = ext

        # Define the label dict
        self.label_dict = None

        # Define the label hierarchy
        self.label_hierarchy = label_hierarchy

        # Define the label config file path
        self.label_config_path = label_config_path

        # Define the output size
        self.output_size = output_size

    def check_paths(self) -> None:
        """
        This function checks the paths of the which need to be present for the conversation process
        """

        # Check if the directories are available and generate them if not available
        if not os.path.exists(self.slicer_path):
            raise FileNotFoundError

        if not os.path.exists(self.input_folder_path):
            raise FileNotFoundError

        if not os.path.exists(self.preprocessed_folder_path):
            os.mkdir(self.preprocessed_folder_path)

        if os.path.exists(self.preprocessed_folder_path):
            shutil.rmtree(self.preprocessed_folder_path)
            os.mkdir(self.preprocessed_folder_path)

        if not os.path.exists(self.output_folder_path):
            os.mkdir(self.output_folder_path)

        if os.path.exists(self.output_folder_path):
            shutil.rmtree(self.output_folder_path)
            os.mkdir(self.output_folder_path)

        if not os.path.exists(self.python_script_path):
            raise FileNotFoundError

    def preprocess_folder_structure(
        self, segmentation_operator: Union[tuple, None]
    ) -> None:
        """
        This function preprocesses the folder structure for the conversation process

        Args:
            segmentation_operator       A list of the segmentation operators
        """

        # Check if the segmentation operators are provided
        if segmentation_operator is None:
            segmentation_operator = tuple(self.persons)

        # Copy the images
        copy_tree(
            self.input_folder_path,  # + "/Bratumia_patient`sMR_Anonymised/",
            self.preprocessed_folder_path,
        )

        # Get all the image folders in the preprocessed folder
        # image_folder_names = os.listdir(self.preprocessed_folder_path)

        # Rename the folder in the preprocessed folder
        # for image_folder_name in image_folder_names:
        #    os.rename(
        #        os.path.join(self.preprocessed_folder_path, image_folder_name),
        #        os.path.join(self.preprocessed_folder_path, image_folder_name[2:]),
        #    )

        # Get the new folder name
        # image_folder_names = os.listdir(self.preprocessed_folder_path)

        # # Copy the segmentations to the correct folder
        # for operator in segmentation_operator:
        #     for image_folder_name in image_folder_names:

        #         # Resolve the path for the preprocessed folder
        #         resolved_preprocessed_folder_path = os.path.join(
        #             self.preprocessed_folder_path, image_folder_name, operator
        #         )

        #         # Delete the folder in the preprocessing folder if existing
        #         if os.path.exists(resolved_preprocessed_folder_path):
        #             shutil.rmtree(resolved_preprocessed_folder_path)

        #         # Copy the data
        #         copy_tree(
        #             self.input_folder_path,
        #             os.path.join(self.input_folder_path, operator, image_folder_name),
        #             resolved_preprocessed_folder_path,
        #         )

        # # Generate folder structure at the output
        # for output_folder in image_folder_names:

        #     # Resolve the path for the output folder
        #     resolved_output_folder_path = os.path.join(
        #         self.output_folder_path, output_folder
        #     )

        #     # Delete the folder if it is existing already
        #     if os.path.exists(resolved_output_folder_path):
        #         shutil.rmtree(resolved_output_folder_path)

        #     # Generate the new directory
        #     os.mkdir(os.path.join(self.output_folder_path, output_folder))

    def rotate_image(
        self,
        rotation_correction_angles_path,
        output_folder_path,
        image_name_ids=("T1c", "T1w", "T2w", "FLAIR"),
        random_rotation: tuple = None,
    ):

        # Adjust the image ids to allow the exception handling
        short_image_name_ids = [name for name in image_name_ids if name != "T1c"]

        # Get the subjects from the folder structure
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Read the angle for each subject
        angle_dict = None
        if os.path.exists(rotation_correction_angles_path):
            with open(rotation_correction_angles_path, "r") as fr:
                angle_dict = json.load(fr)

        # Loop through the subjects and rotate the images and the masks
        for subject in subjects:

            # Generate the output folder path
            output_folder_path_internal = os.path.join(output_folder_path, str(subject))

            # Check if a rotation of the images is necessary
            rotation_angle = angle_dict["angles"][str(subject)]
            if rotation_angle == 0 and subject not in angle_dict["exceptions"].keys():
                if self.output_folder_path != output_folder_path_internal:
                    if os.path.exists(output_folder_path_internal):
                        shutil.rmtree(output_folder_path_internal)
                    shutil.copytree(
                        os.path.join(self.output_folder_path, str(subject)),
                        output_folder_path_internal,
                    )
                continue

            # Generate the output folders if necessary or remove the old ones
            if not os.path.exists(output_folder_path_internal):
                os.mkdir(output_folder_path_internal)
            else:
                shutil.rmtree(output_folder_path_internal)
                os.mkdir(output_folder_path_internal)

            # Get the files of the subject
            image_folder_path = os.path.join(self.output_folder_path, subject)
            image_filenames = list()
            try:
                image_filenames = os.listdir(image_folder_path)
                image_filenames = [
                    file
                    for file in image_filenames
                    if os.path.splitext(file)[1] == ".gz"
                ]
            except OSError:
                print(
                    "Error in "
                    + str(DicomNiftiConverter.rotate_image.__name__)
                    + ": No folder available or folder empty"
                )
                continue

            # If random rotation is questioned, generate a random angle
            random_angle = 0
            if random_rotation is not None:
                random_angle = float(
                    random.randint(random_rotation[0], random_rotation[1])
                )

            # Loop through the files for the application of the rotations
            for filename in image_filenames:

                # Separate the filename
                extension = "." + filename.split(".")[1] + "." + filename.split(".")[2]
                filename_parts = filename[: -len(extension)].split("_")

                # Read the image
                reader = sitk.ImageFileReader()
                reader.SetFileName(
                    os.path.join(self.output_folder_path, subject, filename)
                )
                image = reader.Execute()

                # Get the rotation angle and check for exceptions
                rotation_angle = angle_dict["angles"][str(subject)]
                if subject in angle_dict["exceptions"].keys():
                    if (
                        filename_parts[1]
                        == angle_dict["exceptions"][subject]["sequence"]
                        or "T1c" == angle_dict["exceptions"][subject]["sequence"]
                    ):
                        if filename_parts[1] not in short_image_name_ids:
                            rotation_angle = angle_dict["exceptions"][subject]["angle"]

                # Check if a random rotation is questioned
                if random_rotation is not None:
                    rotation_angle += random_angle

                # Compute the angle in radians
                radians = -np.pi * rotation_angle / 180.0

                # Expand the image to preserve from clipping
                print(
                    "------ Subject {}: {} contoured by {} ------".format(
                        subject, filename_parts[1], filename_parts[-1]
                    )
                )
                print(image.GetSize())
                if filename_parts[1] not in image_name_ids:
                    displacement_dim0 = 200
                    displacement_dim1 = 200
                    displacement_dim2 = 200

                    padding_dist_upper = sitk.VectorUInt32(
                        [displacement_dim0, displacement_dim1, displacement_dim2]
                    )
                    padding_dist_lower = sitk.VectorUInt32(
                        [displacement_dim0, displacement_dim1, displacement_dim2]
                    )
                    padding_filter = sitk.ConstantPadImageFilter()
                    padding_filter.SetConstant(0)
                    padding_filter.SetPadLowerBound(padding_dist_lower)
                    padding_filter.SetPadUpperBound(padding_dist_upper)
                    image = padding_filter.Execute(image)

                # Build the transformation (rotation around the z-axis)
                transform = sitk.AffineTransform(3)
                homogeneous_rot_matrix = np.array(
                    [
                        [np.cos(radians), -np.sin(radians), 0],
                        [np.sin(radians), np.cos(radians), 0],
                        [0, 0, 1],
                    ]
                )
                transform.SetMatrix(homogeneous_rot_matrix.ravel())
                reference_image = image
                interpolator = sitk.sitkNearestNeighbor

                # Check if the file is an image file or a mask
                if filename_parts[1] in image_name_ids:
                    interpolator = sitk.sitkBSpline

                # Resample the image
                resampled_img = sitk.Resample(
                    image, reference_image, transform, interpolator, 0, sitk.sitkFloat32
                )
                print(resampled_img.GetSize())

                if filename_parts[1] not in image_name_ids:

                    # Cast the image to an integer pixel type
                    cast_filter = sitk.CastImageFilter()
                    cast_filter.SetOutputPixelType(sitk.sitkInt8)
                    casted_img = cast_filter.Execute(resampled_img)

                    # Get the bounding box
                    stats_filter = sitk.LabelShapeStatisticsImageFilter()
                    stats_filter.ComputeOrientedBoundingBoxOn()
                    stats_filter.Execute(casted_img)
                    bounding_box = stats_filter.GetBoundingBox(1)

                    # Crop the image to the content
                    crop_filter = sitk.CropImageFilter()
                    resampled_img_size = resampled_img.GetSize()
                    bounding_box_lower_vec = sitk.VectorUInt32(
                        [bounding_box[0], bounding_box[1], bounding_box[2]]
                    )
                    bounding_box_upper_vec = sitk.VectorUInt32(
                        [
                            resampled_img_size[0] - bounding_box[0] - bounding_box[3],
                            resampled_img_size[1] - bounding_box[1] - bounding_box[4],
                            resampled_img_size[2] - bounding_box[2] - bounding_box[5],
                        ]
                    )
                    crop_filter.SetLowerBoundaryCropSize(bounding_box_lower_vec)
                    crop_filter.SetUpperBoundaryCropSize(bounding_box_upper_vec)
                    resampled_img = crop_filter.Execute(resampled_img)
                    print(resampled_img.GetSize())

                # Write the image file to the appropriate location
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(output_folder_path_internal, filename))
                writer.Execute(resampled_img)

    def convert_to_nifti(self) -> None:
        """
        This function controls the conversation of the DICOM images to nifti images by using the python interface
        of the 3DSlicer

        Note:
            - The function displays the GUI of the 3DSlicer when no new file is generated. This is a workaround, since
              the 3DSlicer isn't loading DICOM series with warnings
            - The conversation procedure takes a long time for a large number of files, since the 3DSlicer needs to
              be closed and restarted for each dataset / operator. Be patient!
        """

        # Get the subjects from the folder structure
        subjects = os.listdir(self.preprocessed_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Convert the subjects dataset to nifti
        for subject in subjects:

            print("=================================================================")
            print("Converting Subject {}".format(str(subject)))
            print("=================================================================")

            data_items = os.listdir(
                os.path.join(self.preprocessed_folder_path, subject)
            )

            for data_item in data_items:

                # Get the image modality
                if data_item.startswith("A_"):
                    name = data_item.split(sep="-")[1]

                    # Renaming the specific images
                    if "T1w_KM" in name:
                        name = "T1c"

                    if "T1w_NAT" in name:
                        name = "T1w"

                    # Define the input path
                    input_path = os.path.join(
                        self.preprocessed_folder_path, subject, data_item
                    )

                    # Define the output path
                    output_path = os.path.join(self.output_folder_path, subject)

                    # Build the command for slicer and start the script
                    if os.name == "nt":
                        command_string = (
                            r'"{}"'.format(self.slicer_path)
                            + " --no-main-window --python-script "
                            + self.python_script_path
                            + " --input-folder "
                            + input_path
                            + " --output-folder "
                            + output_path
                            + " -e -p "
                            + str(subject)
                            + " -m "
                            + name
                        )
                        with subprocess.Popen(command_string) as proc:
                            proc.wait(120)

                    else:
                        command_string = (
                            self.slicer_path
                            + " --no-main-window --python-script "
                            + self.python_script_path
                            + " --input-folder "
                            + input_path
                            + " --output-folder "
                            + output_path
                            + " -e -p "
                            + str(subject)
                            + " -m "
                            + name
                        )
                        os.system(command_string)

                else:

                    # Define the name
                    name = data_item

                    # Define the input path
                    input_path = os.path.join(
                        self.preprocessed_folder_path, subject, data_item
                    )

                    # Define the output path
                    output_path = os.path.join(self.output_folder_path, subject)

                    # Get the number of files in a directory
                    num_of_files_before = len(
                        [
                            file
                            for file in os.listdir(output_path)
                            if os.path.isfile(os.path.join(output_path, file))
                        ]
                    )

                    # Build the command for slicer and start the script
                    if os.name == "nt":

                        command_string = (
                            r'"{}"'.format(self.slicer_path)
                            + " --no-main-window --python-script "
                            + self.python_script_path
                            + " --input-folder "
                            + input_path
                            + " --output-folder "
                            + output_path
                            + " -p "
                            + str(subject)
                            + " -m "
                            + name
                        )
                        with subprocess.Popen(command_string) as proc:
                            proc.wait(120)

                    else:
                        command_string = (
                            self.slicer_path
                            + " --no-main-window --python-script "
                            + self.python_script_path
                            + " --input-folder "
                            + input_path
                            + " --output-folder "
                            + output_path
                            + " -p "
                            + str(subject)
                            + " -m "
                            + name
                        )
                        os.system(command_string)

                    # Get the number of files after the conversation
                    num_of_files_after = len(
                        [
                            file
                            for file in os.listdir(output_path)
                            if os.path.isfile(os.path.join(output_path, file))
                        ]
                    )

                    # Check if the number of files has raised (otherwise the conversation does not took place)
                    if num_of_files_after == num_of_files_before:
                        # Build the command string (with the application displayed to react on dialog boxes)
                        if os.name == "nt":
                            command_string = (
                                r'"{}"'.format(self.slicer_path)
                                + " --python-script "
                                + self.python_script_path
                                + " --input-folder "
                                + input_path
                                + " --output-folder "
                                + output_path
                                + " -p "
                                + str(subject)
                                + " -m "
                                + name
                            )

                            # Start the script
                            with subprocess.Popen(command_string) as proc:
                                proc.wait(999999)

                        else:
                            command_string = (
                                self.slicer_path
                                + " --python-script "
                                + self.python_script_path
                                + " --input-folder "
                                + input_path
                                + " --output-folder "
                                + output_path
                                + " -p "
                                + str(subject)
                                + " -m "
                                + name
                            )

                            # Start the script
                            os.system(command_string)

    def search_for_person(self, path: str, filename: str) -> str:
        """
        This function computes the name of an operator, which is missing in the filename

        Args:
            path        The path to the correct subject directory
            filename    The filename with the missing operator name

        Returns:
            str         The new filename containing the correct name of the missing operator
        """
        # Define a variable for the new filename
        filename_new = ""

        # Separate the filename
        underline_splitted_filename = filename.split("_")

        # Get files in the directory
        files = os.listdir(path)

        # Define a list to store the congruent names
        congruent_files = list()

        # Search for the other names to get the missing name
        for file in files:
            if str(underline_splitted_filename[1]) in str(file):

                file_splitted = file.split("_")
                file_splitted[-1] = file_splitted[-1].split(".")[0]

                if len(underline_splitted_filename) == 4:
                    if len(file_splitted[-1]) > 0:
                        if str(underline_splitted_filename[2]) in str(file_splitted[2]):
                            if str(file_splitted[-1]) not in congruent_files:
                                congruent_files.append(str(file_splitted[-1]))

                else:
                    if len(file_splitted[-1]) > 0:
                        if str(file_splitted[-1]) not in congruent_files:
                            congruent_files.append(str(file_splitted[-1]))

        # Check if name is present in the directory
        persons2 = self.persons.copy()
        for person in self.persons:

            if person in congruent_files:
                persons2.remove(person)

        # Change only the name, if there is just one name left
        if len(persons2) == 1:

            # Build the new filename
            for i in range(len(underline_splitted_filename) - 1):
                filename_new += underline_splitted_filename[i] + "_"
            filename_new += persons2[0] + ".nii.gz"

        # Return the old filename when there is no name added
        else:
            filename_new = filename

        # Return the new filename
        return filename_new

    def get_correct_filename(self, path: str) -> [str, bool]:
        """
        Computes the standardized / correct filename from the path of a file

        Args:
            path        The path of the existing file

        Returns:
            str         The standardized / correct filename
            bool        Reports if the filename has changed
        """

        # Define an indicator when the file has changed
        has_changed = False

        # Get the filename
        filename = os.path.basename(path)

        # Split the filename in its elements
        filename_parts = filename.split(sep="_")

        # Remove the extension from the appropriate element
        if (
            len(filename_parts[-1]) > len(self.ext)
            or str(filename_parts[-1]) == self.ext
        ):
            filename_parts[-1] = filename_parts[-1][: -len(self.ext)]

        # Variable to store the new filename
        filename_new = ""

        # ----------------------------------------------------------
        # Check the path elements and correct them if they are wrong
        # ----------------------------------------------------------

        label = str(filename_parts[1])
        subject = str(filename_parts[0])

        if label.startswith("T1c"):
            filename_new = subject + "_T1c" + self.ext

        elif label.startswith("T1w"):
            filename_new = subject + "_T1w" + self.ext

        elif label.startswith("T2"):
            filename_new = subject + "_T2w" + self.ext

        elif label.startswith("FLAIR"):
            filename_new = subject + "_FLAIR" + self.ext

        elif label.startswith("GTV"):
            filename_new = subject + "_GTV_" + str(filename_parts[2]) + self.ext

        elif label.startswith("CTV") or label.startswith("CVT"):
            if len(filename_parts) == 3:
                filename_new = subject + "_CTV_" + str(filename_parts[2]) + self.ext
            elif len(filename_parts) == 4:
                filename_new = subject + "_CTV_" + str(filename_parts[-2]) + self.ext
            else:
                filename_new = (
                    subject + "_CTV_" + str(filename_parts[1])[-2:] + self.ext
                )

        elif label.startswith("BrainStem") or label.startswith("Brainstem"):
            filename_new = subject + "_BrainStem_" + str(filename_parts[2]) + self.ext

        elif label.startswith("OpticChiasm"):
            filename_new = subject + "_OpticChiasm_" + str(filename_parts[2]) + self.ext

        elif label.startswith("Pituitary"):
            filename_new = subject + "_Pituitary_" + str(filename_parts[2]) + self.ext

        elif label.startswith("Cochlea"):
            filename_new = (
                subject
                + "_Cochlea_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("Eye"):
            filename_new = (
                subject
                + "_Eye_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("Hippocampus"):
            filename_new = (
                subject
                + "_Hippocampus_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("Lacrimal"):
            filename_new = (
                subject
                + "_Lacrimal_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("Lens"):
            filename_new = (
                subject
                + "_Lens_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("OpticNerve"):
            filename_new = (
                subject
                + "_OpticNerve_"
                + str(filename_parts[2])
                + "_"
                + str(filename_parts[3])
                + self.ext
            )

        elif label.startswith("Retina"):
            if len(filename_parts) == 3:
                filename_new = (
                    subject
                    + "_Retina_"
                    + str(filename_parts[1][-1])
                    + "_"
                    + str(filename_parts[2])
                    + self.ext
                )
            else:
                filename_new = (
                    subject
                    + "_Retina_"
                    + str(filename_parts[2])
                    + "_"
                    + str(filename_parts[3])
                    + self.ext
                )

        elif label.startswith("labelmask"):
            filename_new = filename

        else:
            if len(filename_parts) == 3:
                filename_new = (
                    "x_"
                    + subject
                    + "_"
                    + str(filename_parts[1])
                    + "_"
                    + str(filename_parts[2])
                    + self.ext
                )
            else:
                filename_new = "x_" + subject + "_" + str(filename_parts[1]) + self.ext

        # -------------------------------------------------------------------------
        # Check if the operator person is missing or the extension is present twice
        # ------------------------------------------------------------------------

        excluded_files = ["T1c", "T1w", "T2w", "FLAIR", "labelmask"]
        splitted_filename_new = filename_new.split("_")

        # Exclude the images since there is no operator
        if not any(x in splitted_filename_new[1] for x in excluded_files):

            # Check if the extension is present twice
            if str(splitted_filename_new[-1]) == str(self.ext + self.ext) or not any(
                x in str(filename_new) for x in self.persons
            ):
                # Find the missing persons name and introduce it into the new filename
                # filename_new = self.search_for_person(os.path.join(self.output_folder_path, subject, filename_new))
                filename_new = self.search_for_person(
                    os.path.join(self.output_folder_path, subject), filename_new
                )

        # Check if the extension is present twice in the new filename
        splitted_filename_new2 = filename_new.split(".")
        if len(splitted_filename_new2) > 3:

            # Build the new filename
            filename_new = ""
            for i in range(len(splitted_filename_new2) - 2):
                filename_new += splitted_filename_new2[i]
            filename_new += (
                "." + splitted_filename_new2[-2] + "." + splitted_filename_new2[-1]
            )

        # -----------------------------------------
        # Return the new filename and the indicator
        # -----------------------------------------
        if filename != filename_new:
            has_changed = True

        return filename_new, has_changed

    def postprocess_structure(self) -> None:
        """
        Performs the postprocessing of the filenames to have an appropriate structure without any deviations
        """
        # Get the subjects
        subjects = os.listdir(self.output_folder_path)

        # Get the filenames for each subject
        for subject in subjects:

            filenames = os.listdir(self.output_folder_path + "/" + subject + "/")

            # Loop through the filenames in the subjects directory
            for filename in filenames:

                # Check and correct the filename if it is wrong
                src_path = os.path.join(self.output_folder_path, subject, filename)
                filename_new, changed = self.get_correct_filename(src_path)

                # If the filename was wrong, rename the file or delete it if it is not used for further processing
                if changed:
                    trg_path = os.path.join(
                        self.output_folder_path, subject, filename_new
                    )

                    # Remove the marked files
                    if filename_new.startswith("x"):
                        os.remove(src_path)

                    # Rename the files with a new filename and print the changes
                    else:
                        os.rename(src_path, trg_path)
                        print(
                            "Old filename: %30s \tNew filename: %30s"
                            % (filename, filename_new)
                        )

    @staticmethod
    def get_std_label(
        label: str, label_type: LabelType = LabelType.Long
    ) -> Union[None, str]:
        """
        Function to get the standard form of the label according to the label type specified

        Args:
            label           The label to bring in the standard form
            label_type      The type / length of the label

        Returns:
            The label in its standard form w.r.t. the selected label type
        """

        # Generate the default return value
        label_new = None

        # Split the label
        splitted_label = label[:-7].split(sep="_")

        # Check if the label is an image or not
        image_elements = ["T1", "T2", "FLAIR"]
        if not any(x in splitted_label[1] for x in image_elements):

            # Returns only the name of the structure without a side
            if label_type == LabelType.ExtraShort:
                label_new = str(splitted_label[1])

            # Returns the name of the structure and the side
            else:
                # Check if the label is from an element which not present on both sides of the brain
                undirected_elements = ["TV", "Pituitary", "BrainStem", "OpticChiasm"]
                if (
                    any(x in splitted_label[1] for x in undirected_elements)
                    or len(splitted_label) < 4
                ):

                    if label_type == LabelType.Short:

                        # Construct the short label
                        label_new = str(splitted_label[1])

                    else:
                        # Construct the long label
                        label_new = (
                            str(splitted_label[1]) + "_" + str(splitted_label[2])
                        )

                # For elements which are present on both sides of the head
                else:

                    if label_type == LabelType.Short:

                        # Construct the short label
                        label_new = (
                            str(splitted_label[1]) + "_" + str(splitted_label[2])
                        )

                    else:
                        # Construct the label
                        label_new = (
                            str(splitted_label[1])
                            + "_"
                            + str(splitted_label[2])
                            + "_"
                            + str(splitted_label[3])
                        )

        # Return the new label
        return label_new

    def get_unique_labels(
        self, label_type: LabelType = LabelType.Long, excluded_labels: list = ()
    ) -> list:
        """
        Function to compute the unique labels

        Args:
            label_type          The type / length of the label
            excluded_labels     A list of labels to exclude from the search

        Returns:
            The list of unique labels found in the data structure
        """

        # An empty list to get the unique labels
        unique_labels = list()

        # Get all the subjects in the output
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Loop through the subjects
        for subject in subjects:

            # Get the labels in the subject directories
            labels = os.listdir(os.path.join(self.output_folder_path, subject))
            labels.sort()

            for label in labels:

                label_new = self.get_std_label(label, label_type=label_type)

                if not any(x in label for x in excluded_labels):
                    if (
                        label_new not in unique_labels
                        and label_new not in excluded_labels
                    ):
                        if label_new is not None:
                            unique_labels.append(label_new)

        # Sort the unique labels
        unique_labels.sort()

        # Return the unique labels
        return unique_labels

    @staticmethod
    def natural_sorting(text):
        """
        Helper function to enable a natural sorting
        """
        text2int = lambda txt: int(txt) if txt.isdigit() else txt
        return [text2int(x) for x in re.split("(\d+)", text)]

    def get_missing_labels(
        self, unique_labels: Union[list, None], label_type: LabelType = LabelType.Long
    ):
        """
        Function to compute the label, which are missing for each subject

        Args:
            unique_labels       List of the unique labels
            label_type          The type / length of the label
        """

        # Get the unique labels if they are not provided
        if unique_labels is None:
            self.get_unique_labels(label_type=label_type)

        # Generate a new dict to store the list of missing labels per subject
        missing_labels = dict()

        # Get the subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Loop through the directories to get the missing labels
        for subject in subjects:

            # Generate a new list
            std_label_list = list()

            # Get all the labels
            labels = os.listdir(os.path.join(self.output_folder_path, subject))

            # Get all the standard labels
            for label in labels:
                std_label = self.get_std_label(label, label_type=label_type)
                if std_label not in std_label_list:
                    std_label_list.append(std_label)

            # Get the missing labels
            missing_labels_list = list(set(unique_labels).difference(std_label_list))

            # Add the missing labels to the appropriate dict
            missing_labels[str(subject)] = missing_labels_list

        return missing_labels

    def generate_label_config(
        self, label_type: LabelType = LabelType.Short, excluded_labels: list = ()
    ) -> None:
        """
        This function generates the label configuration and stores it

        Args:
            label_type          The type / length of the labels
            excluded_labels     The labels which should be excluded
        """

        # Add the label masks (if existing) to the excluded label masks if already generated
        excluded_labels.append("label")

        # Get the unique labels
        unique_labels = self.get_unique_labels(
            label_type=label_type, excluded_labels=excluded_labels
        )

        # Generate the label dict
        self.label_dict = {label: (idx + 1) for idx, label in enumerate(unique_labels)}

        # Load the label dict
        stored_label_dict = None
        if os.path.exists(self.label_config_path):
            with open(self.label_config_path, "r") as fr:
                stored_label_dict = json.load(fr)

        # Save the label dict if it is not existing
        else:
            with open(self.label_config_path, "w+") as fw:
                json.dump(self.label_dict, fw)

        # Check if there are differences between the label dicts
        if stored_label_dict is not None:
            if self.label_dict != stored_label_dict:

                # Save the label dict to a config file
                with open("label_config.json", "w+") as fw:
                    json.dump(self.label_dict, fw)

    def resample_images(
        self,
        ref_img_id: str = "T1c",
        resample_image_ids: list = ("T1c", "T1w", "T2w", "FLAIR"),
    ) -> None:
        """
        This function resamples and saves the images (T1c, T1w, T2w, FLAIR) to the specified output size

        Args:
            ref_img_id              The name of the reference image
            resample_image_ids      The names of the other images (used to build the paths)
        """

        # Get the subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Check and correct the images to resample, if the reference image is part of the list
        resampled_img_ids_corr = list(resample_image_ids)
        if ref_img_id in resampled_img_ids_corr:
            resampled_img_ids_corr.remove(ref_img_id)

        # Loop through the subjects
        for subject in subjects:

            # Generate the filenames of the images
            filenames = {
                image: (subject + "_" + image + self.ext)
                for image in resampled_img_ids_corr
            }

            # Get the path of the reference image
            ref_img_path = os.path.join(
                self.output_folder_path, subject, subject + "_" + ref_img_id + self.ext
            )

            # Import the reference image
            ref_img_reader = sitk.ImageFileReader()
            ref_img_reader.SetFileName(ref_img_path)
            ref_img = ref_img_reader.Execute()

            # Generate a new image with the desired output size
            ref2_img = sitk.Image(
                self.output_size[0],
                self.output_size[1],
                self.output_size[2],
                sitk.sitkFloat32,
            )
            ref2_img.SetSpacing(spacing=ref_img.GetSpacing())
            ref2_img.SetDirection(direction=ref_img.GetDirection())
            ref2_img.SetOrigin(origin=ref_img.GetOrigin())

            # Generate a new reference image of the appropriate output size
            ref_img = sitk.Resample(
                ref_img,
                ref2_img,
                sitk.Transform(),
                sitk.sitkBSpline,
                0,
                sitk.sitkFloat32,
            )

            # Save the reference image with additional string "resampled"
            print(
                "---------------------------------------------------------------------------"
            )
            print(
                "Saving the reference image "
                + ref_img_id
                + " of subject "
                + subject
                + "..."
            )
            sitk.WriteImage(
                ref_img,
                os.path.join(
                    self.output_folder_path,
                    subject,
                    subject + "_" + ref_img_id + "_resampled" + self.ext,
                ),
                True,
            )

            # Loop through the images to resample
            for img_id, filename in filenames.items():

                # Print the state
                print(
                    "Resampling the image " + img_id + " of subject " + subject + "..."
                )

                # Get the path to the image to resample
                temp_img_path = os.path.join(self.output_folder_path, subject, filename)

                # Load the image to resample
                temp_img_reader = sitk.ImageFileReader()
                temp_img_reader.SetFileName(temp_img_path)
                temp_img = temp_img_reader.Execute()

                # Resample the appropriate image with the properties of the reference image
                resampled_img = sitk.Resample(
                    temp_img,
                    ref_img,
                    sitk.Transform(),
                    sitk.sitkBSpline,
                    0,
                    sitk.sitkFloat32,
                )

                # Save the resampled image
                sitk.WriteImage(
                    resampled_img,
                    os.path.join(
                        self.output_folder_path,
                        subject,
                        subject + "_" + img_id + "_resampled" + self.ext,
                    ),
                    True,
                )
            print(" ")

    def padding_to_cube(
        self,
        padding_image_ids: tuple = (
            "labelmask_all",
            "labelmask_EE",
            "labelmask_EH",
            "labelmask_MB",
        ),
    ) -> None:
        """
        This function pads the images and labelmasks to have equal space in all dimensions

        Args:
            padding_image_ids           The name of the data to pad

        """

        # Get the subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Loop through the subjects
        for subject in subjects:

            # Print the state
            print("Padding the images of subject " + subject + " to a cube...")

            # Generate the filenames of the images
            filenames = {
                file: (subject + "_" + file + self.ext) for file in padding_image_ids
            }

            # Define a variable to store the maximal axis size
            max_size = None

            # Loop through the data files
            for filename in filenames:

                # Load the data
                file_path = os.path.join(
                    self.output_folder_path,
                    subject,
                    subject + "_" + filename + self.ext,
                )
                data = sitk.ReadImage(file_path)

                # Get the maximal size of the largest axis for the first data
                if max_size is None:
                    max_size = max(sitk.Image.GetSize(data))

                # Get the size of the data
                data_size = sitk.Image.GetSize(data)

                # Compute the padding in each dimension
                padding_vec = sitk.VectorUInt32(
                    (
                        abs(data_size[0] - max_size),
                        abs(data_size[1] - max_size),
                        abs(data_size[2] - max_size),
                    )
                )

                # Instantiate a new filter
                padding_filter = sitk.ConstantPadImageFilter()
                padding_filter.SetConstant(0.0)
                padding_filter.SetPadUpperBound(padding_vec)

                # Perform the padding
                padded_data = padding_filter.Execute(data)

                # Save the padded image
                sitk.WriteImage(
                    padded_data,
                    os.path.join(
                        self.output_folder_path,
                        subject,
                        subject + "_" + filename + "_final" + self.ext,
                    ),
                    True,
                )

    def combine_label_masks(
        self,
        ref_img_id: str = "T1c",
        label_hierarchy: list = None,
        label_type: LabelType = LabelType.Short,
        combination_function: Callable = None,
    ) -> None:
        """
        This function combines the labels per subject and operator and resamples them to the specified output size

        Note:
            - The labels are added / overwritten in a sequential hierarchy what allows them to overlap

        Args:
            label_hierarchy         A list with the sequence of labels
            label_type              The type / length of the labels
            combination_function    A function to combine all subject-specific label masks
        """

        # Generate the label dict if it is not already existing
        if self.label_dict is None:
            self.generate_label_config(excluded_labels=[])

        # Get the subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Check and instantiate a sequence for the labels
        if label_hierarchy is None or []:
            sequence = self.label_hierarchy
        else:
            sequence = label_hierarchy

        # Loop through the subjects
        for subject in subjects:

            # Get all filenames
            filenames = os.listdir(os.path.join(self.output_folder_path, subject))
            filenames_operator = {operator: list() for operator in self.persons}

            # Loop through the filenames and sort them according to the operator
            for filename in filenames:
                if any(x in filename for x in self.persons) and "final" not in filename:

                    # Get the persons name
                    person = filename[
                        -(len(self.ext) + len(self.persons[0])) : -len(self.ext)
                    ]

                    # Check if the person is existing in the dictionary
                    if any(x in person for x in filenames_operator.keys()):

                        # Append the filename to the specific operator
                        filenames_operator[person].append(filename)

                    # Notify if an error is detected
                    else:
                        print(
                            "Error during filename sorting! Error encountered during the processing of "
                            + filename
                        )

            # Get the correct image as a base for the parameters
            base_img_path = os.path.join(
                self.output_folder_path, subject, subject + "_" + ref_img_id + self.ext
            )

            # Import the base image
            reader_base_img = sitk.ImageFileReader()
            reader_base_img.SetFileName(base_img_path)
            base_img = reader_base_img.Execute()

            # Get the properties of the base image
            base_origin = base_img.GetOrigin()
            base_spacing = base_img.GetSpacing()
            base_direction = base_img.GetDirection()

            # Generate a new base image of size output_size
            base2_img = sitk.Image(
                self.output_size[0],
                self.output_size[1],
                self.output_size[2],
                sitk.sitkUInt8,
            )

            # Set the properties of the second base image
            base2_img.SetOrigin(base_origin)
            base2_img.SetSpacing(base_spacing)
            base2_img.SetDirection(base_direction)
            base2_size = base2_img.GetSize()

            # Generate an empty dict to store all images per subject
            combined_mask = {operator: None for operator in self.persons}

            # Generate new numpy arrays from an Image to allow the combination of the masks
            combined_mask_numpy = {
                operator: sitk.GetArrayFromImage(sitk.Image(base2_size, sitk.sitkUInt8))
                for operator in self.persons
            }

            # Loop through the operator
            for operator in filenames_operator.keys():

                # Make a print to the prompt to show the progress
                print(
                    "Combining the labels of the subject "
                    + str(subject)
                    + " and operator "
                    + str(operator)
                    + "..."
                )

                # Build the sorted list of filenames (according to the hierarchy defined previously)
                ordered_filenames_operator = list()
                for element in sequence:
                    for filename in filenames_operator[operator]:
                        if element in filename:
                            ordered_filenames_operator.append(filename)

                # Loop through the ordered filenames
                for filename in ordered_filenames_operator:

                    # Generate the label dict and get the standard label and index
                    std_label = self.get_std_label(filename, label_type=label_type)
                    label_index = self.label_dict[std_label]

                    # Get the temporary label mask to add
                    temp_mask_path = os.path.join(
                        self.output_folder_path, subject, filename
                    )
                    reader_temp_mask = sitk.ImageFileReader()
                    reader_temp_mask.SetFileName(temp_mask_path)
                    temp_mask = reader_temp_mask.Execute()

                    # Resample the label mask to have the same properties as the base image
                    resampled_img = sitk.Resample(
                        temp_mask,
                        base2_img,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0,
                        sitk.sitkUInt8,
                    )

                    # Get the numpy array of the resampled image
                    resampled_img_numpy = sitk.GetArrayFromImage(resampled_img)

                    # Mask the numpy array to prevent from deleting content added from other labels
                    resampled_img_mask = ma.masked_not_equal(resampled_img_numpy, 0)

                    # Adjust the label index
                    resampled_img_numpy *= label_index

                    # Fuse the label mask with the appropriate numpy mask
                    combined_mask_numpy[operator][
                        resampled_img_mask.mask
                    ] = resampled_img_numpy[resampled_img_mask.mask]

                    # Print the max value if out of bounds
                    if np.max(combined_mask_numpy[operator]) >= 20:
                        raise ValueError(
                            "Value larger than 20 in "
                            + str(subject)
                            + " and operator "
                            + str(operator)
                        )

            # Save the combined masks as Nifti-file and store it in the combined mask for further processing
            for operator in self.persons:

                # Convert the combined numpy mask to an Image
                img = sitk.GetImageFromArray(combined_mask_numpy[operator])

                # Set the meta parameters for the image
                img.SetOrigin(base_origin)
                img.SetDirection(base_direction)
                img.SetSpacing(base_spacing)

                # Add the image to the combined mask
                # --> only used when further processing is desired
                combined_mask[operator] = img

                # Write the image to the appropriate path
                sitk.WriteImage(
                    img,
                    os.path.join(
                        self.output_folder_path,
                        subject,
                        subject + "_labelmask_" + str(operator) + self.ext,
                    ),
                    True,
                )

            # Perform the combination of the operator-specific label masks, if a combination function is provided
            if combination_function is not None:
                print(
                    "Combining all labels of the subject "
                    + str(subject)
                    + " to one label mask...\n"
                )
                combination_function(combined_mask, subject)

    def labelvoting(self, combined_mask: dict, subject: str) -> None:
        """
        This function performs the labelvoting and combines the label masks from all operators to one label mask

        Note:
            - This function is usually passed as a parameter to the combine_label_mask function
            - The generated label mask is stored into the output directory and named in the following order:
              "[subject_id]_labelmask_all.[extension]"

        Args:
            combined_mask           The combined mask of sitk.Images
            subject                 The subject id
        """

        # Generate and fill a new vector of images
        vector_imgs = sitk.VectorOfImage()
        [vector_imgs.push_back(image) for image in combined_mask.values()]

        # Perform the label voting
        label_voting_filter = sitk.LabelVotingImageFilter()
        final_label_map = label_voting_filter.Execute(vector_imgs)

        # Adjust the maximal label map value for undefined classes
        final_label_map_direction = final_label_map.GetDirection()
        final_label_map_origin = final_label_map.GetOrigin()
        final_label_map_spacing = final_label_map.GetSpacing()
        final_label_map_numpy = sitk.GetArrayFromImage(final_label_map)
        np.putmask(final_label_map_numpy, final_label_map_numpy == 18, 0)
        final_label_map = sitk.GetImageFromArray(final_label_map_numpy)
        final_label_map.SetOrigin(final_label_map_origin)
        final_label_map.SetDirection(final_label_map_direction)
        final_label_map.SetSpacing(final_label_map_spacing)

        print(
            "The combined mask of "
            + str(subject)
            + " has a max value of "
            + str(np.max(final_label_map_numpy))
        )

        sitk.WriteImage(
            final_label_map,
            os.path.join(
                self.output_folder_path, subject, subject + "_labelmask_all" + self.ext
            ),
            True,
        )

    def register_images(
        self,
        ref_img_id: str = "T1c",
        register_image_ids: list = ("T1w", "T2w", "FLAIR"),
    ):

        # Get the subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Loop through the subjects
        for subject in subjects:

            # Print out the progress
            print("Registrating subject {}...".format(subject))

            # Get all filenames
            filenames = os.listdir(os.path.join(self.output_folder_path, subject))

            # Build a new dict with the image ids to register
            filenames_to_register = {image: str for image in register_image_ids}

            # Define a variable for the filename of the reference image
            filename_ref_img = ""

            # Get all image files to register and the reference image
            for filename in filenames:

                # Check if the current filename is the reference image
                if filename.find(ref_img_id) != -1:
                    filename_ref_img = filename

                for image_id in filenames_to_register.keys():

                    # Check if the filename contains a name in filenames_to_register
                    if filename.find(image_id) != -1:

                        # Add the filename to the list
                        # Only one file can exist per image id
                        filenames_to_register[image_id] = filename

            # Load the reference image, if existing
            assert (
                filename_ref_img != ""
            ), "Registration Error: The reference file is not existing!"
            ref_img = sitk.ReadImage(
                os.path.join(self.output_folder_path, subject, filename_ref_img)
            )

            # Save the reference image with a new filename to be consistent with the naming
            filename_ref_pure = os.path.splitext(os.path.splitext(filename_ref_img)[0])[
                0
            ]
            sitk.WriteImage(
                ref_img,
                os.path.join(self.output_folder_path, subject, filename_ref_pure)
                + "_reg"
                + self.ext,
            )

            # Loop through the images to register
            for img_id, filename in filenames_to_register.items():

                # Load the floating image
                floating_img = sitk.ReadImage(
                    os.path.join(self.output_folder_path, subject, filename)
                )

                # Register the image
                registration = reg.MultiModalRegistration(number_of_iterations=350)
                parameters = reg.MultiModalRegistrationParams(ref_img)
                registered_img = registration.execute(floating_img, parameters)

                # Get the filename without extension
                filename_pure = os.path.splitext(os.path.splitext(filename)[0])[0]

                # Save the image as a new image
                sitk.WriteImage(
                    registered_img,
                    os.path.join(self.output_folder_path, subject, filename_pure)
                    + "_reg"
                    + self.ext,
                )

    def delete_files(self, base_dir_path: str, substr: str) -> None:

        # Check if the substring is not empty
        assert (
            substr != "" or substr is not None
        ), "Deletion Error: There must be a substring"

        # Check if the base directory path is given
        assert (
            base_dir_path != "" or base_dir_path is not None
        ), "Deletion Error: There must be a base directory"

        # Get the subjects
        subjects = os.listdir(base_dir_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Loop through the subjects
        for subject in subjects:

            # Get all filenames
            filenames = os.listdir(os.path.join(self.output_folder_path, subject))

            # Loop through the filenames
            for filename in filenames:

                # If the filename is contained delete the file
                if filename.find(substr) != -1:
                    os.remove(os.path.join(self.output_folder_path, subject, filename))

    def evaluation_orientation(self) -> None:
        """This method generates a pdf file with images of each subject"""

        # Import the appropriate packages
        import matplotlib.backends.backend_pdf as pdf
        import matplotlib.pyplot as plt

        # Get all subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Instantiate a new dict of dicts to store the filenames of the files to check
        subject_files = dict(dict())
        [subject_files.update({str(subject): dict()}) for subject in subjects]

        # Loop through the subjects and look for the appropriate files to include in the pdf
        for subject in subjects:

            # Get all filenames
            filenames = os.listdir(os.path.join(self.output_folder_path, subject))

            # Select only the filenames with final in it
            for filename in filenames:

                # Check if this is a final file
                if "final" in filename:

                    if "T1c" in filename:
                        subject_files[str(subject)]["T1c"] = filename
                    elif "T1w" in filename:
                        subject_files[str(subject)]["T1w"] = filename
                    elif "T2w" in filename:
                        subject_files[str(subject)]["T2w"] = filename
                    elif "FLAIR" in filename:
                        subject_files[str(subject)]["FLAIR"] = filename
                    elif "labelmask_all" in filename:
                        subject_files[str(subject)]["labelmask"] = filename
                    else:
                        pass

        # Loop through the selected files and generate a pdf
        pp = pdf.PdfPages(
            os.path.join(os.path.expanduser("~/Desktop"), "evaluation_orientation.pdf")
        )

        for subject, data in subject_files.items():

            for seq, filename in data.items():

                # Report the current state
                print("Analyzing subject " + str(subject) + ": Sequence " + str(seq))

                # Import the appropriate image
                img = sitk.ReadImage(
                    os.path.join(self.output_folder_path, str(subject), filename)
                )
                np_img = sitk.GetArrayFromImage(img)

                # Get the samples from the image
                selected_idx = [
                    int(np_img.shape[0] * 0.4),
                    int(np_img.shape[0] * 0.45),
                    int(np_img.shape[0] * 0.5),
                    int(np_img.shape[0] * 0.55),
                    int(np_img.shape[0] * 0.6),
                    int(np_img.shape[0] * 0.65),
                ]
                samples_selected = list()
                [samples_selected.append(np_img[i, :, :]) for i in selected_idx]

                # Generate the plot
                # x, y = None, None
                fig, axes = plt.subplots(3, 2, figsize=(4, 6), sharex=True, sharey=True)
                i = 0
                for axis_x in range(axes.shape[1]):
                    for axis_y in range(axes.shape[0]):
                        axes[axis_y, axis_x].imshow(samples_selected[i])
                        axes[axis_y, axis_x].set_title(
                            "Subject " + str(subject) + ": " + str(seq)
                        )
                        i += 1
                pp.savefig(fig)
                plt.close()

        pp.close()

    def check_and_correct_orientation(self, output_orientation: str = "LPS") -> None:
        """This method checks and corrects the orientation of all generated images acc. to the specified orientation"""

        # Get all subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Get all the data for the subjects
        subject_files = {}
        for subject_folder in subjects:
            image_files = os.listdir(
                os.path.join(self.output_folder_path, subject_folder)
            )
            image_files = [
                os.path.join(self.output_folder_path, subject_folder, file)
                for file in image_files
                if file.endswith(".nii.gz")
            ]
            subject_files = {**subject_files, **{subject_folder: image_files}}

        # Definitions for itk since enums are not wrapped currently
        # (see https://itk.org/pipermail/insight-users/2017-May/054606.html)
        ITK_COORDINATE_UNKNOWN = 0
        ITK_COORDINATE_Right = 2
        ITK_COORDINATE_Left = 3
        ITK_COORDINATE_Posterior = 4
        ITK_COORDINATE_Anterior = 5
        ITK_COORDINATE_Inferior = 8
        ITK_COORDINATE_Superior = 9

        ITK_COORDINATE_PrimaryMinor = 0
        ITK_COORDINATE_SecondaryMinor = 8
        ITK_COORDINATE_TertiaryMinor = 16

        ITK_COORDINATE_ORIENTATION_RIP = (
            (ITK_COORDINATE_Right << ITK_COORDINATE_PrimaryMinor)
            + (ITK_COORDINATE_Inferior << ITK_COORDINATE_SecondaryMinor)
            + (ITK_COORDINATE_Posterior << ITK_COORDINATE_TertiaryMinor)
        )
        ITK_COORDINATE_ORIENTATION_LPS = (
            (ITK_COORDINATE_Left << ITK_COORDINATE_PrimaryMinor)
            + (ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor)
            + (ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor)
        )
        ITK_COORDINATE_ORIENTATION_PIR = (
            (ITK_COORDINATE_Posterior << ITK_COORDINATE_PrimaryMinor)
            + (ITK_COORDINATE_Inferior << ITK_COORDINATE_SecondaryMinor)
            + (ITK_COORDINATE_Right << ITK_COORDINATE_TertiaryMinor)
        )

        # Check and correct the orientation
        if output_orientation == "LPS":
            specified_orientation = ITK_COORDINATE_ORIENTATION_LPS
        elif output_orientation == "RIP":
            specified_orientation = ITK_COORDINATE_ORIENTATION_RIP
        elif output_orientation == "PIR":
            specified_orientation = ITK_COORDINATE_ORIENTATION_PIR
        else:
            return

        float_image_names = ["T1c", "T1w", "T2w", "FLAIR"]
        PixelType = itk.F
        ImageType = itk.Image[PixelType, 3]
        for subject, image_files in subject_files.items():
            for image_file in image_files:
                print(
                    f"Check and correct orientation of subject {subject}: {image_file}"
                )
                # if any(entry in image_file for entry in float_image_names):
                #     PixelType = itk.F
                # else:
                #     PixelType = itk.F  # Check if SI or UI
                # ImageType = itk.Image[PixelType, 3]
                reader = itk.ImageFileReader[ImageType].New(FileName=image_file)
                reader.Update()
                orienter = itk.OrientImageFilter[ImageType, ImageType].New()
                orienter.SetInput(reader.GetOutput())
                orienter.SetUseImageDirection(True)
                orienter.SetDesiredCoordinateOrientation(specified_orientation)
                orienter.Update()
                writer = itk.ImageFileWriter[ImageType].New(FileName=image_file)
                writer.SetInput(orienter.GetOutput())
                writer.Update()

    def correct_orientation(self, subjects_to_corr: list) -> None:
        """This method corrects the orientation of the appropriate subjects"""

        # Get all subjects
        subjects = os.listdir(self.output_folder_path)
        subjects.sort(key=DicomNiftiConverter.natural_sorting)

        # Instantiate a new dict of dicts to store the filenames of the files to correct
        subject_files = dict(dict())
        subjects_to_corr = list(map(str, subjects_to_corr))
        for subject_to_corr in subjects_to_corr:
            if subject_to_corr in subjects:
                subject_files.update({subject_to_corr: dict()})

        # Get the appropriate filenames
        for subject, data in subject_files.items():

            # Get all filenames
            filenames = os.listdir(os.path.join(self.output_folder_path, subject))

            for filename in filenames:

                if "final" in filename:

                    if "T1c" in filename:
                        subject_files[str(subject)]["T1c"] = filename
                    elif "T1w" in filename:
                        subject_files[str(subject)]["T1w"] = filename
                    elif "T2w" in filename:
                        subject_files[str(subject)]["T2w"] = filename
                    elif "FLAIR" in filename:
                        subject_files[str(subject)]["FLAIR"] = filename
                    elif "labelmask_all" in filename:
                        subject_files[str(subject)]["labelmask_all"] = filename
                    elif "labelmask_EE" in filename:
                        subject_files[str(subject)]["labelmask_EE"] = filename
                    elif "labelmask_EH" in filename:
                        subject_files[str(subject)]["labelmask_EH"] = filename
                    elif "labelmask_MB" in filename:
                        subject_files[str(subject)]["labelmask_MB"] = filename
                    else:
                        pass

        # Load and correct the files
        for subject, data in subject_files.items():

            for seq, filename in data.items():

                # Report progress
                print("Reorienting Subject " + subject + ": Sequence " + seq)

                # Load the appropriate image
                img = sitk.ReadImage(
                    os.path.join(self.output_folder_path, subject, filename)
                )

                # Reorient the image
                img_np = sitk.GetArrayFromImage(img)
                img_np = np.rot90(img_np, axes=(2, 0))
                img_np = np.rot90(img_np, axes=(0, 1))
                img_np = np.roll(img_np, shift=-45, axis=2)

                # Get the image from the orientated matrix and transfer the properties
                img_new = sitk.GetImageFromArray(img_np)
                img_new.SetDirection(img.GetDirection())
                img_new.SetOrigin(img.GetOrigin())
                img_new.SetSpacing(img.GetSpacing())

                # Save the reoriented image
                sitk.WriteImage(
                    img_new, os.path.join(self.output_folder_path, subject, filename)
                )


if __name__ == "__main__":
    # ==========================================
    # Description
    # ==========================================
    # This script converts the DICOM files provided by the Insel hospital to the Nifti files and appropriate label
    # masks (one per subject and operator).
    #
    # Please comment and uncomment the lines in the following main structure to control the program flow

    # ==========================================
    # Preparation
    # ==========================================

    # Instantiation of a converter object
    converter = DicomNiftiConverter(
        slicer_path="/Applications/Slicer.app/Contents/MacOS/Slicer",
        python_script_path="/Users/amithkamath/repo/stochastic_segmentation_networks/SlicerRTBatchProcessing/BatchProcessing.py",
        input_folder_path="/Users/amithkamath/data/alternative_segmentations/ISAS_GBM_003",
        preprocessed_folder_path="/Users/amithkamath/data/alternative_segmentations/proc_ISAS_GBM_003",
        output_folder_path="/Users/amithkamath/data/alternative_segmentations/nii_ISAS_GBM_003",
        label_config_path="label_config.json",
        label_hierarchy=[
            "Eye",
            "Retina",
            "Lacrimal",
            "Lens",
            "OpticNerve",
            "OpticChiasm",
            "BrainStem",
            "Hippocampus",
            "Pituitary",
            "Cochlea",
            "CTV",
            "GTV",
        ],
        ext=".nii.gz",
        persons=["EE", "MB", "EH"],
        output_size=[256, 256, 192],
    )

    # converter = DicomNiftiConverter(
    #     slicer_path="~/slicer/Slicer-4.8.1-linux-amd64/Slicer",
    #     python_script_path="~/PycharmProjects/oar-eruefenacht/SlicerRTBatchProcessing/BatchProcessing.py",
    #     input_folder_path="~/Desktop/insel_data2/",
    #     preprocessed_folder_path="~/Desktop/Preprocessed3",
    #     output_folder_path="~/Desktop/Output3",
    #     label_config_path="label_config.json",
    #     label_hierarchy=[
    #         "Eye",
    #         "Retina",
    #         "Lacrimal",
    #         "Lens",
    #         "OpticNerve",
    #         "OpticChiasm",
    #         "BrainStem",
    #         "Hippocampus",
    #         "Pituitary",
    #         "Cochlea",
    #     ],
    #     ext=".nii.gz",
    #     persons=["EE", "MB", "EH"],
    #     output_size=[256, 256, 192],
    # )

    # converter = DicomNiftiConverter(
    #     slicer_path="~/slicer/Slicer-4.8.1-linux-amd64/Slicer",
    #     python_script_path="~/PycharmProjects/oar-eruefenacht/SlicerRTBatchProcessing/BatchProcessing.py",
    #     input_folder_path="~/Desktop/testdata/",
    #     preprocessed_folder_path="~/Desktop/Preprocessed3",
    #     output_folder_path="~/Desktop/Output3",
    #     label_config_path="label_config.json",
    #     label_hierarchy=[
    #         "Eye",
    #         "Retina",
    #         "Lacrimal",
    #         "Lens",
    #         "OpticNerve",
    #         "OpticChiasm",
    #         "BrainStem",
    #         "Hippocampus",
    #         "Pituitary",
    #         "Cochlea",
    #         "GTV",
    #     ],
    #     ext=".nii.gz",
    #     persons=["EE", "MB", "EH"],
    #     output_size=[256, 256, 192],
    # )

    print("=================================================================")
    print("Preprocessing...")
    print("=================================================================")
    # Check and generate the paths, if not provided
    converter.check_paths()
    # Preprocess the folder structure
    converter.preprocess_folder_structure(None)

    # ==========================================
    # Convert the DICOM files to Nifti
    # ==========================================
    print("=================================================================")
    print("Converting...")
    print("=================================================================")
    converter.convert_to_nifti()

    # print("=================================================================")
    # print("Adjust the rotation around the z-axis...")
    # print("=================================================================")
    # converter.rotate_image(
    #    "./rotation_correction_angles_2.json", "D:\\temp5\\", random_rotation=(-20, 20)
    # )
    # converter.rotate_image('./rotation_correction_angles_2.json', 'D:\\temp4\\', random_rotation=None)
    # converter.output_folder_path = "D:\\temp5\\"

    # ==========================================
    # Post-processing the files
    # ==========================================
    print("--------------------------------------")
    print("Postprocessing_PointCloud...")
    print("--------------------------------------")
    converter.postprocess_structure()

    # ==========================================
    # Check the generated data
    # ==========================================
    print("--------------------------------------")
    print("Unique labels")
    print("--------------------------------------")
    unique_labels = converter.get_unique_labels(label_type=LabelType.Short)
    print(unique_labels)

    print("--------------------------------------")
    print("Missing labels")
    print("--------------------------------------")
    missing_labels = converter.get_missing_labels(
        unique_labels, label_type=LabelType.ExtraShort
    )
    print(missing_labels)

    # ==========================================
    # Registration of the data
    # ==========================================
    print("--------------------------------------")
    print("Registration...")
    print("--------------------------------------")
    # Delete the old registration files
    # converter.delete_files(converter.output_folder_path, '_reg')
    # Register the images
    converter.register_images()

    # ==========================================
    # Combine the binary masks per subject
    # ==========================================
    # Combines the label masks to get one mask per subject and operator
    converter.combine_label_masks(
        label_type=LabelType.Short, combination_function=converter.labelvoting
    )

    # ==================================================
    # Resample the images to have equal size per subject
    # ==================================================
    # Resampling the images to have equal size
    # -> When no registration is performed use this:
    # converter.resample_images()

    # -> Otherwise, use this:
    converter.resample_images(
        ref_img_id="T1c_reg",
        resample_image_ids=["T1c_reg", "T1w_reg", "T2w_reg", "FLAIR_reg"],
    )

    # ==================================================================
    # Padding the images and labels to have equal size in all dimensions
    # ==================================================================
    converter.padding_to_cube()

    # ==================================
    # Correction of the data orientation
    # ==================================
    # converter.correct_orientation(subjects_to_corr=[1, 2, 6, 11, 15, 23])
    converter.check_and_correct_orientation()

    # ==================================
    # Evaluation of the data orientation
    # ==================================
    converter.evaluation_orientation()
