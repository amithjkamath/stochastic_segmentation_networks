import os
import glob

if __name__ == '__main__':

    # Specify the path to search in (incl. subdirectories)
    directory_to_rename = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/insel_data2/')

    # Check if the directory is existing
    if os.path.exists(directory_to_rename):
        print('Directory exists!')

    # Instantiate a new list to store the paths to the dcm files
    old_file_paths = list()

    # Instantiate a new list for the new file paths
    new_file_paths = list()

    # Loop through the directory and subdirectories to find the DICOM files
    for (dirpath, dirnames, filenames) in os.walk(directory_to_rename):
        for file in filenames:

            # Check if the file is a DICOM one
            if file[-4:] == '.dcm':

                # Append the old file path to the appropriate list
                old_file_paths.append(os.path.join(dirpath, file))

                # Replace the dots with minus characters and append the new filename to the appropriate list
                new_filename = file[:-4].replace(".", "-")
                new_file_paths.append(os.path.join(dirpath, new_filename + '.dcm'))

    # Rename the files
    for (old_file_path, new_file_path) in zip(old_file_paths, new_file_paths):

        os.rename(old_file_path, new_file_path)