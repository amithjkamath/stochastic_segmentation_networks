from __future__ import absolute_import, division, print_function

# Makes moving python2 to python3 much easier and ensures that nasty bugs involving integer division don't creep in
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import argparse
import sys
import logging
from DICOMLib import DICOMUtils


# ------------------------------------------------------------------------------
# BatchStructureSetConversion
#   Convert structures in structure set to labelmaps and save them to disk
#
class BatchStructureSetConversion(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Batch Structure Set Conversion"
        parent.categories = ["Testing.SlicerRT Tests"]
        parent.dependencies = ["DicomRtImportExport", "Segmentations"]
        parent.contributors = ["Csaba Pinter (Queen's)"]
        parent.helpText = """
    This is a module for converting DICOM structure set to labelmaps and saving them to disk.
    """
        parent.acknowledgementText = """This file was originally developed by Csaba Pinter, PerkLab, Queen's University and was supported through the Applied Cancer Research Unit program of Cancer Care Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care"""  # replace with organization, grant and thanks.
        self.parent = parent

        # Add this test to the SelfTest module's list for discovery when the module
        # is created.  Since this module may be discovered before SelfTests itself,
        # create the list if it doesn't already exist.
        try:
            slicer.selfTests
        except AttributeError:
            slicer.selfTests = {}
        slicer.selfTests["BatchStructureSetConversion"] = self.runTest

    def runTest(self):
        tester = BatchStructureSetConversionTest()
        tester.runTest()


# ------------------------------------------------------------------------------
# BatchStructureSetConversionWidget
#
class BatchStructureSetConversionWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        self.developerMode = True
        ScriptedLoadableModuleWidget.setup(self)


# ------------------------------------------------------------------------------
# BatchStructureSetConversionLogic
#
class BatchStructureSetConversionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget
    """

    def __init__(self, patient_id, modality):
        ScriptedLoadableModuleLogic.__init__(self)

        self.dataDir = slicer.app.temporaryPath + "/BatchStructureSetConversion"
        if not os.access(self.dataDir, os.F_OK):
            os.mkdir(self.dataDir)

        self.patient_id = patient_id

        self.modality = modality

    def LoadFirstPatientIntoSlicer(self):
        # Choose first patient from the patient list
        patient = slicer.dicomDatabase.patients()[0]
        DICOMUtils.loadPatientByUID(patient)

    def ConvertStructureSetToLabelmap(self):
        import vtkSegmentationCorePython as vtkSegmentationCore

        labelmapsToSave = []

        # Get all segmentation nodes from the scene
        segmentationNodes = slicer.util.getNodes("vtkMRMLSegmentationNode*")

        for segmentationNode in segmentationNodes.values():
            logging.info("  Converting structure set " + segmentationNode.GetName())
            # Set referenced volume as rasterization reference
            referenceVolume = slicer.vtkSlicerDicomRtImportExportModuleLogic.GetReferencedVolumeByDicomForSegmentation(
                segmentationNode
            )
            if referenceVolume == None:
                logging.error(
                    "No reference volume found for segmentation "
                    + segmentationNode.GetName()
                )
                continue

            # Perform conversion
            binaryLabelmapRepresentationName = (
                vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
            )
            segmentation = segmentationNode.GetSegmentation()
            segmentation.CreateRepresentation(binaryLabelmapRepresentationName)

            # Create labelmap volume nodes from binary labelmaps
            segmentIDs = vtk.vtkStringArray()
            segmentation.GetSegmentIDs(segmentIDs)

            # Get the correct ending for the saving process
            operatorName = ""
            for segmentIndex in xrange(0, segmentIDs.GetNumberOfValues()):
                segmentID = segmentIDs.GetValue(segmentIndex)
                if segmentID.startswith("CTV"):
                    operatorName = str(segmentID[-2:])

            # Loop through the segmentations
            for segmentIndex in xrange(0, segmentIDs.GetNumberOfValues()):
                segmentID = segmentIDs.GetValue(segmentIndex)
                logging.info("segmentID: " + str(segmentID))
                segment = segmentation.GetSegment(segmentID)
                binaryLabelmap = segment.GetRepresentation(
                    vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
                )
                if not binaryLabelmap:
                    logging.error(
                        "Failed to retrieve binary labelmap from segment "
                        + segmentID
                        + " in segmentation "
                        + segmentationNode.GetName()
                    )
                    continue
                labelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
                slicer.mrmlScene.AddNode(labelmapNode)

                # Get the correct name for the CTV and GTV segmentation
                if "TV" in segmentID:
                    labelmapName = self.patient_id + "_" + segmentID
                else:
                    labelmapName = (
                        self.patient_id + "_" + segmentID + "_" + operatorName
                    )

                labelmapNode.SetName(labelmapName)
                if not slicer.vtkSlicerSegmentationsModuleLogic.CreateLabelmapVolumeFromOrientedImageData(
                    binaryLabelmap, labelmapNode
                ):
                    logging.error(
                        "Failed to create labelmap from segment "
                        + segmentID
                        + " in segmentation "
                        + segmentationNode.GetName()
                    )
                    continue

                # Append volume to list
                labelmapsToSave.append(labelmapNode)

        return labelmapsToSave

    def SaveLabelmaps(self, labelmapsToSave, outputDir):
        for labelmapNode in labelmapsToSave:
            # Clean up file name and set path
            fileName = labelmapNode.GetName() + ".nii.gz"  # Default: '.nrrd'
            charsToRemove = ["!", "?", ":", ";"]
            fileName = fileName.translate(None, "".join(charsToRemove))
            fileName = fileName.replace(" ", "_")
            filePath = outputDir + "/" + fileName
            logging.info(
                "  Saving structure "
                + labelmapNode.GetName()
                + "\n    to file "
                + fileName
            )

            # Save to file
            success = slicer.util.saveNode(labelmapNode, filePath)
            if not success:
                logging.error("Failed to save labelmap: " + filePath)

    def SaveImages(self, outputDir, node_key="vtkMRMLScalarVolumeNode*"):
        # Save all of the ScalarVolumes (or whatever is in node_key) to .nii.gz files
        sv_nodes = slicer.util.getNodes(node_key)
        logging.info(
            "Save image volumes nodes to directory %s: %s"
            % (outputDir, ",".join(sv_nodes.keys()))
        )
        for imageNode in sv_nodes.values():
            # Clean up file name and set path
            fileName = (
                self.patient_id + "_" + self.modality + ".nii.gz"
            )  # Default: '.nrrd' _____ + imageNode.GetName() + '_'
            charsToRemove = ["!", "?", ":", ";"]
            fileName = fileName.translate(None, "".join(charsToRemove))
            fileName = fileName.replace(" ", "_")
            filePath = outputDir + "/" + fileName
            logging.info(
                "  Saving image " + imageNode.GetName() + "\n    to file " + fileName
            )

            # Save to file
            success = slicer.util.saveNode(imageNode, filePath)
            if not success:
                logging.error("Failed to save image volume: " + filePath)


def main(argv):
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Batch Structure Set Conversion")
        parser.add_argument(
            "-i",
            "--input-folder",
            dest="input_folder",
            metavar="PATH",
            default="-",
            required=True,
            help="Folder of input DICOM study (or database path to use existing)",
        )
        parser.add_argument(
            "-x",
            "--exist-db",
            dest="exist_db",
            default=False,
            required=False,
            action="store_true",
            help="Process an existing database",
        )
        parser.add_argument(
            "-e",
            "--export-images",
            dest="export_images",
            default=False,
            required=False,
            action="store_true",
            help="Export image data with labelmaps",
        )
        parser.add_argument(
            "-o",
            "--output-folder",
            dest="output_folder",
            metavar="PATH",
            default=".",
            help="Folder for output labelmaps",
        )
        parser.add_argument(
            "-p",
            "--patient-id",
            dest="patient_id",
            metavar="PATIENT_ID",
            default="0",
            help="The ID of the patient",
        )
        parser.add_argument(
            "-m",
            "--modality",
            dest="modality",
            metavar="MODALITY",
            default="0",
            help="The image modality",
        )

        args = parser.parse_args(argv)

        # Check required arguments
        if args.input_folder == "-":
            logging.warning("Please specify input DICOM study folder!")
        if args.output_folder == ".":
            logging.info(
                "Current directory is selected as output folder (default). To change it, please specify --output-folder"
            )

        # Convert to python path style
        input_folder = args.input_folder.replace("\\", "/")
        output_folder = args.output_folder.replace("\\", "/")
        exist_db = args.exist_db
        export_images = args.export_images

        # Perform batch conversion
        logic = BatchStructureSetConversionLogic(args.patient_id, args.modality)

        def save_rtslices(output_dir):
            # package the saving code into a subfunction
            logging.info("Convert loaded structure set to labelmap volumes")
            labelmaps = logic.ConvertStructureSetToLabelmap()

            logging.info("Save labelmaps to directory " + output_dir)
            logic.SaveLabelmaps(labelmaps, output_dir)
            if export_images:
                logic.SaveImages(output_dir)
            logging.info("DONE")

        if exist_db:
            logging.info("BatchStructureSet running in existing database mode")
            DICOMUtils.openDatabase(input_folder)
            all_patients = slicer.dicomDatabase.patients()
            logging.info("Must Process Patients %s" % len(all_patients))
            for patient in all_patients:
                slicer.mrmlScene.Clear(0)  # clear the scene
                DICOMUtils.loadPatientByUID(patient)
                output_dir = os.path.join(output_folder, patient)
                if not os.access(output_dir, os.F_OK):
                    os.mkdir(output_dir)
                save_rtslices(output_dir)
        else:
            logging.info("Import DICOM data from " + input_folder)
            DICOMUtils.openTemporaryDatabase()
            DICOMUtils.importDicom(input_folder)

            # Get the number of patients in the database
            all_patients = slicer.dicomDatabase.patients()

            for patient in all_patients:

                # Clear the scene
                slicer.mrmlScene.Clear(0)

                # Load the patient
                DICOMUtils.loadPatientByUID(patient)

                # Process and save the segmentations
                save_rtslices(output_folder)

    except Exception as e:
        print(e)

    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
