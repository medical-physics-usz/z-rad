from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
import multiprocessing
import json

from gui.toolbox_gui import (CustomButton, CustomLabel, CustomBox, CustomTextField)
from logic.preprocessing import Preprocessing


class PreprocessingTab(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = None
        self.LoadDirButton = None
        self.LoadDirLabel = None
        self.InputDataTypeComboBox = None
        self.FolderPrefixLabel = None
        self.FolderPrefixTextField = None
        # Set start_folder folder
        self.StartFolderLabel = None
        self.StartFolderTextField = None
        # Set stop_folder folder
        self.StopFolderLabel = None
        self.StopFolderTextField = None
        # Set a list of studied patients
        self.ListOfPatientFoldersLabel = None
        self.ListOfPatientFoldersTextField = None
        self.NumberOfThreadsComboBox = None
        # Set save directory
        self.SaveDirButton = None
        self.SaveDirLabel = None
        # Set studied structures
        self.DicomStructuresLabel = None
        self.DicomStructuresTextField = None
        self.NiftiStructuresLabel = None
        self.NiftiStructuresTextField = None
        self.NiftiImageLabel = None
        self.NiftiImageTextField = None
        self.OutputImagingTypeComboBox = None
        self.OutputDataTypeComboBox = None
        self.ResampleResolutionLabel = None
        self.ResampleResolutionTextField = None
        self.ImageInterpolationMethodComboBox = None
        self.ResampleDimensionComboBox = None
        self.MaskInterpolationMethodComboBox = None
        self.MaskInterpolationThresholdLabel = None
        self.MaskInterpolationThresholdTextField = None
        self.RunButton = None

    def show_warning(self, message):
        response = QMessageBox.warning(self, 'Warning!', message, QMessageBox.Retry | QMessageBox.Retry)
        return response == QMessageBox.Retry

    def run_selected_option(self):
        selections_text = [
            ('', self.LoadDirLabel.text().strip(), "Select Load Directory!"),
            ('', self.SaveDirLabel.text().strip(), "Select Save Directory"),
            ('', self.MaskInterpolationThresholdTextField.text().strip(), "Enter Mask Interpolation Threshold"),
            ('', self.ResampleResolutionTextField.text().strip(), "Enter Resample Resolution")]

        for message, text, warning in selections_text:
            if text == message and self.show_warning(warning):
                return

        selections_combo_box = [
            ('No. of Threads:', self.NumberOfThreadsComboBox),
            ('Data Type:', self.InputDataTypeComboBox),
            ('Mask Interpolation:', self.MaskInterpolationMethodComboBox),
            ('Save Data as:', self.OutputDataTypeComboBox),
            ('Image Interpolation:', self.ImageInterpolationMethodComboBox),
            ('Resample Dimension:', self.ResampleDimensionComboBox)]

        for message, comboBox in selections_combo_box:
            if comboBox.currentText() == message and self.show_warning(f"Select {message.split(':')[0]}"):
                return

        load_dir = self.LoadDirLabel.text()
        folder_prefix = self.FolderPrefixTextField.text().strip()
        start_folder = self.StartFolderTextField.text().strip()
        stop_folder = self.StopFolderTextField.text().strip()

        list_of_patient_folders = []
        if self.ListOfPatientFoldersTextField.text() != '':
            list_of_patient_folders = [int(pat) for pat in str(self.ListOfPatientFoldersTextField.text()).split(",")]

        if (not start_folder or not stop_folder) and not list_of_patient_folders:
            self.show_warning("Incorrectly selected patient folders!")
            return

        save_dir = self.SaveDirLabel.text()

        number_of_threads = self.NumberOfThreadsComboBox.currentText().split(" ")[0]
        input_data_type = self.InputDataTypeComboBox.currentText()
        dicom_structures = [ROI.strip() for ROI in self.DicomStructuresTextField.text().split(",")]

        if (not self.NiftiImageTextField.text().strip() and
                self.InputDataTypeComboBox.currentText() == 'NIFTI'):
            self.show_warning("Enter NIFTI image")
            return
        nifti_image = self.NiftiImageTextField.text()

        nifti_structures = [ROI.strip() for ROI in self.NiftiStructuresTextField.text().split(",")]
        mask_interpolation_method = self.MaskInterpolationMethodComboBox.currentText()
        mask_interpolation_threshold = self.MaskInterpolationThresholdTextField.text()
        output_data_type = self.OutputDataTypeComboBox.currentText()
        resample_resolution = float(self.ResampleResolutionTextField.text())

        if (self.OutputImagingTypeComboBox.currentText() == 'Set Imaging as:'
                and self.OutputDataTypeComboBox.currentText() == 'DICOM' and self.show_warning("Select Imaging")):
            return
        output_imaging_type = self.OutputImagingTypeComboBox.currentText()

        image_interpolation_method = self.ImageInterpolationMethodComboBox.currentText()
        resample_dimension = self.ResampleDimensionComboBox.currentText()

        print({'Data location': load_dir, 'Folder folder_prefix': folder_prefix, 'Start folder': start_folder,
               'Stop folder': stop_folder,
               'Save Directory': save_dir, 'List of patients': list_of_patient_folders,
                'Names of the studied DICOM structures': dicom_structures, 'NIFTI image': nifti_image,
               'NIFTI mask': nifti_structures, 'Mask interpolation': mask_interpolation_method,
               'Mask threshold': mask_interpolation_threshold, 'Texture Resolution': resample_resolution,
               'Data type': input_data_type, '# of cores': number_of_threads, 'Imaging modality': output_imaging_type,
               'Interpolation method': image_interpolation_method, 'Data Dim': resample_dimension,
               'Save As': output_data_type})

        prep = Preprocessing(load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders,
                             input_data_type, dicom_structures,
                             nifti_image, nifti_structures, mask_interpolation_method, mask_interpolation_threshold,
                            image_interpolation_method, resample_resolution,
                             resample_dimension, save_dir, output_data_type, output_imaging_type, number_of_threads)
        prep.resample()

    def open_directory(self, key):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if directory and key:
            self.LoadDirLabel.setText(directory)
        elif directory and not key:
            self.SaveDirLabel.setText(directory)

    def save_input_data(self):
        data = {
            'Data location': self.LoadDirLabel.text(),
            'Folder folder_prefix': self.FolderPrefixTextField.text(),
            'Start folder': self.StartFolderTextField.text(),
            'Stop folder': self.StopFolderTextField.text(),
            'List of patients': self.ListOfPatientFoldersTextField.text(),
            'Data Type': self.InputDataTypeComboBox.currentText(),
            'Save directory': self.SaveDirLabel.text(),
            '# of cores': self.NumberOfThreadsComboBox.currentText(),
            'Imaging modality': self.OutputImagingTypeComboBox.currentText(),
            'Studied structures': self.DicomStructuresTextField.text(),
            'NIFTI image': self.NiftiImageTextField.text(),
            'NIFTI mask': self.NiftiStructuresTextField.text(),
            'Resizing': self.ResampleResolutionTextField.text(),
            'Interpolation': self.ImageInterpolationMethodComboBox.currentText(),
            'Resizing dim': self.ResampleDimensionComboBox.currentText(),
            'Mask interpolation method': self.MaskInterpolationMethodComboBox.currentText(),
            'Mask interpolation threshold': self.MaskInterpolationThresholdTextField.text(),
            'Save as': self.OutputDataTypeComboBox.currentText(),
            'Imaging': self.OutputImagingTypeComboBox.currentText()

        }
        with open('input/last_saved_user_prep_input.json', 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        try:
            with open('input/last_saved_user_prep_input.json', 'r') as file:
                data = json.load(file)
                self.LoadDirLabel.setText(data.get('Data location', ''))
                self.FolderPrefixTextField.setText(data.get('Folder folder_prefix', ''))
                self.StartFolderTextField.setText(data.get('Start folder', ''))
                self.StopFolderTextField.setText(data.get('Stop folder', ''))
                self.ListOfPatientFoldersTextField.setText(data.get('List of patients', ''))
                self.InputDataTypeComboBox.setCurrentText(data.get('Data Type', ''))
                self.SaveDirLabel.setText(data.get('Save directory', ''))
                self.NumberOfThreadsComboBox.setCurrentText(data.get('# of cores', ''))
                self.OutputImagingTypeComboBox.setCurrentText(data.get('Imaging modality', ''))
                self.DicomStructuresTextField.setText(data.get('Studied structures', ''))
                self.NiftiImageTextField.setText(data.get('NIFTI image', ''))
                self.NiftiStructuresTextField.setText(data.get('NIFTI mask', ''))
                self.ResampleResolutionTextField.setText(data.get('Resizing', ''))
                self.ImageInterpolationMethodComboBox.setCurrentText(data.get('Interpolation', ''))
                self.ResampleDimensionComboBox.setCurrentText(data.get('Resizing dim', ''))
                self.MaskInterpolationMethodComboBox.setCurrentText(data.get('Mask interpolation method', ''))
                self.MaskInterpolationThresholdTextField.setText(data.get('Mask interpolation threshold', ''))
                self.OutputDataTypeComboBox.setCurrentText(data.get('Save as', ''))
                self.OutputImagingTypeComboBox.setCurrentText(data.get('Save as', ''))
        except FileNotFoundError:
            print("No previous data found!")

    def initUI(self):
        # Create a QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Path to load the files
        self.LoadDirButton = CustomButton('Load Directory', 14, 30, 50, 200, 50, self,
        style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;")
        self.LoadDirLabel = CustomLabel('', 14, 300, 50, 1400, 50, self)
        self.LoadDirLabel.setAlignment(Qt.AlignCenter)

        self.LoadDirButton.clicked.connect(lambda: self.open_directory(key=True))

        # Set used data type
        self.InputDataTypeComboBox = CustomBox(14, 60, 300, 140, 50, self, item_list=["Data Type:", "DICOM", "NIFTI"])

        self.InputDataTypeComboBox.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        # Set folder_prefix
        self.FolderPrefixLabel = CustomLabel('Prefix:', 18, 320, 140, 150, 50, self, style="color: white;")
        self.FolderPrefixTextField = CustomTextField("Enter...", 14, 400, 140, 100, 50, self)

        # Set start_folder folder
        self.StartFolderLabel = CustomLabel('Start:', 18, 520, 140, 150, 50, self, style="color: white;")
        self.StartFolderTextField = CustomTextField("Enter...", 14, 590, 140, 100, 50, self)

        # Set stop_folder folder
        self.StopFolderLabel = CustomLabel('Stop:', 18, 710, 140, 150, 50, self, style="color: white;")
        self.StopFolderTextField = CustomTextField("Enter...", 14, 775, 140, 100, 50, self)

        # Set a list of studied patients
        self.ListOfPatientFoldersLabel = CustomLabel('List of Patients:', 18, 900, 140,
                                                     200, 50, self, style="color: white;")
        self.ListOfPatientFoldersTextField = CustomTextField("E.g. 1, 5, 10, 34...", 14, 1080, 140,
                                                             350, 50, self)
        # Set # of used cores
        no_of_threads = ['No. of Threads:']
        for core in range(multiprocessing.cpu_count()):
            if core == 0:
                no_of_threads.append(str(core + 1) + " thread")
            else:
                no_of_threads.append(str(core + 1) + " threads")
        self.NumberOfThreadsComboBox = CustomBox(14, 1450, 140, 210, 50, self, item_list=no_of_threads)

        # Set save directory
        self.SaveDirButton = CustomButton('Save Directory', 14, 30, 220, 200, 50, self,
                                    style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;")
        self.SaveDirLabel = CustomLabel('', 14, 300, 220, 1400, 50, self)
        self.SaveDirLabel.setAlignment(Qt.AlignCenter)
        self.SaveDirButton.clicked.connect(lambda: self.open_directory(key=False))

        # Set studied structures
        self.DicomStructuresLabel = CustomLabel('Studied str.:', 18, 370, 300, 200, 50, self, style="color: white;")
        self.DicomStructuresTextField = CustomTextField("E.g. CTV, liver...", 14, 510, 300, 475, 50, self)
        self.DicomStructuresLabel.hide()
        self.DicomStructuresTextField.hide()

        self.NiftiStructuresLabel = CustomLabel('NIFTI Str. Files:', 18,
                                                370, 300, 200, 50, self, style="color: white;")
        self.NiftiStructuresTextField = CustomTextField("E.g. CTV, liver...", 14,
                                                        540, 300, 250, 50, self)
        self.NiftiStructuresLabel.hide()
        self.NiftiStructuresTextField.hide()

        # WHAT?
        self.NiftiImageLabel = CustomLabel('NIFTI Image File:', 18, 800, 300, 200, 50, self, style="color: white;")
        self.NiftiImageTextField = CustomTextField("E.g. imageCT.nii.gz", 14, 990, 300, 220, 50, self)
        self.NiftiImageLabel.hide()
        self.NiftiImageTextField.hide()

        # Set output_imaging_type
        self.OutputImagingTypeComboBox = CustomBox(14, 1450, 300, 210, 50,
                                                   self, item_list=["Set Imaging as:", "CT", "MR", "PT"])
        self.OutputImagingTypeComboBox.hide()
        self.OutputDataTypeComboBox = CustomBox(14, 1450 - 230, 300, 210, 50,
                                                self, item_list=["Save Data as:", "DICOM", "NIFTI"])
        self.OutputDataTypeComboBox.currentTextChanged.connect(self.save_box_changed)

        self.ResampleResolutionLabel = CustomLabel('Resize Resolution (mm):', 18,
                                                   370, 380, 300, 50, self, style="color: white;")
        self.ResampleResolutionTextField = CustomTextField("E.g. 1", 14,
                                                           650, 380, 100, 50, self)

        self.ImageInterpolationMethodComboBox = CustomBox(14, 775, 380, 210, 50, self,
                                                          item_list=['Image Interpolation:', "NN", "Linear", "BSpline", "Gaussian"])

        self.ResampleDimensionComboBox = CustomBox(14, 1000, 380, 210, 50, self, item_list=['Resample Dimension:', "2D", "3D"])

        self.MaskInterpolationMethodComboBox = CustomBox(14, 370, 460, 210, 50, self,
                                                         item_list=['Mask Interpolation:', "NN", "Linear", "BSpline", "Gaussian"])

        self.MaskInterpolationThresholdLabel = CustomLabel('Mask Interpolation Threshold:', 18,
                                                           600, 460, 360, 50, self, style="color: white;")
        self.MaskInterpolationThresholdTextField = CustomTextField("E.g. 0.75", 14, 930, 460, 100, 50, self)
        self.MaskInterpolationThresholdTextField.setText('0.5')
        self.MaskInterpolationThresholdLabel.hide()
        self.MaskInterpolationThresholdTextField.hide()
        self.MaskInterpolationMethodComboBox.currentTextChanged.connect(lambda:
            (self.MaskInterpolationThresholdLabel.show(), self.MaskInterpolationThresholdTextField.show())
            if self.MaskInterpolationMethodComboBox.currentText() != 'NN'
               and self.MaskInterpolationMethodComboBox.currentText() != 'Mask Interpolation:'
            else (self.MaskInterpolationThresholdLabel.hide(), self.MaskInterpolationThresholdTextField.hide()))

        self.RunButton = CustomButton('Run', 20, 910, 590, 80, 50, self, style=None)
        self.RunButton.clicked.connect(self.run_selected_option)

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the combo box's value is changed
        if text == 'DICOM':
            self.NiftiImageLabel.hide()
            self.NiftiImageTextField.hide()
            self.DicomStructuresLabel.show()
            self.DicomStructuresTextField.show()
            self.NiftiStructuresLabel.hide()
            self.NiftiStructuresTextField.hide()
        elif text == 'NIFTI':
            self.NiftiImageLabel.show()
            self.NiftiImageTextField.show()
            self.DicomStructuresLabel.hide()
            self.DicomStructuresTextField.hide()
            self.NiftiStructuresLabel.show()
            self.NiftiStructuresTextField.show()

        else:
            self.DicomStructuresLabel.hide()
            self.DicomStructuresTextField.hide()
            self.NiftiImageLabel.hide()
            self.NiftiImageTextField.hide()
            self.NiftiStructuresLabel.hide()
            self.NiftiStructuresTextField.hide()

    def save_box_changed(self, text):
        if text == 'DICOM':
            self.OutputImagingTypeComboBox.show()
        elif text == 'NIFTI':
            self.OutputImagingTypeComboBox.hide()
        else:
            self.OutputImagingTypeComboBox.hide()
