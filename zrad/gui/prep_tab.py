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
        self.load_dir_button = None
        self.load_dir_label = None
        self.input_data_type_combo_box = None
        self.folder_prefix_label = None
        self.folder_prefix_text_field = None
        self.start_folder_label = None
        self.start_folder_text_field = None
        self.stop_folder_label = None
        self.stop_folder_text_field = None
        self.list_of_patient_folders_label = None
        self.list_of_patient_folders_text_field = None
        self.number_of_threads_combo_box = None
        self.save_dir_button = None
        self.save_dir_label = None
        self.dicom_structures_label = None
        self.dicom_structures_text_field = None
        self.nifti_structures_label = None
        self.nifti_structures_text_field = None
        self.nifti_image_label = None
        self.nifti_image_text_field = None
        self.output_imaging_type_combo_box = None
        self.output_data_type_combo_box = None
        self.ResampleResolutionLabel = None
        self.ResampleResolutionTextField = None
        self.image_interpolation_method_combo_box = None
        self.resample_dimension_combo_box = None
        self.mask_interpolation_method_combo_box = None
        self.mask_interpolation_threshold_label = None
        self.mask_interpolation_threshold_text_field = None
        self.run_button = None

    def show_warning(self, message):
        """
        Display a warning message box and return the user's response.
        """
        response = QMessageBox.warning(self, 'Warning!', message, QMessageBox.Retry | QMessageBox.Retry)
        return response == QMessageBox.Retry

    def run_selected_input(self):
        """
        Validate selections and execute the preprocessing operation.
        """
        # Validate text field inputs
        selections_text = [
            ('', self.load_dir_label.text().strip(), "Select Load Directory!"),
            ('', self.save_dir_label.text().strip(), "Select Save Directory"),
            ('', self.mask_interpolation_threshold_text_field.text().strip(), "Enter Mask Interpolation Threshold"),
            ('', self.ResampleResolutionTextField.text().strip(), "Enter Resample Resolution")
        ]

        for message, text, warning in selections_text:
            if text == message and self.show_warning(warning):
                return

        # Validate combo box selections
        selections_combo_box = [
            ('No. of Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Mask Interpolation:', self.mask_interpolation_method_combo_box),
            ('Save Data as:', self.output_data_type_combo_box),
            ('Image Interpolation:', self.image_interpolation_method_combo_box),
            ('Resample Dimension:', self.resample_dimension_combo_box)
        ]

        for message, comboBox in selections_combo_box:
            if (comboBox.currentText() == message
                    and self.show_warning(f"Select {message.split(':')[0]}")):
                return

        # Collect values from GUI elements
        load_dir = self.load_dir_label.text()
        folder_prefix = self.folder_prefix_text_field.text().strip()
        start_folder = self.start_folder_text_field.text().strip()
        stop_folder = self.stop_folder_text_field.text().strip()

        list_of_patient_folders = []
        if self.list_of_patient_folders_text_field.text() != '':
            list_of_patient_folders = [
                int(pat) for pat in str(self.list_of_patient_folders_text_field.text()).split(",")
            ]

        if (not start_folder or not stop_folder) and not list_of_patient_folders:
            self.show_warning("Incorrectly selected patient folders!")
            return

        save_dir = self.save_dir_label.text()

        number_of_threads = self.number_of_threads_combo_box.currentText().split(" ")[0]
        input_data_type = self.input_data_type_combo_box.currentText()
        dicom_structures = [ROI.strip() for ROI in self.dicom_structures_text_field.text().split(",")]

        if (not self.nifti_image_text_field.text().strip()
                and self.input_data_type_combo_box.currentText() == 'NIFTI'):
            self.show_warning("Enter NIFTI image")
            return
        nifti_image = self.nifti_image_text_field.text()

        # Collect values from GUI elements
        nifti_structures = [ROI.strip() for ROI in self.nifti_structures_text_field.text().split(",")]
        mask_interpolation_method = self.mask_interpolation_method_combo_box.currentText()
        mask_interpolation_threshold = self.mask_interpolation_threshold_text_field.text()
        output_data_type = self.output_data_type_combo_box.currentText()
        resample_resolution = float(self.ResampleResolutionTextField.text())

        if (self.output_imaging_type_combo_box.currentText() == 'Set Imaging as:'
                and self.output_data_type_combo_box.currentText() == 'DICOM'
                and self.show_warning("Select Imaging")):
            return
        output_imaging_type = self.output_imaging_type_combo_box.currentText()

        # Collect values from GUI elements
        image_interpolation_method = self.image_interpolation_method_combo_box.currentText()
        resample_dimension = self.resample_dimension_combo_box.currentText()

    # Create a Preprocessing instance and start the resampling process
        prep_instance = Preprocessing(
            load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders,
            input_data_type, dicom_structures, nifti_image, nifti_structures,
            mask_interpolation_method, mask_interpolation_threshold,
            image_interpolation_method, resample_resolution, resample_dimension,
            save_dir, output_data_type, output_imaging_type, number_of_threads)

        prep_instance.resample()

    def open_directory(self, key):
        """
        Open a directory selection dialog and update the corresponding label.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if directory:
            if key:
                self.load_dir_label.setText(directory)
            else:
                self.save_dir_label.setText(directory)

    def save_input_data(self):
        """
        Save current input data to a JSON file.
        """
        data = {
            'Data Location': self.load_dir_label.text(),
            'Folder Prefix': self.folder_prefix_text_field.text(),
            'Start Folder': self.start_folder_text_field.text(),
            'Stop Folder': self.stop_folder_text_field.text(),
            'List of Patients': self.list_of_patient_folders_text_field.text(),
            'Input Data Type': self.input_data_type_combo_box.currentText(),
            'Save Directory': self.save_dir_label.text(),
            'No. of Threads': self.number_of_threads_combo_box.currentText(),
            'Imaging Modality': self.output_imaging_type_combo_box.currentText(),
            'DICOM Structures': self.dicom_structures_text_field.text(),
            'NIFTI Image': self.nifti_image_text_field.text(),
            'NIFTI Structures': self.nifti_structures_text_field.text(),
            'Resizing': self.ResampleResolutionTextField.text(),
            'Interpolation': self.image_interpolation_method_combo_box.currentText(),
            'Resizing Dim.': self.resample_dimension_combo_box.currentText(),
            'Mask Interpolation Method': self.mask_interpolation_method_combo_box.currentText(),
            'Mask Interpolation Threshold': self.mask_interpolation_threshold_text_field.text(),
            'Output Data Type': self.output_data_type_combo_box.currentText(),
            'Imaging': self.output_imaging_type_combo_box.currentText()

        }
        with open('input/last_saved_user_prep_input.json', 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        """
        Load input data from a JSON file.
        """
        try:
            with open('input/last_saved_user_prep_input.json', 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('Data Location', ''))
                self.folder_prefix_text_field.setText(data.get('Folder Prefix', ''))
                self.start_folder_text_field.setText(data.get('Start Folder', ''))
                self.stop_folder_text_field.setText(data.get('Stop Folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('List of Patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('Input Data Type', ''))
                self.save_dir_label.setText(data.get('Save Directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('No. of Threads', ''))
                self.output_imaging_type_combo_box.setCurrentText(data.get('Imaging Modality', ''))
                self.dicom_structures_text_field.setText(data.get('DICOM Structures', ''))
                self.nifti_image_text_field.setText(data.get('NIFTI Image', ''))
                self.nifti_structures_text_field.setText(data.get('NIFTI Structures', ''))
                self.ResampleResolutionTextField.setText(data.get('Resizing', ''))
                self.image_interpolation_method_combo_box.setCurrentText(data.get('Interpolation', ''))
                self.resample_dimension_combo_box.setCurrentText(data.get('Resizing Dim.', ''))
                self.mask_interpolation_method_combo_box.setCurrentText(data.get('Mask Interpolation Method', ''))
                self.mask_interpolation_threshold_text_field.setText(data.get('Mask Interpolation Threshold', ''))
                self.output_data_type_combo_box.setCurrentText(data.get('Output Data Type', ''))
        except FileNotFoundError:
            print("No previous data found!")

    def init_tab(self):
        # Create a QVBoxLayout for the tab layout
        self.layout = QVBoxLayout(self)

        # Load Directory Button and Label
        self.load_dir_button = CustomButton(
            'Load Directory',
            14, 30, 50, 200, 50, self,
            style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;"
        )
        self.load_dir_label = CustomLabel('', 14, 300, 50, 1400, 50, self)
        self.load_dir_label.setAlignment(Qt.AlignCenter)
        self.load_dir_button.clicked.connect(lambda: self.open_directory(key=True))

        # Input Data Type ComboBox
        self.input_data_type_combo_box = CustomBox(
            14, 60, 300, 140, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIFTI"
            ]
        )
        self.input_data_type_combo_box.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        # Folder Prefix TextField and Label
        self.folder_prefix_label = CustomLabel(
            'Prefix:',
            18, 320, 140, 150, 50, self,
            style="color: white;"
        )
        self.folder_prefix_text_field = CustomTextField(
            "Enter...",
            14, 400, 140, 100, 50, self
        )

        # Start and Stop Folder TextFields and Labels
        self.start_folder_label = CustomLabel(
            'Start:',
            18, 520, 140, 150, 50, self,
            style="color: white;"
        )
        self.start_folder_text_field = CustomTextField(
            "Enter...",
            14, 590, 140, 100, 50, self
        )
        self.stop_folder_label = CustomLabel(
            'Stop:',
            18, 710, 140, 150, 50, self,
            style="color: white;")
        self.stop_folder_text_field = CustomTextField(
            "Enter...",
            14, 775, 140, 100, 50, self
        )

        # List of Patient Folders TextField and Label
        self.list_of_patient_folders_label = CustomLabel(
            'List of Patients:',
            18, 900, 140, 200, 50, self,
            style="color: white;"
        )
        self.list_of_patient_folders_text_field = CustomTextField(
            "E.g. 1, 5, 10, 34...",
            14, 1080, 140, 350, 50, self)

        # Number of Threads ComboBox
        no_of_threads = ['No. of Threads:']
        for core in range(multiprocessing.cpu_count()):
            if core == 0:
                no_of_threads.append(str(core + 1) + " thread")
            else:
                no_of_threads.append(str(core + 1) + " threads")
        self.number_of_threads_combo_box = CustomBox(
            14, 1450, 140, 210, 50, self,
            item_list=no_of_threads
        )

        # Save Directory Button and Label
        self.save_dir_button = CustomButton(
            'Save Directory',
            14, 30, 220, 200, 50, self,
            style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;"
        )
        self.save_dir_label = CustomLabel(
            '',
            14, 300, 220, 1400, 50, self
        )
        self.save_dir_label.setAlignment(Qt.AlignCenter)
        self.save_dir_button.clicked.connect(lambda: self.open_directory(key=False))

        # DICOM and NIFTI Structures TextFields and Labels
        self.dicom_structures_label = CustomLabel(
            'Studied str.:',
            18, 370, 300, 200, 50, self,
            style="color: white;"
        )
        self.dicom_structures_text_field = CustomTextField(
            "E.g. CTV, liver...",
            14, 510, 300, 475, 50, self
        )
        self.dicom_structures_label.hide()
        self.dicom_structures_text_field.hide()

        self.nifti_structures_label = CustomLabel(
            'NIFTI Str. Files:',
            18, 370, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_structures_text_field = CustomTextField(
            "E.g. CTV, liver...",
            14, 540, 300, 250, 50, self
        )
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()

        self.nifti_image_label = CustomLabel(
            'NIFTI Image File:',
            18, 800, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT.nii.gz",
            14, 990, 300, 220, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()

        # Output Imaging Type ComboBox
        self.output_imaging_type_combo_box = CustomBox(
            14, 1450, 300, 210, 50, self,
            item_list=[
                "Set Imaging as:", "CT", "MR", "PT"
            ]
        )
        self.output_imaging_type_combo_box.hide()

        # Output Data Type ComboBox
        self.output_data_type_combo_box = CustomBox(
            14, 1450 - 230, 300, 210, 50, self,
            item_list=[
                "Save Data as:", "DICOM", "NIFTI"
            ]
        )
        self.output_data_type_combo_box.currentTextChanged.connect(
            lambda:
                self.output_imaging_type_combo_box.show()
                if self.output_data_type_combo_box.currentText() == "DICOM"
                else
                self.output_imaging_type_combo_box.hide()
            )

        # Resample Resolution Label and TextField
        self.ResampleResolutionLabel = CustomLabel(
            'Resize Resolution (mm):',
            18, 370, 380, 300, 50, self,
            style="color: white;"
        )
        self.ResampleResolutionTextField = CustomTextField(
            "E.g. 1", 14, 650, 380, 100, 50, self
        )

        # Image Interpolation Method ComboBox
        self.image_interpolation_method_combo_box = CustomBox(
            14, 775, 380, 210, 50, self,
            item_list=[
                'Image Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Resample Dimension ComboBox
        self.resample_dimension_combo_box = CustomBox(
            14, 1000, 380, 210, 50, self,
            item_list=[
                'Resample Dimension:', "2D", "3D"
            ]
        )

        # Mask Interpolation Method ComboBox
        self.mask_interpolation_method_combo_box = CustomBox(
            14, 370, 460, 210, 50, self,
            item_list=[
                'Mask Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Mask Interpolation Threshold Label and TextField
        self.mask_interpolation_threshold_label = CustomLabel(
            'Mask Interpolation Threshold:',
            18, 600, 460, 360, 50, self,
            style="color: white;"
        )
        self.mask_interpolation_threshold_text_field = CustomTextField(
            "E.g. 0.75",
            14, 930, 460, 100, 50, self
        )
        self.mask_interpolation_threshold_text_field.setText('0.5')
        self.mask_interpolation_threshold_label.hide()
        self.mask_interpolation_threshold_text_field.hide()
        self.mask_interpolation_method_combo_box.currentTextChanged.connect(
            lambda:
                (
                    self.mask_interpolation_threshold_label.show(),
                    self.mask_interpolation_threshold_text_field.show()
                )
                if self.mask_interpolation_method_combo_box.currentText() not in ['NN', 'Mask Interpolation:']
                else (
                    self.mask_interpolation_threshold_label.hide(),
                    self.mask_interpolation_threshold_text_field.hide()
                )
        )

        # Run Button
        self.run_button = CustomButton(
            'Run',
            20, 910, 590, 80, 50, self,
            style=False
        )
        self.run_button.clicked.connect(self.run_selected_input)

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the ComboBox's value is changed
        if text == 'DICOM':
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.nifti_structures_label.hide()
            self.nifti_structures_text_field.hide()
        elif text == 'NIFTI':
            self.nifti_image_label.show()
            self.nifti_image_text_field.show()
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_structures_label.show()
            self.nifti_structures_text_field.show()
        else:
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.nifti_structures_label.hide()
            self.nifti_structures_text_field.hide()
