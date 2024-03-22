import json

# Import required PyQt5 modules for GUI creation
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from joblib import cpu_count

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomWarningBox, resource_path
from ..logic.preprocessing import Preprocessing


class PreprocessingTab(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = None
        self.load_dir_button = None
        self.load_dir_label = None
        self.input_data_type_combo_box = None
        self.input_imaging_mod_combo_box = None
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
        self.resample_resolution_label = None
        self.resample_resolution_text_field = None
        self.image_interpolation_method_combo_box = None
        self.resample_dimension_combo_box = None
        self.mask_interpolation_method_combo_box = None
        self.mask_interpolation_threshold_label = None
        self.mask_interpolation_threshold_text_field = None
        self.run_button = None

    def run_selected_input(self):
        """
        Validate selections and execute the preprocessing operation.
        """
        # Validate text field inputs
        selections_text = [
            ('', self.load_dir_label.text().strip(), "Select Load Directory!"),
            ('', self.save_dir_label.text().strip(), "Select Save Directory"),
            ('', self.mask_interpolation_threshold_text_field.text().strip(), "Enter Mask Interpolation Threshold"),
            ('', self.resample_resolution_text_field.text().strip(), "Enter Resample Resolution")
        ]

        for message, text, warning in selections_text:
            if text == message and CustomWarningBox(warning).response():
                return

        # Validate combo box selections
        selections_combo_box = [
            ('No. of Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Mask Interpolation:', self.mask_interpolation_method_combo_box),
            ('Image Interpolation:', self.image_interpolation_method_combo_box),
            ('Resample Dimension:', self.resample_dimension_combo_box)
        ]

        for message, combo_box in selections_combo_box:
            if (combo_box.currentText() == message
                    and CustomWarningBox(f"Select {message.split(':')[0]}").response()):
                return

        # Collect values from GUI elements
        load_dir = self.load_dir_label.text()
        save_dir = self.save_dir_label.text()
        start_folder = self.start_folder_text_field.text().strip()
        stop_folder = self.stop_folder_text_field.text().strip()

        list_of_patient_folders = []
        if self.list_of_patient_folders_text_field.text() != '':
            list_of_patient_folders = [
                int(pat) for pat in str(self.list_of_patient_folders_text_field.text()).split(",")
            ]

        if (not start_folder or not stop_folder) and not list_of_patient_folders:
            CustomWarningBox("Incorrectly selected patient folders!").response()
            return

        number_of_threads = int(self.number_of_threads_combo_box.currentText().split(" ")[0])
        input_data_type = self.input_data_type_combo_box.currentText()
        dicom_structures = [ROI.strip() for ROI in self.dicom_structures_text_field.text().split(",")]

        if (not self.nifti_image_text_field.text().strip()
                and self.input_data_type_combo_box.currentText() == 'NIFTI'):
            CustomWarningBox("Enter NIFTI image").response()
            return
        nifti_image = self.nifti_image_text_field.text()

        # Collect values from GUI elements
        nifti_structures = [ROI.strip() for ROI in self.nifti_structures_text_field.text().split(",")]
        mask_interpolation_method = self.mask_interpolation_method_combo_box.currentText()
        mask_interpolation_threshold = float(self.mask_interpolation_threshold_text_field.text())
        resample_resolution = float(self.resample_resolution_text_field.text())

        if (self.input_imaging_mod_combo_box.currentText() == 'Imaging Mod.:'
                and CustomWarningBox("Select Input Imaging Modality").response()):
            return
        input_imaging_mod = self.input_imaging_mod_combo_box.currentText()

        # Collect values from GUI elements
        image_interpolation_method = self.image_interpolation_method_combo_box.currentText()
        resample_dimension = self.resample_dimension_combo_box.currentText()

        structure_set = []
        if input_data_type == 'DICOM':
            structure_set = dicom_structures
        elif input_data_type == 'NIFTI':
            structure_set = nifti_structures

    # Create a Preprocessing instance
        prep_instance = Preprocessing(
            load_dir=load_dir,
            save_dir=save_dir,
            input_data_type=input_data_type,
            input_imaging_mod=input_imaging_mod,
            resample_resolution=resample_resolution,
            resample_dimension=resample_dimension,
            image_interpolation_method=image_interpolation_method,
            mask_interpolation_method=mask_interpolation_method,
            mask_interpolation_threshold=mask_interpolation_threshold,
            start_folder=start_folder,
            stop_folder=stop_folder,
            list_of_patient_folders=list_of_patient_folders,
            structure_set=structure_set,
            nifti_image=nifti_image,
            number_of_threads=number_of_threads)

    # Start the resampling process
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
            'Start Folder': self.start_folder_text_field.text(),
            'Stop Folder': self.stop_folder_text_field.text(),
            'List of Patients': self.list_of_patient_folders_text_field.text(),
            'Input Data Type': self.input_data_type_combo_box.currentText(),
            'Save Directory': self.save_dir_label.text(),
            'No. of Threads': self.number_of_threads_combo_box.currentText(),
            'DICOM Structures': self.dicom_structures_text_field.text(),
            'NIFTI Image': self.nifti_image_text_field.text(),
            'NIFTI Structures': self.nifti_structures_text_field.text(),
            'Resizing': self.resample_resolution_text_field.text(),
            'Interpolation': self.image_interpolation_method_combo_box.currentText(),
            'Resizing Dim.': self.resample_dimension_combo_box.currentText(),
            'Mask Interpolation Method': self.mask_interpolation_method_combo_box.currentText(),
            'Mask Interpolation Threshold': self.mask_interpolation_threshold_text_field.text(),
            'input mod': self.input_imaging_mod_combo_box.currentText()

        }
        file_path = resource_path('zrad/input/last_saved_prep_user_input.json')
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        """
        Load input data from a JSON file.
        """
        file_path = resource_path('zrad/input/last_saved_prep_user_input.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('Data Location', ''))
                self.start_folder_text_field.setText(data.get('Start Folder', ''))
                self.stop_folder_text_field.setText(data.get('Stop Folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('List of Patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('Input Data Type', ''))
                self.save_dir_label.setText(data.get('Save Directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('No. of Threads', ''))
                self.dicom_structures_text_field.setText(data.get('DICOM Structures', ''))
                self.nifti_image_text_field.setText(data.get('NIFTI Image', ''))
                self.nifti_structures_text_field.setText(data.get('NIFTI Structures', ''))
                self.resample_resolution_text_field.setText(data.get('Resizing', ''))
                self.image_interpolation_method_combo_box.setCurrentText(data.get('Interpolation', ''))
                self.resample_dimension_combo_box.setCurrentText(data.get('Resizing Dim.', ''))
                self.mask_interpolation_method_combo_box.setCurrentText(data.get('Mask Interpolation Method', ''))
                self.mask_interpolation_threshold_text_field.setText(data.get('Mask Interpolation Threshold', ''))
                self.input_imaging_mod_combo_box.setCurrentText(data.get('input mod', ''))

        except FileNotFoundError:
            print("No previous data found!")

    def init_tab(self):
        # Create a QVBoxLayout for the tab layout
        self.layout = QVBoxLayout(self)

        # Load Directory Button and Label
        self.load_dir_button = CustomButton(
            'Load Directory',
            14, 30, 50, 200, 50, self,
            style=True)
        self.load_dir_label = CustomLabel('', 14, 300, 50, 1400, 50, self)
        self.load_dir_label.setAlignment(Qt.AlignCenter)
        self.load_dir_button.clicked.connect(lambda: self.open_directory(key=True))

        # Input Data Type ComboBox
        self.input_data_type_combo_box = CustomBox(
            14, 40, 300, 160, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIFTI"
            ]
        )
        self.input_data_type_combo_box.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        #  Start and Stop Folder TextFields and Labels
        self.start_folder_label = CustomLabel(
            'Start Folder:',
            18, 520, 140, 150, 50, self,
            style="color: white;"
        )
        self.start_folder_text_field = CustomTextField(
            "Enter...",
            14, 660, 140, 100, 50, self
        )
        self.stop_folder_label = CustomLabel(
            'Stop Folder:',
            18, 780, 140, 150, 50, self,
            style="color: white;")
        self.stop_folder_text_field = CustomTextField(
            "Enter...",
            14, 920, 140, 100, 50, self
        )

        # List of Patient Folders TextField and Label
        self.list_of_patient_folders_label = CustomLabel(
            'List of Folders:',
            18, 1050, 140, 210, 50, self,
            style="color: white;"
        )
        self.list_of_patient_folders_text_field = CustomTextField(
            "E.g. 1, 5, 10, 34...",
            14, 1220, 140, 210, 50, self)

        # Number of Threads ComboBox
        no_of_threads = ['No. of Threads:']
        for core in range(cpu_count()):
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
            style=True)
        self.save_dir_label = CustomLabel(
            '',
            14, 300, 220, 1400, 50, self
        )
        self.save_dir_label.setAlignment(Qt.AlignCenter)
        self.save_dir_button.clicked.connect(lambda: self.open_directory(key=False))

        self.input_imaging_mod_combo_box = CustomBox(
            14, 320, 140, 170, 50, self,
            item_list=[
                "Imaging Mod.:", "CT", "MR", "PT"
            ]
        )

        # DICOM and NIFTI Structures TextFields and Labels
        self.dicom_structures_label = CustomLabel(
            'Studied str.:',
            18, 595, 300, 200, 50, self,
            style="color: white;"
        )
        self.dicom_structures_text_field = CustomTextField(
            "E.g. CTV, liver...",
            14, 735, 300, 475, 50, self
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
            14, 550, 300, 220, 50, self
        )
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()

        self.nifti_image_label = CustomLabel(
            'NIFTI Image File:',
            18, 790, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT.nii.gz",
            14, 990, 300, 220, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()

        # Resample Resolution Label and TextField
        self.resample_resolution_label = CustomLabel(
            'Resample Resolution (mm):',
            18, 370, 380, 300, 50, self,
            style="color: white;"
        )
        self.resample_resolution_text_field = CustomTextField(
            "E.g. 1", 14, 675, 380, 90, 50, self
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
        # This slot will be called whenever the file type combobox value is changed
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
