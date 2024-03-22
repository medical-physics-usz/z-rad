import json
import multiprocessing

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomCheckBox, CustomWarningBox, \
    resource_path
from ..logic.radiomics import Radiomics


class RadiomicsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = None
        self.load_dir_button = None
        self.load_dir_label = None
        self.input_data_type_combo_box = None
        self.folder_prefix_label = None
        self.start_folder_label = None
        self.start_folder_text_field = None
        self.stop_folder_label = None
        self.stop_folder_text_field = None
        self.list_of_patient_folders_label = None
        self.list_of_patient_folders_text_field = None
        self.number_of_threads_label = None
        self.number_of_threads_combo_box = None
        self.save_dir_button = None
        self.save_dir_label = None
        self.dicom_structures_label = None
        self.dicom_structures_text_field = None
        self.nifti_structures_label = None
        self.nifti_structure_text_field = None
        self.nifti_image_label = None
        self.nifti_image_text_field = None
        self.intensity_range_label = None
        self.intensity_range_text_field = None
        self.input_imaging_mod_combo_box = None
        self.outlier_detection_check_box = None
        self.outlier_detection_label = None
        self.outlier_detection_text_field = None
        self.intensity_range_check_box = None
        self.discretization_combo_box = None
        self.bin_size_text_field = None
        self.bin_number_text_field = None
        self.aggr_dim_and_method_combo_box = None
        self.weighting_combo_box = None

        self.run_button = None

    def run_selected_option(self):

        selections_text = [
            ('', self.load_dir_label.text().strip(), "Select Load Directory!"),
            ('', self.save_dir_label.text().strip(), "Select Save Directory"),
        ]

        for message, text, warning in selections_text:
            if text == message and CustomWarningBox(warning).response():
                return

        # Validate combo box selections
        selections_combo_box = [
            ('No. of Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Discretization:', self.discretization_combo_box),
            ('Texture Features Aggr. Method:', self.aggr_dim_and_method_combo_box),
            ('Imaging Mod.:', self.input_imaging_mod_combo_box)
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

        input_imaging_mod = self.input_imaging_mod_combo_box.currentText()
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
        nifti_structures = [ROI.strip() for ROI in self.nifti_structure_text_field.text().split(",")]
        intensity_range = ''
        if self.intensity_range_check_box.isChecked():
            if self.intensity_range_text_field.text() == '':
                CustomWarningBox("Enter intensity range").response()
                return
            intensity_range = [np.inf if intensity.strip() == '' else float(intensity)
                               for intensity in self.intensity_range_text_field.text().split(',')]
        outlier_range = ''
        if self.outlier_detection_check_box.isChecked():
            if self.outlier_detection_text_field.text() == '':
                CustomWarningBox("Enter Confidence Interval").response()
                return
            outlier_range = float(self.outlier_detection_text_field.text())
        bin_number = ''
        bin_size = ''
        if self.discretization_combo_box.currentText() == 'Number of Bins':
            if self.bin_number_text_field.text() == '':
                CustomWarningBox("Enter Number of Bins").response()
                return
            bin_number = int(self.bin_number_text_field.text())

        if self.discretization_combo_box.currentText() == 'Bin Size':
            if self.bin_size_text_field.text() == '':
                CustomWarningBox("Enter Bin Size").response()
                return
            bin_size = float(self.bin_size_text_field.text())
        structure_set = None
        if input_data_type == 'DICOM':
            structure_set = dicom_structures
        elif input_data_type == 'NIFTI':
            structure_set = nifti_structures

        slice_weighting = None
        slice_median = None
        if (self.aggr_dim_and_method_combo_box.currentText().split(',')[0] == '2D'
                and self.weighting_combo_box.currentText() == 'Slice Averaging:'):
            CustomWarningBox("Select Slice Averaging:!").response()
            return
        elif (self.aggr_dim_and_method_combo_box.currentText().split(',')[0] == '2D'
                and self.weighting_combo_box.currentText() == 'Mean'):
            slice_weighting = False
            slice_median = False
        elif (self.aggr_dim_and_method_combo_box.currentText().split(',')[0] == '2D'
                and self.weighting_combo_box.currentText() == 'Weighted Mean'):
            slice_weighting = True
            slice_median = False

        elif (self.aggr_dim_and_method_combo_box.currentText().split(',')[0] == '2D'
                and self.weighting_combo_box.currentText() == 'Median'):
            slice_weighting = False
            slice_median = True

        if structure_set == ['']:
            CustomWarningBox("Enter Structures").response()
            return

        aggr_dim, aggr_method = self.aggr_dim_and_method_combo_box.currentText().split(',')

        if aggr_method.strip() == 'merged':
            aggr_method = 'MERG'
        elif aggr_method.strip() == 'averaged':
            aggr_method = 'AVER'
        elif aggr_method.strip() == 'slice-merged':
            aggr_method = 'SLICE_MERG'
        elif aggr_method.strip() == 'direction-merged':
            aggr_method = 'DIR_MERG'

        rad_instance = Radiomics(load_dir, save_dir,
                                 input_data_type, input_imaging_mod,
                                 intensity_range, outlier_range,
                                 bin_number, bin_size, aggr_dim,
                                 aggr_method, slice_weighting, slice_median,
                                 start_folder, stop_folder, list_of_patient_folders,
                                 structure_set, nifti_image,
                                 number_of_threads)

        rad_instance.extract_radiomics()

    def open_directory(self, key):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if directory and key:
            self.load_dir_label.setText(directory)
        elif directory and not key:
            self.save_dir_label.setText(directory)

    def save_input_data(self):
        data = {
            'Data location': self.load_dir_label.text(),
            'Start folder': self.start_folder_text_field.text(),
            'Stop folder': self.stop_folder_text_field.text(),
            'List of patients': self.list_of_patient_folders_text_field.text(),
            'Data Type': self.input_data_type_combo_box.currentText(),
            'Save directory': self.save_dir_label.text(),
            '# of cores': self.number_of_threads_combo_box.currentText(),
            'DICOM mask': self.dicom_structures_text_field.text(),
            'Nifti mask': self.nifti_structure_text_field.text(),
            'Nifti image': self.nifti_image_text_field.text(),
            'Intensity range': self.intensity_range_text_field.text(),
            'Agr. strategy': self.aggr_dim_and_method_combo_box.currentText(),
            'Binning': self.discretization_combo_box.currentText(),
            'bin no.': self.bin_number_text_field.text(),
            'bin size': self.bin_size_text_field.text(),
            'int range: y/n': self.intensity_range_check_box.checkState(),
            'outlier detection': self.outlier_detection_check_box.checkState(),
            'weighting': self.weighting_combo_box.currentText(),
            'input im modality': self.input_imaging_mod_combo_box.currentText()
        }
        file_path = resource_path('zrad/input/last_saved_rad_user_input.json')
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        file_path = resource_path('zrad/input/last_saved_rad_user_input.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('Data location', ''))
                self.start_folder_text_field.setText(data.get('Start folder', ''))
                self.stop_folder_text_field.setText(data.get('Stop folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('List of patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('Data Type', ''))
                self.save_dir_label.setText(data.get('Save directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('# of cores', ''))
                self.nifti_structure_text_field.setText(data.get('Nifti mask', ''))
                self.dicom_structures_text_field.setText(data.get('DICOM mask', ''))
                self.nifti_image_text_field.setText(data.get('Nifti image', ''))
                self.intensity_range_text_field.setText(data.get('Intensity range', ''))
                self.aggr_dim_and_method_combo_box.setCurrentText(data.get('Agr. strategy', ''))
                self.discretization_combo_box.setCurrentText(data.get('Binning', ''))
                self.bin_number_text_field.setText(data.get('bin no.', ''))
                self.bin_size_text_field.setText(data.get('bin size', ''))
                self.intensity_range_check_box.setCheckState(data.get('int range: y/n', ''))
                self.outlier_detection_check_box.setCheckState(data.get('outlier detection', ''))
                self.weighting_combo_box.setCurrentText(data.get('weighting', ''))
                self.input_imaging_mod_combo_box.setCurrentText(data.get('input im modality', ''))
        except FileNotFoundError:
            print("No previous data found!")

    def init_tab(self):
        # Create a QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Path to load the files
        self.load_dir_button = CustomButton(
            'Load Directory',
            14, 30, 50, 200, 50, self,
            style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;"
        )
        self.load_dir_label = CustomLabel(
            '',
            14, 300, 50, 1400, 50, self)
        self.load_dir_label.setAlignment(Qt.AlignCenter)

        self.load_dir_button.clicked.connect(lambda: self.open_directory(key=True))

        # Set used data type
        self.input_data_type_combo_box = CustomBox(
            14, 60, 300, 140, 50, self,
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

        self.input_imaging_mod_combo_box = CustomBox(
            14, 320, 140, 170, 50, self,
            item_list=[
                "Imaging Mod.:", "CT", "MR", "PT"
            ]
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

        # Set # of used cores
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

        # Set save directory
        self.save_dir_button = CustomButton(
            'Save Directory',
            14, 30, 220, 200, 50, self,
            style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;"
        )
        self.save_dir_label = CustomLabel('', 14, 300, 220, 1400, 50, self)
        self.save_dir_label.setAlignment(Qt.AlignCenter)
        self.save_dir_button.clicked.connect(lambda: self.open_directory(key=False))

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
        self.nifti_structure_text_field = CustomTextField(
            "E.g. CTV, liver...",
            14, 540, 300, 250, 50, self
        )
        self.nifti_structures_label.hide()
        self.nifti_structure_text_field.hide()

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

        self.outlier_detection_check_box = CustomCheckBox(
            'Outlier Detection',
            18, 375, 460, 250, 50, self
        )

        self.outlier_detection_label = CustomLabel(
            'Confidence Interval (in \u03C3):',
            18, 640, 460, 350, 50, self,
            style="color: white;"
        )
        self.outlier_detection_text_field = CustomTextField(
            "E.g. 3",
            14, 930, 460, 100, 50, self
        )
        self.outlier_detection_label.hide()
        self.outlier_detection_text_field.hide()
        self.outlier_detection_check_box.stateChanged.connect(
            lambda: (self.outlier_detection_label.show(), self.outlier_detection_text_field.show())
            if self.outlier_detection_check_box.isChecked()
            else (self.outlier_detection_label.hide(), self.outlier_detection_text_field.hide()))

        self.intensity_range_label = CustomLabel(
            'Intensity range:',
            18, 635, 375, 200, 50, self,
            style="color: white;"
        )
        self.intensity_range_text_field = CustomTextField(
            "E.g. 50, inf",
            14, 820, 375, 210, 50, self
        )

        self.intensity_range_label.hide()
        self.intensity_range_text_field.hide()

        self.intensity_range_check_box = CustomCheckBox(
            'Intensity Range',
            18, 375, 380, 200, 50, self)
        self.intensity_range_check_box.stateChanged.connect(
            lambda: (self.intensity_range_label.show(), self.intensity_range_text_field.show())
            if self.intensity_range_check_box.isChecked()
            else (self.intensity_range_label.hide(), self.intensity_range_text_field.hide())
        )

        self.discretization_combo_box = CustomBox(
            14, 375, 540, 170, 50, self,
            item_list=[
                "Discretization:", "Number of Bins", "Bin Size"
                      ]
        )
        self.bin_number_text_field = CustomTextField(
            "E.g. 5",
            14, 555, 540, 100, 50, self
        )
        self.bin_size_text_field = CustomTextField("E.g. 50", 14, 555, 540, 100, 50, self)
        self.bin_number_text_field.hide()
        self.bin_size_text_field.hide()
        self.discretization_combo_box.currentTextChanged.connect(self.changed_discretization)

        self.aggr_dim_and_method_combo_box = CustomBox(
            14, 1100, 375, 300, 50, self,
            item_list=[
                "Texture Features Aggr. Method:",
                "2D, averaged",
                "2D, slice-merged",
                "2.5D, direction-merged",
                "2.5D, merged",
                "3D, averaged",
                "3D, merged"
            ]
        )

        self.weighting_combo_box = CustomBox(
            14, 1450, 375, 175, 50, self,
            item_list=[
                "Slice Averaging:", "Mean", "Weighted Mean", "Median"]
        )

        self.weighting_combo_box.hide()
        self.aggr_dim_and_method_combo_box.currentTextChanged.connect(self.changed_aggr_dim)

        self.run_button = CustomButton('Run', 20, 910, 590, 80, 50, self, style=False)
        self.run_button.clicked.connect(self.run_selected_option)

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the combo box's value is changed
        if text == 'DICOM':
            self.nifti_structures_label.hide()
            self.nifti_structure_text_field.hide()
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
        elif text == 'NIFTI':
            self.nifti_structures_label.show()
            self.nifti_structure_text_field.show()
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_image_label.show()
            self.nifti_image_text_field.show()

        else:
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_structures_label.hide()
            self.nifti_structure_text_field.hide()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()

    def changed_discretization(self, text):
        if text == 'Number of Bins':
            self.bin_number_text_field.show()
            self.bin_size_text_field.hide()
        elif text == 'Bin Size':
            self.bin_number_text_field.hide()
            self.bin_size_text_field.show()
        else:
            self.bin_number_text_field.hide()
            self.bin_size_text_field.hide()

    def changed_aggr_dim(self, text):
        if text.split(',')[0] == '2D':
            self.weighting_combo_box.show()
        else:
            self.weighting_combo_box.hide()
