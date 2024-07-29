import json
import os

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomCheckBox, CustomWarningBox, \
    CustomInfo, data_io
from ..logic.radiomics import Radiomics


class RadiomicsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(1220, 640)
        self.layout = QVBoxLayout(self)

        data_io(self)

        # Set used data type
        self.input_data_type_combo_box = CustomBox(
            20, 300, 160, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIfTI"
            ]
        )

        self.input_data_type_combo_box.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        self.dicom_structures_label = CustomLabel(
            'Structures:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.dicom_structures_text_field = CustomTextField(
            "E.g. CTV, liver... or ExtractAllMasks",
            300, 300, 450, 50, self
        )
        self.dicom_structures_label.hide()
        self.dicom_structures_text_field.hide()
        self.dicom_structures_info_label = CustomInfo(
            ' i',
            'Type ROIs of interest (e.g. CTV, liver), \nor type ExtractAllMasks to use all ROIs from RTSTRUCT.',
            760, 300, 14, 14, self
        )
        self.dicom_structures_info_label.hide()

        self.nifti_structures_label = CustomLabel(
            'NIfTI Mask Files:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_structure_text_field = CustomTextField(
            "E.g. CTV, liver...",
            345, 300, 220, 50, self
        )
        self.nifti_structures_label.hide()
        self.nifti_structure_text_field.hide()
        self.nifti_structure_info_label = CustomInfo(
            ' i',
            'Provide NIfTI masks of interest, without the file extensions: '
            '\nif the files of interest are GTV.nii.gz and liver.nii, provide: GTV, liver.',
            575, 300, 14, 14, self
        )
        self.nifti_structure_info_label.hide()

        self.nifti_image_label = CustomLabel(
            'NIfTI Image File(s):',
            600, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. filtered_imageCT, imageCT",
            760, 300, 220, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label = CustomInfo(
            ' i',
            'If radiomics extracted from the filtered NIfTI image, '
                    '\nboth filtered and original NIfTI images should be provided. '
                    '\nSpecify the filtered NIFTI image first, followed by the original NIfTI image.'
                    '\nSpecify NIfTI image file(s) without file extensions:'
                    '\nE.g. imageCT (when extracting radiomics from the imageCT.nii.gz);'
                    '\nE.g. filtered_imageCT, imageCT (when extracting radiomics from the filtered_imageCT.nii.gz).',
            990, 300, 14, 14, self
        )
        self.nifti_image_info_label.hide()

        self.outlier_detection_check_box = CustomCheckBox(
            'Outlier Removal (in \u03C3)',
            200, 460, 250, 50, self
        )

        # self.outlier_detection_label = CustomLabel(
        #     'Confidence Interval (in \u03C3):',
        #     640, 460, 350, 50, self,
        #     style="color: white;"
        # )
        self.outlier_detection_text_field = CustomTextField(
            "E.g. 3",
            410, 460, 100, 50, self
        )
        # self.outlier_detection_label.hide()
        self.outlier_detection_text_field.hide()
        self.outlier_detection_check_box.stateChanged.connect(
            lambda: (self.outlier_detection_text_field.show())
            if self.outlier_detection_check_box.isChecked()
            else (self.outlier_detection_text_field.hide()))

        # self.intensity_range_label = CustomLabel(
        #     'Intensity range:',
        #     435, 375, 200, 50, self,
        #     style="color: white;"
        # )
        self.intensity_range_text_field = CustomTextField(
            "E.g. -1000, 400",
            410, 375, 210, 50, self
        )

        # self.intensity_range_label.hide()
        self.intensity_range_text_field.hide()

        self.intensity_range_check_box = CustomCheckBox(
            'Intensity Range',
            200, 380, 200, 50, self)
        self.intensity_range_check_box.stateChanged.connect(
            lambda: (self.intensity_range_text_field.show())
            if self.intensity_range_check_box.isChecked()
            else (self.intensity_range_text_field.hide())
        )

        self.discretization_combo_box = CustomBox(
            700, 460, 170, 50, self,
            item_list=[
                "Discretization:", "Number of Bins", "Bin Size"
            ]
        )
        self.bin_number_text_field = CustomTextField(
            "E.g. 5",
            1000, 460, 100, 50, self
        )
        self.bin_size_text_field = CustomTextField("E.g. 50", 1000, 460, 100, 50, self)

        self.bin_number_text_field.hide()
        self.bin_size_text_field.hide()
        self.discretization_combo_box.currentTextChanged.connect(self.changed_discretization)

        self.aggr_dim_and_method_combo_box = CustomBox(
            700, 375, 250, 50, self,
            item_list=[
                "Texture Aggregation Method:",
                "2D, averaged",
                "2D, slice-merged",
                "2.5D, direction-merged",
                "2.5D, merged",
                "3D, averaged",
                "3D, merged"
            ]
        )

        self.weighting_combo_box = CustomBox(
            1000, 375, 175, 50, self,
            item_list=[
                "Slice Averaging:", "Mean", "Weighted Mean", "Median"]
        )
        self.weighting_combo_box_info_label = CustomInfo(
            ' i',
            "'Mean' approach is in agreement with IBSI. \n'Weighted Mean' and 'Median' "
            "are custom solutions developed at USZ.",
            1185, 375, 14, 14, self
        )
        self.weighting_combo_box_info_label.hide()

        self.weighting_combo_box.hide()
        self.aggr_dim_and_method_combo_box.currentTextChanged.connect(self.changed_aggr_dim)

        self.run_button = CustomButton('RUN',
                                       600, 590, 80, 50, self, style=False)
        self.run_button.clicked.connect(self.run_selected_option)

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
            ('Imaging Modality:', self.input_imaging_mod_combo_box)
        ]

        for message, combo_box in selections_combo_box:
            if (combo_box.currentText() == message
                    and CustomWarningBox(f"Select {message.split(':')[0]}").response()):
                return

        # Collect values from GUI elements
        input_dir = self.load_dir_label.text()
        output_dir = self.save_dir_label.text()

        start_folder = None
        if self.start_folder_text_field.text().strip() != '':
            start_folder = self.start_folder_text_field.text().strip()

        stop_folder = None
        if self.stop_folder_text_field.text().strip() != '':
            stop_folder = self.stop_folder_text_field.text().strip()

        if self.list_of_patient_folders_text_field.text() != '':
            list_of_patient_folders = [pat.strip() for pat in
                                       str(self.list_of_patient_folders_text_field.text()).split(",")]
        else:
            list_of_patient_folders = None

        number_of_threads = int(self.number_of_threads_combo_box.currentText().split(" ")[0])
        input_data_type = self.input_data_type_combo_box.currentText()
        dicom_structures = [ROI.strip() for ROI in self.dicom_structures_text_field.text().split(",")]

        if (not self.nifti_image_text_field.text().strip()
                and self.input_data_type_combo_box.currentText() == 'NIfTI'):
            CustomWarningBox("Enter NIfTI image").response()
            return
        nifti_images = [file_name.strip() for file_name in self.nifti_image_text_field.text().split(',')]

        # Collect values from GUI elements
        nifti_structures = [ROI.strip() for ROI in self.nifti_structure_text_field.text().split(",")]
        intensity_range = None
        if self.intensity_range_check_box.isChecked():
            if self.intensity_range_text_field.text() == '':
                CustomWarningBox("Enter intensity range").response()
                return
            intensity_range = [np.inf if intensity.strip() == '' else float(intensity)
                               for intensity in self.intensity_range_text_field.text().split(',')]
        outlier_range = None
        if self.outlier_detection_check_box.isChecked():
            if self.outlier_detection_text_field.text() == '':
                CustomWarningBox("Enter Confidence Interval").response()
                return
            outlier_range = float(self.outlier_detection_text_field.text())
        number_of_bins = None
        bin_size = None
        if self.discretization_combo_box.currentText() == 'Number of Bins':
            if self.bin_number_text_field.text() == '':
                CustomWarningBox("Enter Number of Bins").response()
                return
            number_of_bins = int(self.bin_number_text_field.text())

        if self.discretization_combo_box.currentText() == 'Bin Size':
            if self.bin_size_text_field.text() == '':
                CustomWarningBox("Enter Bin Size").response()
                return
            bin_size = float(self.bin_size_text_field.text())
        structure_set = None
        if input_data_type == 'DICOM':
            structure_set = dicom_structures
        elif input_data_type == 'NIfTI':
            structure_set = nifti_structures

        slice_weighting = False
        slice_median = False
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

        if (self.input_imaging_mod_combo_box.currentText() == 'Imaging Modality:'
                and CustomWarningBox("Select Input Imaging Modality").response()):
            return
        input_imaging_modality = self.input_imaging_mod_combo_box.currentText()

        rad_instance = Radiomics(
            input_dir=input_dir,
            output_dir=output_dir,
            input_data_type=input_data_type,
            input_imaging_modality=input_imaging_modality,
            structure_set=structure_set,
            aggr_dim=aggr_dim,
            aggr_method=aggr_method,
            intensity_range=intensity_range,
            outlier_range=outlier_range,
            number_of_bins=number_of_bins,
            bin_size=bin_size,
            slice_weighting=slice_weighting,
            slice_median=slice_median,
            start_folder=start_folder,
            stop_folder=stop_folder,
            list_of_patient_folders=list_of_patient_folders,
            nifti_images=nifti_images,
            number_of_threads=number_of_threads)

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
        """
        Update specific radiology-related fields in the config.json file without
        overwriting existing data.
        """
        data = {
            'rad_load_dir_label': self.load_dir_label.text(),
            'rad_start_folder': self.start_folder_text_field.text(),
            'rad_stop_folder': self.stop_folder_text_field.text(),
            'rad_list_of_patients': self.list_of_patient_folders_text_field.text(),
            'rad_input_data_type': self.input_data_type_combo_box.currentText(),
            'rad_save_dir_label': self.save_dir_label.text(),
            'rad_no_of_threads': self.number_of_threads_combo_box.currentText(),
            'rad_DICOM_structures': self.dicom_structures_text_field.text(),
            'rad_NIfTI_structures': self.nifti_structure_text_field.text(),
            'rad_NIfTI_image': self.nifti_image_text_field.text(),
            'rad_intensity_range': self.intensity_range_text_field.text(),
            'rad_agr_strategy': self.aggr_dim_and_method_combo_box.currentText(),
            'rad_binning': self.discretization_combo_box.currentText(),
            'rad_number_of_bins': self.bin_number_text_field.text(),
            'rad_bin_size': self.bin_size_text_field.text(),
            'rad_intensity_range_check_box': self.intensity_range_check_box.checkState(),
            'rad_outlier_detection_check_box': self.outlier_detection_check_box.checkState(),
            'rad_weighting': self.weighting_combo_box.currentText(),
            'rad_input_image_modality': self.input_imaging_mod_combo_box.currentText()
        }

        file_path = os.path.join(os.getcwd(), 'config.json')

        # Attempt to read the existing data from the file
        try:
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}

        # Update the existing data with the new data
        existing_data.update(data)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def load_input_data(self):
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('rad_load_dir_label', ''))
                self.start_folder_text_field.setText(data.get('rad_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('rad_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('rad_list_of_patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('rad_input_data_type', 'Data Type:'))
                self.save_dir_label.setText(data.get('rad_save_dir_label', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('rad_no_of_threads', 'No. of Threads:'))
                self.nifti_structure_text_field.setText(data.get('rad_NIfTI_structures', ''))
                self.dicom_structures_text_field.setText(data.get('rad_DICOM_structures', ''))
                self.nifti_image_text_field.setText(data.get('rad_NIfTI_image', ''))
                self.intensity_range_text_field.setText(data.get('rad_intensity_range', ''))
                self.aggr_dim_and_method_combo_box.setCurrentText(
                    data.get('rad_agr_strategy', 'Texture Features Aggr. Method:'))
                self.discretization_combo_box.setCurrentText(data.get('rad_binning', 'Discretization:'))
                self.bin_number_text_field.setText(data.get('rad_number_of_bins', ''))
                self.bin_size_text_field.setText(data.get('rad_bin_size', ''))
                self.intensity_range_check_box.setCheckState(data.get('rad_intensity_range_check_box', 0))
                self.outlier_detection_check_box.setCheckState(data.get('rad_outlier_detection_check_box', 0))
                self.weighting_combo_box.setCurrentText(data.get('rad_weighting', 'Slice Averaging:'))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('rad_input_image_modality', 'Imaging Modality:'))
        except FileNotFoundError:
            print("No previous data found!")

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the combo box's value is changed
        if text == 'DICOM':
            self.nifti_structures_label.hide()
            self.nifti_structure_text_field.hide()
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.nifti_image_info_label.hide()
            self.nifti_structure_info_label.hide()
            self.dicom_structures_info_label.show()
        elif text == 'NIfTI':
            self.nifti_structure_info_label.show()
            self.nifti_image_info_label.show()
            self.nifti_structures_label.show()
            self.nifti_structure_text_field.show()
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_image_label.show()
            self.nifti_image_text_field.show()
            self.dicom_structures_info_label.hide()

        else:
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_structures_label.hide()
            self.nifti_image_info_label.hide()
            self.nifti_structure_text_field.hide()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.nifti_structure_info_label.hide()
            self.dicom_structures_info_label.hide()

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
            self.weighting_combo_box_info_label.show()
        else:
            self.weighting_combo_box.hide()
            self.weighting_combo_box_info_label.hide()
