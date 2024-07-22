import json
import os

# Import required PyQt5 modules for GUI creation
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomWarningBox, CustomCheckBox, \
    CustomInfo, data_io
from ..logic.preprocessing import Preprocessing


class PreprocessingTab(QWidget):

    def __init__(self):
        super().__init__()

        self.setMinimumSize(1220, 640)
        self.layout = QVBoxLayout(self)

        data_io(self)

        # Input Data Type ComboBox
        self.input_data_type_combo_box = CustomBox(
            20, 300, 160, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIfTI"
            ]
        )
        self.input_data_type_combo_box.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        self.just_save_as_nifti_check_box = CustomCheckBox(
            'Save as NIfTI without resampling',
            800, 300, 400, 50, self)
        self.just_save_as_nifti_check_box.hide()

        # DICOM and NIfTI Structures TextFields and Labels
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
        self.nifti_structures_text_field = CustomTextField(
            "E.g. CTV, liver...",
            345, 300, 220, 50, self
        )
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()
        self.nifti_structure_info_label = CustomInfo(
            ' i',
            'Provide NIfTI masks of interest, without the file extensions: '
            '\nif the files of interest are GTV.nii.gz and liver.nii, provide: GTV, liver.',
            575, 300, 14, 14, self
        )
        self.nifti_structure_info_label.hide()

        self.nifti_image_label = CustomLabel(
            'NIfTI Image File:',
            600, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT",
            745, 300, 220, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label = CustomInfo(
            ' i',
            'Specify NIfTI image file without the file extension: '
            '\ne.g. if NIfTI image file is imageCT.nii.gz, provide only imageCT.',
            975, 300, 14, 14, self
        )
        self.nifti_image_info_label.hide()

        # Resample Resolution Label and TextField
        self.resample_resolution_label = CustomLabel(
            'Resample Resolution (mm):',
            200, 380, 300, 50, self,
            style="color: white;"
        )
        self.resample_resolution_text_field = CustomTextField(
            "E.g. 1", 420, 380, 90, 50, self
        )

        # Image Interpolation Method ComboBox
        self.image_interpolation_method_combo_box = CustomBox(
            775, 380, 210, 50, self,
            item_list=[
                'Image Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Resample Dimension ComboBox
        self.resample_dimension_combo_box = CustomBox(
            1000, 380, 210, 50, self,
            item_list=[
                'Resample Dimension:', "2D", "3D"
            ]
        )

        # Mask Interpolation Method ComboBox
        self.mask_interpolation_method_combo_box = CustomBox(
            200, 460, 210, 50, self,
            item_list=[
                'Mask Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Mask Interpolation Threshold Label and TextField
        self.mask_interpolation_threshold_label = CustomLabel(
            'Mask Interpolation Threshold:',
            600, 460, 360, 50, self,
            style="color: white;"
        )
        self.mask_interpolation_threshold_text_field = CustomTextField(
            "E.g. 0.75",
            830, 460, 100, 50, self
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
            'RUN',
            600, 590, 80, 50, self,
            style=False
        )
        self.run_button.clicked.connect(self.run_selected_input)

    def run_selected_input(self):

        selections_text = [
            ('', self.load_dir_label.text().strip(), "Select Load Directory!"),
            ('', self.save_dir_label.text().strip(), "Select Save Directory"),
        ]
        for message, text, warning in selections_text:
            if text == message and CustomWarningBox(warning).response():
                return

        selections_combo_box = [
            ('No. of Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
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
            list_of_patient_folders = [pat.strip() for pat in str(self.list_of_patient_folders_text_field.text()).split(",")]
        else:
            list_of_patient_folders = None

        if (self.input_imaging_mod_combo_box.currentText() == 'Imaging Modality:'
                and CustomWarningBox("Select Input Imaging Modality").response()):
            return
        input_imaging_mod = self.input_imaging_mod_combo_box.currentText()

        number_of_threads = int(self.number_of_threads_combo_box.currentText().split(" ")[0])
        input_data_type = self.input_data_type_combo_box.currentText()
        dicom_structures = [ROI.strip() for ROI in self.dicom_structures_text_field.text().split(",")]
        nifti_structures = [ROI.strip() for ROI in self.nifti_structures_text_field.text().split(",")]

        structure_set = None
        if input_data_type == 'DICOM' and len(dicom_structures) > 0:
            structure_set = dicom_structures
        elif input_data_type == 'NIfTI' and len(nifti_structures) > 0:
            structure_set = nifti_structures

        if self.just_save_as_nifti_check_box.isChecked() and input_data_type == 'DICOM':
            just_save_as_nifti = True

            prep_instance = Preprocessing(
                input_dir=input_dir,
                output_dir=output_dir,
                input_data_type=input_data_type,
                input_imaging_modality=input_imaging_mod,
                structure_set=structure_set,
                just_save_as_nifti=just_save_as_nifti,
                start_folder=start_folder,
                stop_folder=stop_folder,
                list_of_patient_folders=list_of_patient_folders,
                number_of_threads=number_of_threads)

            prep_instance.resample()
        else:
            just_save_as_nifti = False

            # Validate text field inputs
            selections_text = [
                ('', self.mask_interpolation_threshold_text_field.text().strip(), "Enter Mask Interpolation Threshold"),
                ('', self.resample_resolution_text_field.text().strip(), "Enter Resample Resolution")
            ]

            for message, text, warning in selections_text:
                if text == message and CustomWarningBox(warning).response():
                    return

            # Validate combo box selections
            selections_combo_box = [
                ('Image Interpolation:', self.image_interpolation_method_combo_box),
                ('Resample Dimension:', self.resample_dimension_combo_box)
            ]
            for message, combo_box in selections_combo_box:
                if (combo_box.currentText() == message
                        and CustomWarningBox(f"Select {message.split(':')[0]}").response()):
                    return
            if (self.mask_interpolation_method_combo_box.currentText() == 'Mask Interpolation:'
                    and structure_set != ['']
                    and CustomWarningBox("Select Mask Interpolation").response()):
                return

            if (not self.nifti_image_text_field.text().strip()
                    and self.input_data_type_combo_box.currentText() == 'NIfTI'):
                CustomWarningBox("Enter NIfTI image").response()
                return
            nifti_image = self.nifti_image_text_field.text()

            # Collect values from GUI elements
            if structure_set == ['']:
                mask_interpolation_method = 'NN'
            else:
                mask_interpolation_method = self.mask_interpolation_method_combo_box.currentText()
            mask_interpolation_threshold = float(self.mask_interpolation_threshold_text_field.text())
            resample_resolution = float(self.resample_resolution_text_field.text())

            # Collect values from GUI elements
            image_interpolation_method = self.image_interpolation_method_combo_box.currentText()
            resample_dimension = self.resample_dimension_combo_box.currentText()

        # Create a Preprocessing instance

            prep_instance = Preprocessing(
                input_dir=input_dir,
                output_dir=output_dir,
                input_data_type=input_data_type,
                input_imaging_modality=input_imaging_mod,
                structure_set=structure_set,
                just_save_as_nifti=just_save_as_nifti,
                resample_resolution=resample_resolution,
                resample_dimension=resample_dimension,
                image_interpolation_method=image_interpolation_method,
                mask_interpolation_method=mask_interpolation_method,
                mask_interpolation_threshold=mask_interpolation_threshold,
                start_folder=start_folder,
                stop_folder=stop_folder,
                list_of_patient_folders=list_of_patient_folders,
                nifti_image=nifti_image,
                number_of_threads=number_of_threads)
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
        Update specific fields in the config.json file without overwriting existing data.
        """
        # Data to be updated
        data = {
            'prep_load_dir_label': self.load_dir_label.text(),
            'prep_start_folder': self.start_folder_text_field.text(),
            'prep_stop_folder': self.stop_folder_text_field.text(),
            'prep_list_of_patients': self.list_of_patient_folders_text_field.text(),
            'prep_input_data_type': self.input_data_type_combo_box.currentText(),
            'prep_save_dir_label': self.save_dir_label.text(),
            'prep_no_of_threads': self.number_of_threads_combo_box.currentText(),
            'prep_DICOM_structures': self.dicom_structures_text_field.text(),
            'prep_NIfTI_image': self.nifti_image_text_field.text(),
            'prep_NIfTI_structures': self.nifti_structures_text_field.text(),
            'prep_resample_resolution': self.resample_resolution_text_field.text(),
            'prep_interpolation_method': self.image_interpolation_method_combo_box.currentText(),
            'prep_resample_dim': self.resample_dimension_combo_box.currentText(),
            'prep_mask_interpolation_method': self.mask_interpolation_method_combo_box.currentText(),
            'prep_mask_interpolation_threshold': self.mask_interpolation_threshold_text_field.text(),
            'prep_input_image_modality': self.input_imaging_mod_combo_box.currentText(),
            'prep_just_save_as_nifti': self.just_save_as_nifti_check_box.checkState()
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
        """
        Load input data from a JSON file.
        """
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('prep_load_dir_label', ''))
                self.start_folder_text_field.setText(data.get('prep_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('prep_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('prep_list_of_patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('prep_input_data_type', 'Data Type:'))
                self.save_dir_label.setText(data.get('prep_save_dir_label', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('prep_no_of_threads', 'No. of Threads:'))
                self.dicom_structures_text_field.setText(data.get('prep_DICOM_structures', ''))
                self.nifti_image_text_field.setText(data.get('prep_NIfTI_image', ''))
                self.nifti_structures_text_field.setText(data.get('prep_NIfTI_structures', ''))
                self.resample_resolution_text_field.setText(data.get('prep_resample_resolution', ''))
                self.image_interpolation_method_combo_box.setCurrentText(
                    data.get('prep_interpolation_method', 'Image Interpolation:'))
                self.resample_dimension_combo_box.setCurrentText(data.get('prep_resample_dim', 'Resample Dimension:'))
                self.mask_interpolation_method_combo_box.setCurrentText(
                    data.get('prep_mask_interpolation_method', 'Mask Interpolation:'))
                self.mask_interpolation_threshold_text_field.setText(
                    data.get('prep_mask_interpolation_threshold', '0.5'))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('prep_input_image_modality', 'Imaging Modality:'))
                self.just_save_as_nifti_check_box.setCheckState(data.get('prep_just_save_as_nifti', 0))

        except FileNotFoundError:
            print("No previous data found!")

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the file type combobox value is changed
        if text == 'DICOM':
            self.just_save_as_nifti_check_box.show()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.nifti_structures_label.hide()
            self.nifti_structures_text_field.hide()
            self.dicom_structures_info_label.show()
            self.nifti_structure_info_label.hide()
            self.nifti_image_info_label.hide()
        elif text == 'NIfTI':
            self.just_save_as_nifti_check_box.hide()
            self.nifti_image_label.show()
            self.nifti_image_text_field.show()
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_structures_label.show()
            self.nifti_structures_text_field.show()
            self.dicom_structures_info_label.hide()
            self.nifti_structure_info_label.show()
            self.nifti_image_info_label.show()
        else:
            self.just_save_as_nifti_check_box.hide()
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.nifti_image_label.hide()
            self.nifti_image_text_field.hide()
            self.nifti_structures_label.hide()
            self.nifti_structures_text_field.hide()
            self.dicom_structures_info_label.hide()
            self.nifti_structure_info_label.hide()
            self.nifti_image_info_label.hide()
