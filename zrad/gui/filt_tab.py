import json
import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomWarningBox, tab_input
from ..logic.filtering import Filtering, Mean, LoG, Wavelets2D, Wavelets3D, Laws


class FilteringTab(QWidget):

    def __init__(self):
        super().__init__()

        self.wavelet_filter_response_map_connected = False

        self.setMinimumSize(1750, 650)
        self.layout = QVBoxLayout(self)

        tab_input(self)

        # Set used data type
        self.input_data_type_combo_box = CustomBox(
            60, 300, 140, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIfTI"
            ]
        )

        self.input_data_type_combo_box.currentTextChanged.connect(
            lambda:
            (self.nifti_image_label.show(),
             self.nifti_image_text_field.show())
            if self.input_data_type_combo_box.currentText() == "NIfTI"
            else
            (self.nifti_image_label.hide(),
             self.nifti_image_text_field.hide())
        )

        self.nifti_image_label = CustomLabel(
            'NIfTI image file:',
            320, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT",
            510, 300, 350, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()

        # Set output_imaging_type
        self.filter_combo_box = CustomBox(
            60, 380, 140, 50, self,
            item_list=[
                "Filter Type:", "Mean", "Laplacian of Gaussian", "Laws Kernels", "Wavelets"
            ]
        )
        self.filter_combo_box.currentTextChanged.connect(self.filter_combo_box_changed)

        self.padding_type_combo_box = CustomBox(
            480, 380, 150, 50, self,
            item_list=[
                "Padding Type:", "constant", "nearest", "wrap", "reflect"
            ]
        )
        self.padding_type_combo_box.hide()

        self.mean_filter_support_label = CustomLabel(
            'Support:',
            640, 380, 100, 50, self,
            style="color: white;"
        )
        self.mean_filter_support_text_field = CustomTextField(
            "E.g. 15",
            740, 380, 75, 50, self
        )
        self.mean_filter_support_text_field.hide()
        self.mean_filter_support_label.hide()

        self.filter_dimension_combo_box = CustomBox(
            320, 380, 140, 50, self,
            item_list=[
                "Dimension:", "2D", "3D"
            ]
        )
        self.filter_dimension_combo_box.hide()

        self.log_filter_sigma_label = CustomLabel(
            '\u03C3 (in mm):',
            640, 380, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_sigma_text_field = CustomTextField(
            "E.g. 3",
            760, 380, 75, 50, self
        )
        self.log_filter_sigma_label.hide()
        self.log_filter_sigma_text_field.hide()

        self.log_filter_cutoff_label = CustomLabel(
            'Cutoff (in \u03C3):',
            845, 380, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_cutoff_text_field = CustomTextField(
            "E.g. 4",
            990, 380, 75, 50, self)
        self.log_filter_cutoff_label.hide()
        self.log_filter_cutoff_text_field.hide()

        self.laws_filter_response_map_label = CustomLabel(
            'Response Map:',
            650, 380, 200, 50, self,
            style="color: white;"
        )
        self.laws_filter_response_map_text_field = CustomTextField("E.g. L5E5", 830, 380, 100, 50, self)
        self.laws_filter_rot_inv_combo_box = CustomBox(
            950, 380, 190, 50, self,
            item_list=[
                'Pseudo-rot. inv:', 'Enable', 'Disable'
            ]
        )
        self.laws_filter_distance_label = CustomLabel('Distance:', 1160, 380, 200, 50, self, style="color: white;")
        self.laws_filter_distance_text_field = CustomTextField(
            "E.g. 5",
            1270, 380, 75, 50, self
        )
        self.laws_filter_pooling_combo_box = CustomBox(
            1370, 380, 120, 50, self,
            item_list=['Pooling:', 'max', 'min', 'average'])
        self.laws_filter_energy_map_combo_box = CustomBox(
            1515, 380, 140, 50, self,
            item_list=[
                'Energy map:', 'Enable', 'Disable'
            ]
        )
        self.laws_filter_response_map_label.hide()
        self.laws_filter_response_map_text_field.hide()
        self.laws_filter_rot_inv_combo_box.hide()
        self.laws_filter_distance_label.hide()
        self.laws_filter_distance_text_field.hide()
        self.laws_filter_pooling_combo_box.hide()
        self.laws_filter_energy_map_combo_box.hide()

        self.wavelet_filter_type_combo_box = CustomBox(
            820, 380, 170, 50, self,
            item_list=[
                "Wavelet type:", "db3", "db2", "coif1", "haar"
            ]
        )
        self.wavelet_filter_type_combo_box.hide()
        self.wavelet_filter_response_map_combo_box = CustomBox(
            645, 380, 155, 50, self,
            item_list=['Response Map']
        )
        self.wavelet_filter_response_map_combo_box.hide()
        self.wavelet_filter_response_map_2d_combo_box = CustomBox(
            645, 380, 150, 50, self,
            item_list=[
                'Response Map:', 'LL', 'HL', 'LH', 'HH'
            ]
        )
        self.wavelet_filter_response_map_2d_combo_box.hide()
        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            645, 380, 150, 50, self,
            item_list=['Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
                       ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            645, 380, 150, 50, self,
            item_list=[
                'Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
            ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_decomposition_level_combo_box = CustomBox(
            1010, 380, 200, 50, self,
            item_list=[
                'Decomposition Lvl.:', '1', '2'
            ]
        )
        self.wavelet_filter_decomposition_level_combo_box.hide()

        self.wavelet_filter_rot_inv_combo_box = CustomBox(
            1230, 380, 200, 50, self,
            item_list=[
                'Pseudo-rot. inv:', 'Enable', 'Disable'
            ]
        )
        self.wavelet_filter_rot_inv_combo_box.hide()

        self.run_button = CustomButton(
            'Run',
            910, 590, 80, 50, self,
            style=False,
        )
        self.run_button.clicked.connect(self.run_selected_option)

    def run_selected_option(self):
        selections_text = [
            ('', self.load_dir_label.text().strip(), "Select Load Directory!"),
            ('', self.save_dir_label.text().strip(), "Select Save Directory")
        ]

        for message, text, warning in selections_text:
            if text == message and CustomWarningBox(warning).response():
                return

        selections_combo_box = [
            ('No. of Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Filter Type:', self.filter_combo_box),
            ('Dimension:', self.filter_dimension_combo_box),
            ('Padding Type:', self.padding_type_combo_box)
        ]

        for message, combo_box in selections_combo_box:
            if (combo_box.currentText() == message
                    and CustomWarningBox(f"Select {message.split(':')[0]}").response()):
                return

        if (self.input_imaging_mod_combo_box.currentText() == 'Imaging Modality:'
                and CustomWarningBox("Select Input Imaging Modality").response()):
            return

        input_imaging_mod = self.input_imaging_mod_combo_box.currentText()

        # Collect values from GUI elements
        input_dir = self.load_dir_label.text()
        output_dir = self.save_dir_label.text()
        number_of_threads = int(self.number_of_threads_combo_box.currentText().split(" ")[0])
        input_data_type = self.input_data_type_combo_box.currentText()
        filter_type = self.filter_combo_box.currentText()
        filter_dimension = self.filter_dimension_combo_box.currentText()
        filter_padding_type = self.padding_type_combo_box.currentText()

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

        if (not self.nifti_image_text_field.text().strip()
                and self.input_data_type_combo_box.currentText() == 'NIfTI'):
            CustomWarningBox("Enter NIfTI image").response()
            return
        nifti_image = self.nifti_image_text_field.text()

        if filter_type == 'Mean' and not self.mean_filter_support_text_field.text().strip():
            CustomWarningBox("Enter Support!").response()
            return
        if filter_type == 'Mean':
            mean_filter_support = int(self.mean_filter_support_text_field.text().strip())

        if filter_type == 'Laplacian of Gaussian':
            if not self.log_filter_sigma_text_field.text().strip():
                CustomWarningBox("Enter Sigma").response()
                return
            if not self.log_filter_cutoff_text_field.text().strip():
                CustomWarningBox("Enter Cutoff").response()
                return
        log_filter_sigma = self.log_filter_sigma_text_field.text()
        log_filter_cutoff = self.log_filter_cutoff_text_field.text()

        if filter_type == 'Laws Kernels':
            if not self.laws_filter_response_map_text_field.text().strip():
                CustomWarningBox("Enter Response Map").response()
                return
            if self.laws_filter_rot_inv_combo_box.currentText() == 'Pseudo-rot. inv:':
                CustomWarningBox("Select Pseudo-rotational invariance").response()
                return
            if not self.laws_filter_distance_text_field.text().strip():
                CustomWarningBox("Enter Distance").response()
                return
            if self.laws_filter_pooling_combo_box.currentText() == 'Pooling:':
                CustomWarningBox("Select Pooling").response()
                return
            if self.laws_filter_energy_map_combo_box.currentText() == 'Energy map:':
                CustomWarningBox("Select Energy map").response()
                return
        laws_filter_response_map = self.laws_filter_response_map_text_field.text()
        laws_filter_rot_inv = self.laws_filter_rot_inv_combo_box.currentText()
        laws_filter_distance = self.laws_filter_distance_text_field.text()
        laws_filter_pooling = self.laws_filter_pooling_combo_box.currentText()
        if self.laws_filter_energy_map_combo_box.currentText() == 'Enable':
            laws_filter_energy_map = True
        else:
            laws_filter_energy_map = False

        if filter_type == 'Wavelets':
            if ((self.wavelet_filter_response_map_2d_combo_box.currentText() == 'Response Map:'
                 and self.filter_dimension_combo_box.currentText() == '2D')
                    or (self.wavelet_filter_response_map_3d_combo_box.currentText() == 'Response Map:'
                        and self.filter_dimension_combo_box.currentText() == '3D')
                    or (self.wavelet_filter_response_map_combo_box.currentText() == 'Response Map:')):
                CustomWarningBox("Select Response Map").response()
                return
            if self.wavelet_filter_type_combo_box.currentText() == 'Wavelet type:':
                CustomWarningBox("Select Wavelet Type").response()
                return
            if self.wavelet_filter_decomposition_level_combo_box.currentText() == 'Decomposition Lvl.:':
                CustomWarningBox("Select Wavelet Decomposition Level").response()
                return
            if self.wavelet_filter_rot_inv_combo_box.currentText() == 'Pseudo-rot. inv:':
                CustomWarningBox("Select Pseudo-rot. inv").response()
                return
        wavelet_filter_response_map = None
        if filter_dimension == '2D' and filter_type == 'Wavelets':
            wavelet_filter_response_map = self.wavelet_filter_response_map_2d_combo_box.currentText()
        elif filter_dimension == '3D' and filter_type == 'Wavelets':
            wavelet_filter_response_map = self.wavelet_filter_response_map_3d_combo_box.currentText()
        wavelet_filter_type = self.wavelet_filter_type_combo_box.currentText()
        wavelet_filter_decomposition_lvl = None
        if filter_type == 'Wavelets':
            wavelet_filter_decomposition_lvl = int(self.wavelet_filter_decomposition_level_combo_box.currentText())
        wavelet_filter_pseudo_rot_inv = None
        if filter_type == 'Wavelets' and self.wavelet_filter_rot_inv_combo_box.currentText() == 'Enable':
            wavelet_filter_pseudo_rot_inv = True
        elif filter_type == 'Wavelets' and self.wavelet_filter_rot_inv_combo_box.currentText() == 'Disable':
            wavelet_filter_pseudo_rot_inv = False

        if filter_type == 'Mean':
            my_filter = Mean(padding_type=filter_padding_type,
                             support=int(mean_filter_support),
                             dimensionality=filter_dimension
                             )
        elif filter_type == 'Laplacian of Gaussian':
            my_filter = LoG(padding_type=filter_padding_type,
                            sigma_mm=float(log_filter_sigma),
                            cutoff=float(log_filter_cutoff),
                            dimensionality=filter_dimension
                            )
        elif filter_type == 'Laws Kernels':
            my_filter = Laws(response_map=laws_filter_response_map,
                             padding_type=filter_padding_type,
                             dimensionality=filter_dimension,
                             rotation_invariance=laws_filter_rot_inv,
                             pooling=laws_filter_pooling,
                             energy_map=laws_filter_energy_map,
                             distance=int(laws_filter_distance)
                             )
        else:
            if filter_dimension == '2D':
                my_filter = Wavelets2D(wavelet_type=wavelet_filter_type,
                                       padding_type=filter_padding_type,
                                       response_map=wavelet_filter_response_map,
                                       decomposition_level=wavelet_filter_decomposition_lvl,
                                       rotation_invariance=wavelet_filter_pseudo_rot_inv
                                       )
            else:
                my_filter = Wavelets3D(wavelet_type=wavelet_filter_type,
                                       padding_type=filter_padding_type,
                                       response_map=wavelet_filter_response_map,
                                       decomposition_level=wavelet_filter_decomposition_lvl,
                                       rotation_invariance=wavelet_filter_pseudo_rot_inv
                                       )

        filt_instance = Filtering(

            input_dir=input_dir,
            output_dir=output_dir,
            input_data_type=input_data_type,
            input_imaging_modality=input_imaging_mod,
            filter_type=my_filter,
            start_folder=start_folder,
            stop_folder=stop_folder,
            list_of_patient_folders=list_of_patient_folders,
            nifti_image=nifti_image,
            number_of_threads=number_of_threads)

        filt_instance.filtering()

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
        Update specific fields in the config.json file related to filtering settings
        without overwriting existing data.
        """
        data = {
            'filt_load_dir_label': self.load_dir_label.text(),
            'filt_start_folder': self.start_folder_text_field.text(),
            'filt_stop_folder': self.stop_folder_text_field.text(),
            'filt_list_of_patients': self.list_of_patient_folders_text_field.text(),
            'filt_input_image_modality': self.input_imaging_mod_combo_box.currentText(),
            'filt_input_data_type': self.input_data_type_combo_box.currentText(),
            'filt_save_dir_label': self.save_dir_label.text(),
            'filt_no_of_threads': self.number_of_threads_combo_box.currentText(),
            'filt_NIfTI_image': self.nifti_image_text_field.text(),
            'filt_filter_type': self.filter_combo_box.currentText(),
            'filt_filter_dimension': self.filter_dimension_combo_box.currentText(),
            'filt_padding_type': self.padding_type_combo_box.currentText(),
            'filt_Mean_support': self.mean_filter_support_text_field.text(),
            'filt_LoG_sigma': self.log_filter_sigma_text_field.text(),
            'filt_LoG_cutoff': self.log_filter_cutoff_text_field.text(),
            'filt_Laws_response_map': self.laws_filter_response_map_text_field.text(),
            'filt_Laws_rot_inv': self.laws_filter_rot_inv_combo_box.currentText(),
            'filt_Laws_distance': self.laws_filter_distance_text_field.text(),
            'filt_Laws_pooling': self.laws_filter_pooling_combo_box.currentText(),
            'filt_Laws_energy_map': self.laws_filter_energy_map_combo_box.currentText(),
            'filt_Wavelet_resp_map_2D': self.wavelet_filter_response_map_2d_combo_box.currentText(),
            'filt_Wavelet_resp_map_3D': self.wavelet_filter_response_map_3d_combo_box.currentText(),
            'filt_Wavelet_type': self.wavelet_filter_type_combo_box.currentText(),
            'filt_Wavelet_decomp_lvl': self.wavelet_filter_decomposition_level_combo_box.currentText(),
            'filt_Wavelet_rot_inv': self.wavelet_filter_rot_inv_combo_box.currentText()
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
                self.load_dir_label.setText(data.get('filt_load_dir_label', ''))
                self.start_folder_text_field.setText(data.get('filt_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('filt_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('filt_list_of_patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('filt_input_data_type', 'Data Type:'))
                self.save_dir_label.setText(data.get('filt_save_dir_label', ''))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('filt_input_image_modality', 'Imaging Modality:'))
                self.number_of_threads_combo_box.setCurrentText(data.get('filt_no_of_threads', 'No. of Threads:'))
                self.nifti_image_text_field.setText(data.get('filt_NIfTI_image', ''))
                self.filter_combo_box.setCurrentText(data.get('filt_filter_type', 'Filter Type:'))
                self.filter_dimension_combo_box.setCurrentText(data.get('filt_filter_dimension', 'Dimension:'))
                self.padding_type_combo_box.setCurrentText(data.get('filt_padding_type', 'Padding Type:'))
                self.mean_filter_support_text_field.setText(data.get('filt_Mean_support', ''))
                self.log_filter_sigma_text_field.setText(data.get('filt_LoG_sigma', ''))
                self.log_filter_cutoff_text_field.setText(data.get('filt_LoG_cutoff', ''))
                self.laws_filter_response_map_text_field.setText(data.get('filt_Laws_response_map', ''))
                self.laws_filter_rot_inv_combo_box.setCurrentText(data.get('filt_Laws_rot_inv', 'Pseudo-rot. inv:'))
                self.laws_filter_distance_text_field.setText(data.get('filt_Laws_distance', ''))
                self.laws_filter_pooling_combo_box.setCurrentText(data.get('filt_Laws_pooling', 'Pooling:'))
                self.laws_filter_energy_map_combo_box.setCurrentText(data.get('filt_Laws_energy_map', 'Energy map:'))
                self.wavelet_filter_response_map_2d_combo_box.setCurrentText(
                    data.get('filt_Wavelet_resp_map_2D', 'Response Map:'))
                self.wavelet_filter_response_map_3d_combo_box.setCurrentText(
                    data.get('filt_Wavelet_resp_map_3D', 'Response Map:'))
                self.wavelet_filter_type_combo_box.setCurrentText(data.get('filt_Wavelet_type', 'Wavelet type:'))
                self.wavelet_filter_decomposition_level_combo_box.setCurrentText(
                    data.get('filt_Wavelet_decomp_lvl', 'Decomposition Lvl.:'))
                self.wavelet_filter_rot_inv_combo_box.setCurrentText(
                    data.get('filt_Wavelet_rot_inv', 'Pseudo-rot. inv:'))

        except FileNotFoundError:
            print("No previous data found!")

    def filter_combo_box_changed(self, text):
        if text == 'Mean':
            if self.wavelet_filter_response_map_connected:
                self.filter_dimension_combo_box.currentTextChanged.disconnect()
                self.wavelet_filter_response_map_connected = False
            self.padding_type_combo_box.show()
            self.mean_filter_support_text_field.show()
            self.mean_filter_support_label.show()
            self.filter_dimension_combo_box.show()
            self.log_filter_sigma_label.hide()
            self.log_filter_sigma_text_field.hide()
            self.log_filter_cutoff_label.hide()
            self.log_filter_cutoff_text_field.hide()
            self.wavelet_filter_type_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.wavelet_filter_decomposition_level_combo_box.hide()
            self.wavelet_filter_rot_inv_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.laws_filter_response_map_label.hide()
            self.laws_filter_response_map_text_field.hide()
            self.laws_filter_rot_inv_combo_box.hide()
            self.laws_filter_distance_label.hide()
            self.laws_filter_distance_text_field.hide()
            self.laws_filter_pooling_combo_box.hide()
            self.laws_filter_energy_map_combo_box.hide()

        elif text == 'Laplacian of Gaussian':
            if self.wavelet_filter_response_map_connected:
                self.filter_dimension_combo_box.currentTextChanged.disconnect()
                self.wavelet_filter_response_map_connected = False
            self.padding_type_combo_box.show()
            self.mean_filter_support_text_field.hide()
            self.mean_filter_support_label.hide()
            self.filter_dimension_combo_box.show()
            self.log_filter_sigma_label.show()
            self.log_filter_sigma_text_field.show()
            self.log_filter_cutoff_label.show()
            self.log_filter_cutoff_text_field.show()
            self.wavelet_filter_type_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.wavelet_filter_decomposition_level_combo_box.hide()
            self.wavelet_filter_rot_inv_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.laws_filter_response_map_label.hide()
            self.laws_filter_response_map_text_field.hide()
            self.laws_filter_rot_inv_combo_box.hide()
            self.laws_filter_distance_label.hide()
            self.laws_filter_distance_text_field.hide()
            self.laws_filter_pooling_combo_box.hide()
            self.laws_filter_energy_map_combo_box.hide()

        elif text == "Laws Kernels":
            if self.wavelet_filter_response_map_connected:
                self.filter_dimension_combo_box.currentTextChanged.disconnect()
                self.wavelet_filter_response_map_connected = False
            self.padding_type_combo_box.show()
            self.mean_filter_support_text_field.hide()
            self.mean_filter_support_label.hide()
            self.filter_dimension_combo_box.show()
            self.log_filter_sigma_label.hide()
            self.log_filter_sigma_text_field.hide()
            self.log_filter_cutoff_label.hide()
            self.log_filter_cutoff_text_field.hide()
            self.wavelet_filter_type_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.wavelet_filter_decomposition_level_combo_box.hide()
            self.wavelet_filter_rot_inv_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.laws_filter_response_map_label.show()
            self.laws_filter_response_map_text_field.show()
            self.laws_filter_rot_inv_combo_box.show()
            self.laws_filter_distance_label.show()
            self.laws_filter_distance_text_field.show()
            self.laws_filter_pooling_combo_box.show()
            self.laws_filter_energy_map_combo_box.show()

        elif text == "Wavelets":
            self.padding_type_combo_box.show()
            self.mean_filter_support_text_field.hide()
            self.mean_filter_support_label.hide()
            self.filter_dimension_combo_box.show()
            self.log_filter_sigma_label.hide()
            self.log_filter_sigma_text_field.hide()
            self.log_filter_cutoff_label.hide()
            self.log_filter_cutoff_text_field.hide()
            self.wavelet_filter_type_combo_box.show()
            if self.filter_dimension_combo_box.currentText() == 'Dimension:':
                self.wavelet_filter_response_map_combo_box.show()
                self.wavelet_filter_response_map_3d_combo_box.hide()
                self.wavelet_filter_response_map_2d_combo_box.hide()
            elif self.filter_dimension_combo_box.currentText() == '3D':
                self.wavelet_filter_response_map_combo_box.hide()
                self.wavelet_filter_response_map_3d_combo_box.show()
                self.wavelet_filter_response_map_2d_combo_box.hide()
            elif self.filter_dimension_combo_box.currentText() == '2D':
                self.wavelet_filter_response_map_combo_box.hide()
                self.wavelet_filter_response_map_3d_combo_box.hide()
                self.wavelet_filter_response_map_2d_combo_box.show()

            self.filter_dimension_combo_box.currentTextChanged.connect(self.wavelet_response_map_gui)
            self.wavelet_filter_response_map_connected = True
            self.wavelet_filter_decomposition_level_combo_box.show()
            self.wavelet_filter_rot_inv_combo_box.show()
            self.laws_filter_response_map_label.hide()
            self.laws_filter_response_map_text_field.hide()
            self.laws_filter_rot_inv_combo_box.hide()
            self.laws_filter_distance_label.hide()
            self.laws_filter_distance_text_field.hide()
            self.laws_filter_pooling_combo_box.hide()
            self.laws_filter_energy_map_combo_box.hide()

        else:
            if self.wavelet_filter_response_map_connected:
                self.filter_dimension_combo_box.currentTextChanged.disconnect()
                self.wavelet_filter_response_map_connected = False
            self.padding_type_combo_box.hide()
            self.mean_filter_support_text_field.hide()
            self.mean_filter_support_label.hide()
            self.filter_dimension_combo_box.hide()
            self.log_filter_sigma_label.hide()
            self.log_filter_sigma_text_field.hide()
            self.log_filter_cutoff_label.hide()
            self.log_filter_cutoff_text_field.hide()
            self.wavelet_filter_type_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.wavelet_filter_decomposition_level_combo_box.hide()
            self.wavelet_filter_rot_inv_combo_box.hide()
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
            self.laws_filter_response_map_label.hide()
            self.laws_filter_response_map_text_field.hide()
            self.laws_filter_rot_inv_combo_box.hide()
            self.laws_filter_distance_label.hide()
            self.laws_filter_distance_text_field.hide()
            self.laws_filter_pooling_combo_box.hide()
            self.laws_filter_energy_map_combo_box.hide()

    def wavelet_response_map_gui(self, text):
        if text == '2D':
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.show()
            self.wavelet_filter_response_map_combo_box.hide()
        elif text == '3D':
            self.wavelet_filter_response_map_3d_combo_box.show()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.hide()
        else:
            self.wavelet_filter_response_map_3d_combo_box.hide()
            self.wavelet_filter_response_map_2d_combo_box.hide()
            self.wavelet_filter_response_map_combo_box.show()
