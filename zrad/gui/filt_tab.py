import json
import logging
import os
from datetime import datetime

from joblib import Parallel, delayed
from tqdm import tqdm

from ._base_tab import BaseTab, load_images
from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomWarningBox, CustomInfo, CustomInfoBox
from ..logic.exceptions import InvalidInputParametersError, DataStructureError
from ..logic.filtering import Filtering
from ..logic.toolbox_logic import get_logger, close_all_loggers, tqdm_joblib

logging.captureWarnings(True)


def _get_filtering(input_params):
    if input_params["filter_type"] == 'Mean':
        filtering = Filtering(
            filtering_method=input_params["filter_type"],
            padding_type=input_params["filter_padding_type"],
            support=int(input_params["filter_mean_support"]),
            dimensionality=input_params["filter_dimension"],
        )

    elif input_params["filter_type"] == 'Laplacian of Gaussian':
        filtering = Filtering(
            filtering_method=input_params["filter_type"],
            padding_type=input_params["filter_padding_type"],
            sigma_mm=float(input_params["filter_log_sigma"]),
            cutoff=float(input_params["filter_log_cutoff"]),
            dimensionality=input_params["filter_dimension"],
        )
    elif input_params["filter_type"] == 'Laws Kernels':
        filtering = Filtering(
            filtering_method=input_params["filter_type"],
            response_map=input_params["filter_laws_response_map"],
            padding_type=input_params["filter_padding_type"],
            dimensionality=input_params["filter_dimension"],
            rotation_invariance=input_params["filter_laws_rot_inv"] == 'Enable',
            pooling=input_params["filter_laws_pooling"],
            energy_map=input_params["filter_laws_energy_map"] == 'Enable',
            distance=int(input_params["filter_laws_distance"])
        )
    elif input_params["filter_type"] == 'Wavelets':
        if input_params["filter_dimension"] == '2D':
            filtering = Filtering(
                filtering_method=input_params["filter_type"],
                dimensionality=input_params["filter_dimension"],
                padding_type=input_params["filter_padding_type"],
                wavelet_type=input_params["filter_wavelet_type"],
                response_map=input_params["filter_wavelet_resp_map_2D"],
                decomposition_level=int(input_params["filter_wavelet_decomp_lvl"]),
                rotation_invariance=input_params["filter_wavelet_rot_inv"] == 'Enable'
            )
        elif input_params["filter_dimension"] == '3D':
            filtering = Filtering(
                filtering_method=input_params["filter_type"],
                dimensionality=input_params["filter_dimension"],
                padding_type=input_params["filter_padding_type"],
                wavelet_type=input_params["filter_wavelet_type"],
                response_map=input_params["filter_wavelet_resp_map_3D"],
                decomposition_level=int(input_params["filter_wavelet_decomp_lvl"]),
                rotation_invariance=input_params["filter_wavelet_rot_inv"] == 'Enable'
            )
        else:
            raise InvalidInputParametersError(f"Filter_dimension {input_params['filter_dimension']} is not supported.")
    else:
         raise InvalidInputParametersError(f"Filter_type {input_params['filter_type']} not supported.")

    return filtering


def _get_filename(input_params):
    input_params = input_params  # Local reference for easier access

    # Define filename formats for each filter type
    filter_formats = {
        'Mean': '{filter_type}_{filter_dimension}_{filter_mean_support}support_{filter_padding_type}',
        'Laplacian of Gaussian': '{filter_type}_{filter_dimension}_{filter_log_sigma}sigma_{filter_log_cutoff}cutoff_{filter_padding_type}',
        'Laws Kernels': ('{filter_type}_{filter_dimension}_{filter_laws_response_map}_'
                         '{filter_laws_rot_inv}_{filter_laws_pooling}_{filter_laws_energy_map}'
                         '{filter_laws_distance}_{filter_padding_type}')
    }

    # Inner function to format the filename for Wavelets filter type
    def format_wavelets_filename(input_params):
        """Helper method to format the filename for Wavelets filter type."""
        base_format = ('{filter_wavelet_type}_{filter_dimension}_{filter_wavelet_resp_map}_'
                       '{filter_wavelet_decomp_lvl}_{filter_wavelet_rot_inv}_{filter_padding_type}')

        if input_params["filter_dimension"] == '2D':
            return base_format.format(
                filter_wavelet_resp_map=input_params["filter_wavelet_resp_map_2D"],
                **input_params
            )
        elif input_params["filter_dimension"] == '3D':
            return base_format.format(
                filter_wavelet_resp_map=input_params["filter_wavelet_resp_map_3D"],
                **input_params
            )
        else:
            raise InvalidInputParametersError(f"Unknown filter dimension: {input_params['filter_dimension']}")

    # Check for Wavelets filter type, which has specific 2D and 3D formats
    if input_params["filter_type"] == 'Wavelets':
        filename = format_wavelets_filename(input_params)
    else:
        # Default case for other filter types using predefined formats
        filename_format = filter_formats.get(input_params["filter_type"])
        if filename_format:
            filename = filename_format.format(**input_params)
        else:
            raise InvalidInputParametersError(f"Unknown filter type: {input_params['filter_type']}")

    return f"{filename}.nii.gz"


def process_patient_folder(input_params, patient_folder):
    # Logger
    logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = get_logger(logger_date_time + '_Filtering')

    # Initialize Filtering instance
    filtering = _get_filtering(input_params)

    logger.info(f"Filtering patient's {patient_folder} image.")

    try:
        image = load_images(input_params, patient_folder)
    except DataStructureError as e:
        logger.error(e)
        logger.error(f"Patient {patient_folder} could not be loaded and is skipped.")
        return
    image_new = filtering.apply_filter(image)

    # Save new image
    filename = _get_filename(input_params)
    output_path = os.path.join(input_params["output_directory"], patient_folder, filename)
    image_new.save_as_nifti(output_path)


class FilteringTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.init_dicom_elements()
        self.init_nifti_elements()
        self.init_filtering_elements()
        self.connect_signals()
        self.wavelet_filter_response_map_connected = False

    def init_dicom_elements(self):
        pass

    def init_nifti_elements(self):
        # Image
        self.nifti_image_label = CustomLabel(
            'NIfTI Image:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT",
            300, 300, 200, 50, self
        )
        self.nifti_image_info_label = CustomInfo(
            ' i',
            'Specify NIfTI image file without file extension',
            510, 300, 14, 14, self
        )
        self._hide_nifti_elements()

    def _show_dicom_elements(self):
        pass

    def _hide_dicom_elements(self):
        pass

    def _show_nifti_elements(self):
        self.nifti_image_label.show()
        self.nifti_image_text_field.show()
        self.nifti_image_info_label.show()

    def _hide_nifti_elements(self):
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label.hide()

    def connect_signals(self):
        self.input_data_type_combo_box.currentTextChanged.connect(self.file_type_changed)
        self.run_button.clicked.connect(self.run_selection)
        self.filter_combo_box.currentTextChanged.connect(self._filter_combo_box_changed)

    def init_filtering_elements(self):
        # Set output_imaging_type
        self.filter_combo_box = CustomBox(
            20, 380, 160, 50, self,
            item_list=[
                "Filter Type:", "Mean", "Laplacian of Gaussian", "Laws Kernels", "Wavelets"
            ]
        )

        self.padding_type_combo_box = CustomBox(
            340, 380, 150, 50, self,
            item_list=[
                "Padding Type:", "constant", "nearest", "wrap", "reflect"
            ]
        )
        self.padding_type_combo_box.hide()

        self.mean_filter_support_label = CustomLabel(
            'Support:',
            200, 460, 100, 50, self,
            style="color: white;"
        )
        self.mean_filter_support_text_field = CustomTextField(
            "E.g. 15",
            275, 460, 75, 50, self
        )
        self.mean_filter_support_text_field.hide()
        self.mean_filter_support_label.hide()

        self.filter_dimension_combo_box = CustomBox(
            200, 380, 120, 50, self,
            item_list=[
                "Dimension:", "2D", "3D"
            ]
        )
        self.filter_dimension_combo_box.hide()

        self.log_filter_sigma_label = CustomLabel(
            '\u03C3 (in mm):',
            200, 460, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_sigma_text_field = CustomTextField(
            "E.g. 3",
            290, 460, 60, 50, self
        )
        self.log_filter_sigma_label.hide()
        self.log_filter_sigma_text_field.hide()

        self.log_filter_cutoff_label = CustomLabel(
            'Cutoff (in \u03C3):',
            375, 460, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_cutoff_text_field = CustomTextField(
            "E.g. 4",
            480, 460, 60, 50, self)
        self.log_filter_cutoff_label.hide()
        self.log_filter_cutoff_text_field.hide()

        self.laws_filter_response_map_label = CustomLabel(
            'Response Map:',
            200, 460, 200, 50, self,
            style="color: white;"
        )
        self.laws_filter_response_map_text_field = CustomTextField("E.g. L5E5", 325, 460, 100, 50, self)
        self.laws_filter_rot_inv_combo_box = CustomBox(
            450, 460, 160, 50, self,
            item_list=[
                'Rotation invariance:', 'Enable', 'Disable'
            ]
        )
        self.laws_filter_distance_label = CustomLabel('Distance:', 630, 460, 200, 50, self, style="color: white;")
        self.laws_filter_distance_text_field = CustomTextField(
            "E.g. 5",
            710, 460, 60, 50, self
        )
        self.laws_filter_pooling_combo_box = CustomBox(
            800, 460, 120, 50, self,
            item_list=['Pooling:', 'max', 'min', 'average'])
        self.laws_filter_energy_map_combo_box = CustomBox(
            950, 460, 140, 50, self,
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
            200, 460, 170, 50, self,
            item_list=[
                "Wavelet type:", "db3", "db2", "coif1", "haar"
            ]
        )
        self.wavelet_filter_type_combo_box.hide()
        self.wavelet_filter_response_map_combo_box = CustomBox(
            390, 460, 150, 50, self,
            item_list=['Response Map:']
        )
        self.wavelet_filter_response_map_combo_box.hide()
        self.wavelet_filter_response_map_2d_combo_box = CustomBox(
            390, 460, 150, 50, self,
            item_list=[
                'Response Map:', 'LL', 'HL', 'LH', 'HH'
            ]
        )
        self.wavelet_filter_response_map_2d_combo_box.hide()
        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            390, 460, 150, 50, self,
            item_list=['Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
                       ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            390, 460, 150, 50, self,
            item_list=[
                'Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
            ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_decomposition_level_combo_box = CustomBox(
            560, 460, 175, 50, self,
            item_list=[
                'Decomposition level:', '1', '2'
            ]
        )
        self.wavelet_filter_decomposition_level_combo_box.hide()

        self.wavelet_filter_rot_inv_combo_box = CustomBox(
            750, 460, 175, 50, self,
            item_list=[
                'Rotation invariance:', 'Enable', 'Disable'
            ]
        )
        self.wavelet_filter_rot_inv_combo_box.hide()

        self.run_button = CustomButton(
            'RUN',
            600, 590, 80, 50, self,
            style=False,
        )

    def check_input_parameters(self):
        # Validate combo box selections
        self._validate_combo_selections()
        self.check_common_input_parameters()

        self.input_params["nifti_image_name"] = self.get_text_from_text_field(self.input_params["nifti_image_name"])

        if self.input_params['filter_type'] == 'Mean' and not self.mean_filter_support_text_field.text().strip():
            error_msg = "Enter Support!"
            raise InvalidInputParametersError(error_msg)
        if self.input_params['filter_type'] == 'Laplacian of Gaussian':
            if not self.log_filter_sigma_text_field.text().strip():
                error_msg = "Enter Sigma"
                raise InvalidInputParametersError(error_msg)
            if not self.log_filter_cutoff_text_field.text().strip():
                error_msg = "Enter Cutoff"
                raise InvalidInputParametersError(error_msg)

        if self.input_params['filter_type'] == 'Laws Kernels':
            if not self.laws_filter_response_map_text_field.text().strip():
                error_msg = "Enter Response Map"
                raise InvalidInputParametersError(error_msg)
            if self.laws_filter_rot_inv_combo_box.currentText() == 'Rotation invariance:':
                error_msg = "Select Pseudo-rotational invariance"
                raise InvalidInputParametersError(error_msg)
            if not self.laws_filter_distance_text_field.text().strip():
                error_msg = "Enter Distance"
                raise InvalidInputParametersError(error_msg)
            if self.laws_filter_pooling_combo_box.currentText() == 'Pooling:':
                error_msg = "Select Pooling"
                raise InvalidInputParametersError(error_msg)
            if self.laws_filter_energy_map_combo_box.currentText() == 'Energy map:':
                error_msg = "Select Energy map"
                raise InvalidInputParametersError(error_msg)

        if self.input_params['filter_type'] == 'Wavelets':
            if self.wavelet_filter_type_combo_box.currentText() == 'Wavelet type:':
                error_msg = "Select Wavelet Type"
                raise InvalidInputParametersError(error_msg)
            if self.wavelet_filter_decomposition_level_combo_box.currentText() == 'Decomposition level:':
                error_msg = "Select Wavelet Decomposition Level"
                raise InvalidInputParametersError(error_msg)
            if self.wavelet_filter_rot_inv_combo_box.currentText() == 'Rotation invariance:':
                error_msg = "Select Pseudo-rot. inv"
                raise InvalidInputParametersError(error_msg)
            if ((self.wavelet_filter_response_map_3d_combo_box.currentText() == 'Response Map:' and self.filter_dimension_combo_box.currentText() == '3D') or
                (self.wavelet_filter_response_map_2d_combo_box.currentText() == 'Response Map:' and self.filter_dimension_combo_box.currentText() == '2D')):
                error_msg = "Select Response Map"
                raise InvalidInputParametersError(error_msg)

    def get_input_parameters(self):
        input_parameters = {
            'input_directory': self.load_dir_text_field.text(),
            'start_folder': self.start_folder_text_field.text(),
            'stop_folder': self.stop_folder_text_field.text(),
            'list_of_patient_folders': self.list_of_patient_folders_text_field.text(),
            'input_image_modality': self.input_imaging_mod_combo_box.currentText(),
            'input_data_type': self.input_data_type_combo_box.currentText(),
            'output_directory': self.save_dir_text_field.text(),
            'number_of_threads': self.number_of_threads_combo_box.currentText(),
            'nifti_image_name': self.nifti_image_text_field.text(),
            'filter_type': self.filter_combo_box.currentText(),
            'filter_dimension': self.filter_dimension_combo_box.currentText(),
            'filter_padding_type': self.padding_type_combo_box.currentText(),
            'filter_mean_support': self.mean_filter_support_text_field.text(),
            'filter_log_sigma': self.log_filter_sigma_text_field.text(),
            'filter_log_cutoff': self.log_filter_cutoff_text_field.text(),
            'filter_laws_response_map': self.laws_filter_response_map_text_field.text(),
            'filter_laws_rot_inv': self.laws_filter_rot_inv_combo_box.currentText(),
            'filter_laws_distance': self.laws_filter_distance_text_field.text(),
            'filter_laws_pooling': self.laws_filter_pooling_combo_box.currentText(),
            'filter_laws_energy_map': self.laws_filter_energy_map_combo_box.currentText(),
            'filter_wavelet_resp_map_2D': self.wavelet_filter_response_map_2d_combo_box.currentText(),
            'filter_wavelet_resp_map_3D': self.wavelet_filter_response_map_3d_combo_box.currentText(),
            'filter_wavelet_type': self.wavelet_filter_type_combo_box.currentText(),
            'filter_wavelet_decomp_lvl': self.wavelet_filter_decomposition_level_combo_box.currentText(),
            'filter_wavelet_rot_inv': self.wavelet_filter_rot_inv_combo_box.currentText()
        }
        self.input_params = input_parameters

    def run_selection(self):
        """Executes filtering based on user-selected options."""
        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time + '_Filtering')
        self.logger.info("Filtering started")

        # Prepare input parameters for radiomics extraction
        self.get_input_parameters()

        # Check input parameters
        try:
            self.check_input_parameters()
        except InvalidInputParametersError as e:
            # Stop execution if input parameters are invalid
            self.logger.error(e)
            CustomWarningBox(str(e)).response()
            return

        # Get patient folders
        list_of_patient_folders = self.get_patient_folders()

        # Process each patient folder
        if list_of_patient_folders:
            n_jobs = self.input_params["number_of_threads"]
            if n_jobs == 1:
                for patient_folder in tqdm(list_of_patient_folders, desc="Patient directories"):
                    process_patient_folder(self.input_params, patient_folder)
            else:
                with tqdm_joblib(tqdm(desc="Patient directories", total=len(list_of_patient_folders))):
                    Parallel(n_jobs=n_jobs)(delayed(process_patient_folder)(self.input_params, patient_folder) for patient_folder in list_of_patient_folders)
        else:
            CustomWarningBox("No patients to filter.")

        self.logger.info("Filtering finished!")
        CustomInfoBox("Filtering finished!").response()

    def save_settings(self):
        """
        Update specific fields in the config.json file related to filtering settings
        without overwriting existing data.
        """
        self.get_input_parameters()
        data = {'filtering_' + key: value for key, value in self.input_params.items()}
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

    def load_settings(self):
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_text_field.setText(data.get('filtering_input_directory', ''))
                self.start_folder_text_field.setText(data.get('filtering_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('filtering_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('filtering_list_of_patient_folders', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('filtering_input_data_type', 'Data Type:'))
                self.save_dir_text_field.setText(data.get('filtering_output_directory', ''))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('filtering_input_image_modality', 'Imaging Modality:'))
                self.number_of_threads_combo_box.setCurrentText(data.get('filtering_number_of_threads', 'Threads:'))
                self.nifti_image_text_field.setText(data.get('filtering_nifti_image_name', ''))
                self.filter_combo_box.setCurrentText(data.get('filtering_filter_type', 'Filter Type:'))
                self.filter_dimension_combo_box.setCurrentText(data.get('filtering_filter_dimension', 'Dimension:'))
                self.padding_type_combo_box.setCurrentText(data.get('filtering_filter_padding_type', 'Padding Type:'))
                self.mean_filter_support_text_field.setText(data.get('filtering_filter_mean_support', ''))
                self.log_filter_sigma_text_field.setText(data.get('filtering_filter_log_sigma', ''))
                self.log_filter_cutoff_text_field.setText(data.get('filtering_filter_log_cutoff', ''))
                self.laws_filter_response_map_text_field.setText(data.get('filtering_filter_laws_response_map', ''))
                self.laws_filter_rot_inv_combo_box.setCurrentText(data.get('filtering_filter_laws_rot_inv', 'Rotation invariance:'))
                self.laws_filter_distance_text_field.setText(data.get('filtering_filter_laws_distance', ''))
                self.laws_filter_pooling_combo_box.setCurrentText(data.get('filtering_filter_laws_pooling', 'Pooling:'))
                self.laws_filter_energy_map_combo_box.setCurrentText(data.get('filtering_filter_laws_energy_map', 'Energy map:'))
                self.wavelet_filter_response_map_2d_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_resp_map_2D', 'Response Map:'))
                self.wavelet_filter_response_map_3d_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_resp_map_3D', 'Response Map:'))
                self.wavelet_filter_type_combo_box.setCurrentText(data.get('filtering_filter_wavelet_type', 'Wavelet type:'))
                self.wavelet_filter_decomposition_level_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_decomp_lvl', 'Decomposition Lvl.:'))
                self.wavelet_filter_rot_inv_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_rot_inv', 'Rotation invariance:'))

        except FileNotFoundError:
            print("No previous data found!")

    def _filter_combo_box_changed(self, text):
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

            self.filter_dimension_combo_box.currentTextChanged.connect(self._wavelet_response_map_combo_box_changed)
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

    def _wavelet_response_map_combo_box_changed(self, text):
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

    def _validate_combo_selections(self):
        """Validate combo box selections."""
        required_selections = [
            ('Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Filter Type:', self.filter_combo_box),
            ('Dimension:', self.filter_dimension_combo_box),
            ('Padding Type:', self.padding_type_combo_box),
            ('Imaging Modality:', self.input_imaging_mod_combo_box)
        ]
        for message, combo_box in required_selections:
            if combo_box.currentText() == message:
                warning_msg = f"Select {message.split(':')[0]}"
                raise InvalidInputParametersError(warning_msg)
