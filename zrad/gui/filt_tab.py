import json
import logging
import sys
from datetime import datetime

from PyQt5.QtCore import QThread

from ..batch import BatchFilter
from ..exceptions import InvalidInputParametersError
from ..toolbox_logic import close_all_loggers, get_config_path, get_logger
from ._base_tab import BaseTab
from .toolbox_gui import (
    CustomBox,
    CustomButton,
    CustomErrorBox,
    CustomInfo,
    CustomInfoBox,
    CustomLabel,
    CustomTextField,
    CustomWarningBox,
    ProcessingProgressDialog,
    ProcessingWorker,
)

logging.captureWarnings(True)

IS_FROZEN = getattr(sys, 'frozen', False)


def create_batch_filter_from_input_params(input_params, parallel_backend):
    response_map = (
        input_params["filter_wavelet_resp_map_3D"]
        if input_params["filter_dimension"] == '3D'
        else input_params["filter_wavelet_resp_map_2D"]
    )
    return BatchFilter(
        input_directory=input_params["input_directory"],
        output_directory=input_params["output_directory"],
        input_data_type=input_params["input_data_type"],
        modality=input_params["input_image_modality"],
        number_of_threads=input_params["number_of_threads"],
        patient_folders=input_params["list_of_patient_folders"],
        start_folder=input_params["start_folder"],
        stop_folder=input_params["stop_folder"],
        nifti_image_name=input_params["nifti_image_name"],
        filter_type=input_params["filter_type"],
        filter_dimension=input_params["filter_dimension"],
        padding_type=input_params["filter_padding_type"],
        mean_support=input_params["filter_mean_support"],
        log_sigma=input_params["filter_log_sigma"],
        log_cutoff=input_params["filter_log_cutoff"],
        laws_response_map=input_params["filter_laws_response_map"],
        laws_rotation_invariance=input_params["filter_laws_rot_inv"],
        laws_pooling=input_params["filter_laws_pooling"],
        laws_energy_map=input_params["filter_laws_energy_map"],
        laws_distance=input_params["filter_laws_distance"],
        wavelet_response_map=response_map,
        wavelet_type=input_params["filter_wavelet_type"],
        wavelet_decomposition_level=input_params["filter_wavelet_decomp_lvl"],
        wavelet_rotation_invariance=input_params["filter_wavelet_rot_inv"],
        gabor_res_mm=input_params["filter_gabor_res_mm"],
        gabor_sigma_mm=input_params["filter_gabor_sigma_mm"],
        gabor_lambda_mm=input_params["filter_gabor_lambda_mm"],
        gabor_gamma=input_params["filter_gabor_gamma"],
        gabor_theta=input_params["filter_gabor_theta"],
        gabor_rotation_invariance=input_params["filter_gabor_rotinv"],
        gabor_orthogonal_planes=input_params["filter_gabor_ortho"],
        riesz_order=input_params.get("filter_riesz_order"),
        structure_tensor_sigma_mm=input_params.get("filter_structure_tensor_sigma_mm"),
        parallel_backend=parallel_backend,
    )


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
        self.nifti_image_label = CustomLabel('NIfTI Image:', 200, 300, 200, 50, self, style="color: white;")
        self.nifti_image_text_field = CustomTextField("E.g. imageCT", 300, 300, 200, 50, self)
        self.nifti_image_info_label = CustomInfo(
            ' i', 'Specify NIfTI image file without file extension', 510, 300, 14, 14, self
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
        pos_y_row1 = 380
        pos_y_row2 = 460

        self.filter_combo_box = CustomBox(
            20,
            pos_y_row1,
            160,
            50,
            self,
            item_list=[
                "Filter Type:",
                "Mean",
                "Laplacian of Gaussian",
                "Riesz-transformed LoG",
                "Laws Kernels",
                "Gabor",
                "Wavelets",
                "Simoncelli",
            ],
        )

        self.padding_type_combo_box = CustomBox(
            340, pos_y_row1, 150, 50, self, item_list=["Padding Type:", "constant", "nearest", "wrap", "reflect"]
        )
        self.padding_type_combo_box.hide()

        self.mean_filter_support_label = CustomLabel('Support:', 200, pos_y_row2, 100, 50, self, style="color: white;")
        self.mean_filter_support_text_field = CustomTextField("E.g. 15", 275, pos_y_row2, 75, 50, self)
        self.mean_filter_support_text_field.hide()
        self.mean_filter_support_label.hide()

        self.filter_dimension_combo_box = CustomBox(
            200, pos_y_row1, 120, 50, self, item_list=["Dimension:", "2D", "3D"]
        )
        self.filter_dimension_combo_box.hide()

        self.log_filter_sigma_label = CustomLabel('\u03c3 (mm):', 200, pos_y_row2, 200, 50, self, style="color: white;")
        self.log_filter_sigma_text_field = CustomTextField("E.g. 3", 290, pos_y_row2, 60, 50, self)
        self.log_filter_sigma_label.hide()
        self.log_filter_sigma_text_field.hide()

        self.log_filter_cutoff_label = CustomLabel(
            'Cutoff (in \u03c3):', 375, pos_y_row2, 200, 50, self, style="color: white;"
        )
        self.log_filter_cutoff_text_field = CustomTextField("E.g. 4", 480, pos_y_row2, 60, 50, self)
        self.log_filter_cutoff_label.hide()
        self.log_filter_cutoff_text_field.hide()

        self.laws_filter_response_map_label = CustomLabel(
            'Response Map:', 200, pos_y_row2, 200, 50, self, style="color: white;"
        )
        self.laws_filter_response_map_text_field = CustomTextField("E.g. L5E5", 325, pos_y_row2, 100, 50, self)
        self.laws_filter_rot_inv_combo_box = CustomBox(
            510, pos_y_row1, 170, 50, self, item_list=['Rotation invariance:', 'Enable', 'Disable']
        )
        self.laws_filter_distance_label = CustomLabel(
            'Distance:', 500, pos_y_row2, 200, 50, self, style="color: white;"
        )
        self.laws_filter_distance_text_field = CustomTextField("E.g. 5", 580, pos_y_row2, 60, 50, self)
        self.laws_filter_pooling_combo_box = CustomBox(
            700, pos_y_row1, 120, 50, self, item_list=['Pooling:', 'max', 'min', 'average']
        )
        self.laws_filter_energy_map_combo_box = CustomBox(
            840, pos_y_row1, 140, 50, self, item_list=['Energy map:', 'Enable', 'Disable']
        )
        self.laws_filter_response_map_label.hide()
        self.laws_filter_response_map_text_field.hide()
        self.laws_filter_rot_inv_combo_box.hide()
        self.laws_filter_distance_label.hide()
        self.laws_filter_distance_text_field.hide()
        self.laws_filter_pooling_combo_box.hide()
        self.laws_filter_energy_map_combo_box.hide()

        self.wavelet_filter_type_combo_box = CustomBox(
            200, pos_y_row2, 170, 50, self, item_list=["Wavelet type:", "db3", "db2", "coif1", "haar"]
        )
        self.wavelet_filter_type_combo_box.hide()
        self.wavelet_filter_response_map_combo_box = CustomBox(
            390, pos_y_row2, 150, 50, self, item_list=['Response Map:']
        )
        self.wavelet_filter_response_map_combo_box.hide()
        self.wavelet_filter_response_map_2d_combo_box = CustomBox(
            390, pos_y_row2, 150, 50, self, item_list=['Response Map:', 'LL', 'HL', 'LH', 'HH']
        )
        self.wavelet_filter_response_map_2d_combo_box.hide()
        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            390,
            pos_y_row2,
            150,
            50,
            self,
            item_list=['Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"],
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            390,
            pos_y_row2,
            150,
            50,
            self,
            item_list=['Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"],
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_decomposition_level_combo_box = CustomBox(
            560, pos_y_row2, 175, 50, self, item_list=['Decomposition level:', '1', '2']
        )
        self.wavelet_filter_decomposition_level_combo_box.hide()

        self.wavelet_filter_rot_inv_combo_box = CustomBox(
            750, pos_y_row2, 175, 50, self, item_list=['Rotation invariance:', 'Enable', 'Disable']
        )
        self.wavelet_filter_rot_inv_combo_box.hide()

        # Gabor
        element_height = 50
        self.gabor_res_label = CustomLabel(
            'Resolution (mm/px):', 200, pos_y_row2, 150, element_height, self, style="color: white;"
        )
        self.gabor_res_label.hide()
        self.gabor_res_field = CustomTextField("", 350, pos_y_row2, 60, element_height, self)
        self.gabor_res_field.hide()

        self.gabor_sigma_label = CustomLabel(
            '\u03c3 (mm):', 430, pos_y_row2, 60, element_height, self, style="color: white;"
        )
        self.gabor_sigma_label.hide()
        self.gabor_sigma_field = CustomTextField("", 490, pos_y_row2, 60, element_height, self)
        self.gabor_sigma_field.hide()

        self.gabor_lambda_label = CustomLabel(
            '\u03bb (mm):', 570, pos_y_row2, 60, element_height, self, style="color: white;"
        )
        self.gabor_lambda_label.hide()
        self.gabor_lambda_field = CustomTextField("", 630, pos_y_row2, 60, element_height, self)
        self.gabor_lambda_field.hide()

        self.gabor_gamma_label = CustomLabel(
            '\u03b3:', 710, pos_y_row2, 20, element_height, self, style="color: white;"
        )
        self.gabor_gamma_label.hide()
        self.gabor_gamma_field = CustomTextField("", 730, pos_y_row2, 60, element_height, self)
        self.gabor_gamma_field.hide()

        self.gabor_theta_label = CustomLabel(
            '\u03b8/\u0394\u03b8 (rad):', 810, pos_y_row2, 80, element_height, self, style="color: white;"
        )
        self.gabor_theta_label.hide()
        self.gabor_theta_field = CustomTextField("", 890, pos_y_row2, 60, element_height, self)
        self.gabor_theta_field.hide()

        self.gabor_rotinv_box = CustomBox(
            510, pos_y_row1, 180, element_height, self, item_list=['Rotation invariance:', 'Enable', 'Disable']
        )
        self.gabor_rotinv_box.hide()
        self.gabor_ortho_box = CustomBox(
            710, pos_y_row1, 170, element_height, self, item_list=['Orthogonal planes:', 'Enable', 'Disable']
        )
        self.gabor_ortho_box.hide()

        # Riesz and Simoncelli
        self.riesz_order_label = CustomLabel(
            'Riesz order:', 510, pos_y_row1, 100, element_height, self, style="color: white;"
        )
        self.riesz_order_field = CustomTextField("E.g. 1,0,0", 610, pos_y_row1, 110, element_height, self)
        self.structure_tensor_sigma_label = CustomLabel(
            'Tensor σ (mm):', 570, pos_y_row2, 120, element_height, self, style="color: white;"
        )
        self.structure_tensor_sigma_field = CustomTextField("Optional", 690, pos_y_row2, 90, element_height, self)
        for widget in self._new_filter_widgets():
            widget.hide()

        self.run_button = CustomButton(
            'RUN',
            600,
            590,
            80,
            50,
            self,
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

        if self.input_params['filter_type'] == 'Riesz-transformed LoG':
            if not self.log_filter_sigma_text_field.text().strip():
                raise InvalidInputParametersError("Enter Sigma")
            if not self.log_filter_cutoff_text_field.text().strip():
                raise InvalidInputParametersError("Enter Cutoff")
            if not self.riesz_order_field.text().strip():
                raise InvalidInputParametersError("Enter Riesz order")

        if self.input_params['filter_type'] == 'Simoncelli':
            if self.wavelet_filter_decomposition_level_combo_box.currentText() == 'Decomposition level:':
                raise InvalidInputParametersError("Select Simoncelli decomposition level")

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
            if (
                self.wavelet_filter_response_map_3d_combo_box.currentText() == 'Response Map:'
                and self.filter_dimension_combo_box.currentText() == '3D'
            ) or (
                self.wavelet_filter_response_map_2d_combo_box.currentText() == 'Response Map:'
                and self.filter_dimension_combo_box.currentText() == '2D'
            ):
                error_msg = "Select Response Map"
                raise InvalidInputParametersError(error_msg)

        if self.input_params['filter_type'] == 'Gabor':
            for fld, name in [
                (self.gabor_res_field, "Resolution"),
                (self.gabor_sigma_field, "Sigma"),
                (self.gabor_lambda_field, "Lambda"),
                (self.gabor_gamma_field, "Gamma"),
                (self.gabor_theta_field, "Theta"),
            ]:
                if not fld.text().strip():
                    raise InvalidInputParametersError(f"Enter Gabor {name}!")
            if self.gabor_rotinv_box.currentText() == 'Rotation invariance:':
                raise InvalidInputParametersError("Select Gabor rotation‑invariance")
            if self.gabor_ortho_box.currentText() == 'Orthogonal planes:':
                raise InvalidInputParametersError("Select Gabor orthogonal‑planes")
            if self.filter_dimension_combo_box.currentText() == '3D':
                CustomInfoBox(
                    "True full 3D Gabor filtering is not implemented.\n Recommendation: Set dimensionality to 2D and enable orthogonal planes. This will provide a light‐weight, “three‐plane” 2D approximation to true 3D Gabor filtering."
                ).response()

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
            'filter_wavelet_rot_inv': self.wavelet_filter_rot_inv_combo_box.currentText(),
            'filter_gabor_res_mm': self.gabor_res_field.text(),
            'filter_gabor_sigma_mm': self.gabor_sigma_field.text(),
            'filter_gabor_lambda_mm': self.gabor_lambda_field.text(),
            'filter_gabor_gamma': self.gabor_gamma_field.text(),
            'filter_gabor_theta': self.gabor_theta_field.text(),
            'filter_gabor_rotinv': self.gabor_rotinv_box.currentText(),
            'filter_gabor_ortho': self.gabor_ortho_box.currentText(),
            'filter_riesz_order': self.riesz_order_field.text(),
            'filter_structure_tensor_sigma_mm': self.structure_tensor_sigma_field.text(),
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

        if IS_FROZEN:
            backend_hint = "threads"
            self.logger.info("Frozen state. Set backend_hint to threads")
        else:
            backend_hint = "processes"
            self.logger.info("Not frozen state. Set backend_hint to processes")

        batch_filter = create_batch_filter_from_input_params(self.input_params, backend_hint)
        try:
            list_of_patient_folders = batch_filter.plan()
        except InvalidInputParametersError as e:
            self.logger.error(e)
            CustomWarningBox(str(e)).response()
            return

        if list_of_patient_folders:
            progress_dialog = ProcessingProgressDialog("Filtering Progress", len(list_of_patient_folders), self)
            progress_dialog.start()

            def work(progress_callback):
                return batch_filter.run(progress_callback=progress_callback)

            worker = ProcessingWorker(work)
            thread = QThread(self)

            def cleanup():
                progress_dialog.finish()
                thread.quit()
                thread.wait()
                worker.deleteLater()
                thread.deleteLater()

            def handle_finished(_result):
                cleanup()
                self.logger.info(
                    "Filtering finished! Processed: %s, skipped: %s, failed: %s",
                    _result.processed_count,
                    _result.skipped_count,
                    _result.failed_count,
                )
                for case_result in _result.errors:
                    self.logger.error("%s: %s", case_result.case_name, case_result.error)
                CustomInfoBox("Filtering finished!").response()

            def handle_error(exc):
                cleanup()
                self.logger.error(exc, exc_info=True)
                CustomErrorBox(str(exc)).response()

            worker.moveToThread(thread)
            worker.progress.connect(progress_dialog.update_progress)
            worker.finished.connect(handle_finished)
            worker.error.connect(handle_error)
            thread.started.connect(worker.run)
            self._processing_thread = thread
            self._processing_worker = worker
            thread.start()
        else:
            CustomWarningBox("No patients to filter.").response()
            return

    def save_settings(self):
        """
        Update specific fields in the config.json file related to filtering settings
        without overwriting existing data.
        """
        self.get_input_parameters()
        data = {'filtering_' + key: value for key, value in self.input_params.items()}
        file_path = get_config_path()

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
        file_path = get_config_path()
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
                    data.get('filtering_input_image_modality', 'Imaging Modality:')
                )
                self.number_of_threads_combo_box.setCurrentText(data.get('filtering_number_of_threads', 'Threads:'))
                self.nifti_image_text_field.setText(data.get('filtering_nifti_image_name', ''))
                self.filter_combo_box.setCurrentText(data.get('filtering_filter_type', 'Filter Type:'))
                self.filter_dimension_combo_box.setCurrentText(data.get('filtering_filter_dimension', 'Dimension:'))
                self.padding_type_combo_box.setCurrentText(data.get('filtering_filter_padding_type', 'Padding Type:'))
                self.mean_filter_support_text_field.setText(data.get('filtering_filter_mean_support', ''))
                self.log_filter_sigma_text_field.setText(data.get('filtering_filter_log_sigma', ''))
                self.log_filter_cutoff_text_field.setText(data.get('filtering_filter_log_cutoff', ''))
                self.laws_filter_response_map_text_field.setText(data.get('filtering_filter_laws_response_map', ''))
                self.laws_filter_rot_inv_combo_box.setCurrentText(
                    data.get('filtering_filter_laws_rot_inv', 'Rotation invariance:')
                )
                self.laws_filter_distance_text_field.setText(data.get('filtering_filter_laws_distance', ''))
                self.laws_filter_pooling_combo_box.setCurrentText(data.get('filtering_filter_laws_pooling', 'Pooling:'))
                self.laws_filter_energy_map_combo_box.setCurrentText(
                    data.get('filtering_filter_laws_energy_map', 'Energy map:')
                )
                self.wavelet_filter_response_map_2d_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_resp_map_2D', 'Response Map:')
                )
                self.wavelet_filter_response_map_3d_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_resp_map_3D', 'Response Map:')
                )
                self.wavelet_filter_type_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_type', 'Wavelet type:')
                )
                self.wavelet_filter_decomposition_level_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_decomp_lvl', 'Decomposition Lvl.:')
                )
                self.wavelet_filter_rot_inv_combo_box.setCurrentText(
                    data.get('filtering_filter_wavelet_rot_inv', 'Rotation invariance:')
                )
                # Gabor-specific fields:
                self.gabor_res_field.setText(data.get('filtering_filter_gabor_res_mm', ''))
                self.gabor_sigma_field.setText(data.get('filtering_filter_gabor_sigma_mm', ''))
                self.gabor_lambda_field.setText(data.get('filtering_filter_gabor_lambda_mm', ''))
                self.gabor_gamma_field.setText(data.get('filtering_filter_gabor_gamma', ''))
                self.gabor_theta_field.setText(data.get('filtering_filter_gabor_theta', ''))
                self.gabor_rotinv_box.setCurrentText(data.get('filtering_filter_gabor_rotinv', 'Rotation invariance:'))
                self.gabor_ortho_box.setCurrentText(data.get('filtering_filter_gabor_ortho', 'Orthogonal planes:'))
                self.riesz_order_field.setText(data.get('filtering_filter_riesz_order', ''))
                self.structure_tensor_sigma_field.setText(data.get('filtering_filter_structure_tensor_sigma_mm', ''))

        except FileNotFoundError:
            print("No previous data found!")

    def _filter_combo_box_changed(self, text):
        self._set_combo_box_items(
            self.padding_type_combo_box, ['Padding Type:', 'constant', 'nearest', 'wrap', 'reflect']
        )
        self._set_combo_box_items(self.wavelet_filter_decomposition_level_combo_box, ['Decomposition level:', '1', '2'])
        for widget in self._new_filter_widgets():
            widget.hide()
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
            # Gabor widgets:
            self.gabor_res_label.hide()
            self.gabor_res_field.hide()
            self.gabor_sigma_label.hide()
            self.gabor_sigma_field.hide()
            self.gabor_lambda_label.hide()
            self.gabor_lambda_field.hide()
            self.gabor_gamma_label.hide()
            self.gabor_gamma_field.hide()
            self.gabor_theta_label.hide()
            self.gabor_theta_field.hide()
            self.gabor_rotinv_box.hide()
            self.gabor_ortho_box.hide()

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
            # Gabor widgets:
            self.gabor_res_label.hide()
            self.gabor_res_field.hide()
            self.gabor_sigma_label.hide()
            self.gabor_sigma_field.hide()
            self.gabor_lambda_label.hide()
            self.gabor_lambda_field.hide()
            self.gabor_gamma_label.hide()
            self.gabor_gamma_field.hide()
            self.gabor_theta_label.hide()
            self.gabor_theta_field.hide()
            self.gabor_rotinv_box.hide()
            self.gabor_ortho_box.hide()

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
            # Gabor widgets:
            self.gabor_res_label.hide()
            self.gabor_res_field.hide()
            self.gabor_sigma_label.hide()
            self.gabor_sigma_field.hide()
            self.gabor_lambda_label.hide()
            self.gabor_lambda_field.hide()
            self.gabor_gamma_label.hide()
            self.gabor_gamma_field.hide()
            self.gabor_theta_label.hide()
            self.gabor_theta_field.hide()
            self.gabor_rotinv_box.hide()
            self.gabor_ortho_box.hide()

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
            # Gabor widgets:
            self.gabor_res_label.hide()
            self.gabor_res_field.hide()
            self.gabor_sigma_label.hide()
            self.gabor_sigma_field.hide()
            self.gabor_lambda_label.hide()
            self.gabor_lambda_field.hide()
            self.gabor_gamma_label.hide()
            self.gabor_gamma_field.hide()
            self.gabor_theta_label.hide()
            self.gabor_theta_field.hide()
            self.gabor_rotinv_box.hide()
            self.gabor_ortho_box.hide()

        elif text == 'Gabor':
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
            self.laws_filter_response_map_label.hide()
            self.laws_filter_response_map_text_field.hide()
            self.laws_filter_rot_inv_combo_box.hide()
            self.laws_filter_distance_label.hide()
            self.laws_filter_distance_text_field.hide()
            self.laws_filter_pooling_combo_box.hide()
            self.laws_filter_energy_map_combo_box.hide()
            # Gabor widgets:
            self.gabor_res_label.show()
            self.gabor_res_field.show()
            self.gabor_sigma_label.show()
            self.gabor_sigma_field.show()
            self.gabor_lambda_label.show()
            self.gabor_lambda_field.show()
            self.gabor_gamma_label.show()
            self.gabor_gamma_field.show()
            self.gabor_theta_label.show()
            self.gabor_theta_field.show()
            self.gabor_rotinv_box.show()
            self.gabor_ortho_box.show()

        elif text in ('Riesz-transformed LoG', 'Simoncelli'):
            if self.wavelet_filter_response_map_connected:
                self.filter_dimension_combo_box.currentTextChanged.disconnect()
                self.wavelet_filter_response_map_connected = False
            self._hide_existing_filter_parameter_widgets()
            self.padding_type_combo_box.show()
            self.filter_dimension_combo_box.show()
            self.riesz_order_label.show()
            self.riesz_order_field.show()
            if text == 'Riesz-transformed LoG':
                self.log_filter_sigma_label.show()
                self.log_filter_sigma_text_field.show()
                self.log_filter_cutoff_label.show()
                self.log_filter_cutoff_text_field.show()
                self.structure_tensor_sigma_label.show()
                self.structure_tensor_sigma_field.show()
            else:
                self._set_combo_box_items(self.padding_type_combo_box, ['Padding Type:', 'nearest', 'wrap'])
                self._set_combo_box_items(
                    self.wavelet_filter_decomposition_level_combo_box, ['Decomposition level:', '1', '2', '3']
                )
                self.wavelet_filter_decomposition_level_combo_box.show()

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

    def _new_filter_widgets(self):
        return (
            self.riesz_order_label,
            self.riesz_order_field,
            self.structure_tensor_sigma_label,
            self.structure_tensor_sigma_field,
        )

    @staticmethod
    def _set_combo_box_items(combo_box, items):
        """Replace a combo box's choices while retaining a still-valid selection."""
        selected = combo_box.currentText()
        combo_box.clear()
        combo_box.addItems(items)
        if selected in items:
            combo_box.setCurrentText(selected)

    def _hide_existing_filter_parameter_widgets(self):
        widgets = (
            self.mean_filter_support_text_field,
            self.mean_filter_support_label,
            self.log_filter_sigma_label,
            self.log_filter_sigma_text_field,
            self.log_filter_cutoff_label,
            self.log_filter_cutoff_text_field,
            self.wavelet_filter_type_combo_box,
            self.wavelet_filter_response_map_3d_combo_box,
            self.wavelet_filter_response_map_2d_combo_box,
            self.wavelet_filter_response_map_combo_box,
            self.wavelet_filter_decomposition_level_combo_box,
            self.wavelet_filter_rot_inv_combo_box,
            self.laws_filter_response_map_label,
            self.laws_filter_response_map_text_field,
            self.laws_filter_rot_inv_combo_box,
            self.laws_filter_distance_label,
            self.laws_filter_distance_text_field,
            self.laws_filter_pooling_combo_box,
            self.laws_filter_energy_map_combo_box,
            self.gabor_res_label,
            self.gabor_res_field,
            self.gabor_sigma_label,
            self.gabor_sigma_field,
            self.gabor_lambda_label,
            self.gabor_lambda_field,
            self.gabor_gamma_label,
            self.gabor_gamma_field,
            self.gabor_theta_label,
            self.gabor_theta_field,
            self.gabor_rotinv_box,
            self.gabor_ortho_box,
        )
        for widget in widgets:
            widget.hide()

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
            ('Imaging Modality:', self.input_imaging_mod_combo_box),
        ]
        for message, combo_box in required_selections:
            if combo_box.currentText() == message:
                warning_msg = f"Select {message.split(':')[0]}"
                raise InvalidInputParametersError(warning_msg)
