import json
import logging
import os
import sys
from datetime import datetime

from PyQt5.QtCore import QThread
from joblib import Parallel, delayed

from ._base_tab import BaseTab, load_images, load_mask
from .toolbox_gui import (
    CustomCheckBox,
    CustomInfo,
    CustomInfoBox,
    CustomLabel,
    CustomTextField,
    CustomWarningBox,
    ProcessingProgressDialog,
    ProcessingWorker,
)
from ..exceptions import InvalidInputParametersError, DataStructureError
from ..image import get_all_structure_names, get_dicom_files
from ..visualization import Visualization
from ..toolbox_logic import get_logger, close_all_loggers, joblib_progress

logging.captureWarnings(True)

IS_FROZEN = getattr(sys, 'frozen', False)


def process_patient_folder(input_params, patient_folder, structure_set):

    local_params = dict(input_params)

    """Function to process each patient folder, used for parallel processing."""
    # Logger
    logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = get_logger(logger_date_time + '_Visualization')
    logger.info(f"Visualization patient's {patient_folder} image.")

    try:
        image = load_images(local_params, patient_folder)
    except (DataStructureError, ValueError) as e:
        logger.error(e)
        logger.error(f"Patient {patient_folder} could not be loaded and is skipped.")
        return

    if local_params["input_data_type"] == 'dicom':
        input_directory = os.path.join(local_params["input_directory"], patient_folder)
        rtstruct_paths = get_dicom_files(input_directory, modality='RTSTRUCT')

        if rtstruct_paths:
            local_params['rtstruct_path'] = rtstruct_paths[0]['file_path']
            if local_params["use_all_structures"]:
                structure_set = get_all_structure_names(local_params['rtstruct_path'])
        else:
            local_params['rtstruct_path'] = None
    masks_set = []
    if structure_set:
        for mask_name in structure_set:
            mask = load_mask(local_params, patient_folder, mask_name, image)
            if mask and mask.array is not None:
                logger.info(
                    f"Processing patient's {patient_folder} ROI: {mask_name}.")
                masks_set.append({mask_name: mask})

    return {"image": image, "image_name": patient_folder, "masks": masks_set}




class VisualizationTab(BaseTab):
    def __init__(self):
        super().__init__(visual_tab=True)
        self.init_dicom_elements()
        self.init_nifti_elements()
        self.connect_signals()

    def init_dicom_elements(self):
        self.dicom_structures_label = CustomLabel(
            'Structures:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.dicom_structures_text_field = CustomTextField(
            "E.g. CTV, liver, ...",
            300, 300, 400, 50, self
        )

        self.dicom_structures_info_label = CustomInfo(
            ' i',
            'Type ROIs of interest (e.g. CTV, liver).',
            710, 300, 14, 14, self
        )

        self.use_all_structures_check_box = CustomCheckBox(
            'All structures',
            750, 300, 150, 50, self)

        self._hide_dicom_elements()

    def init_nifti_elements(self):
        """Initialize UI components related to NIfTI elements."""
        # Structures
        self.nifti_structures_label = CustomLabel(
            'NIfTI Masks:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_structures_text_field = CustomTextField(
            "E.g. CTV, liver, ...",
            300, 300, 230, 50, self
        )
        self.nifti_structures_info_label = CustomInfo(
            ' i',
            'Provide the names of the NIfTI masks you are interested in, excluding the file extensions.'
            '\nFor example, if the files you are interested in are GTV.nii.gz and liver.nii, enter: GTV, liver.',
            545, 300, 14, 14, self
        )

        # Image
        self.nifti_image_label = CustomLabel(
            'NIfTI Image:',
            600, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT",
            700, 300, 120, 50, self
        )
        self.nifti_image_info_label = CustomInfo(
            ' i',
            'Specify NIfTI image file without file extension',
            830, 300, 14, 14, self
        )
        self._hide_nifti_elements()

    def connect_signals(self):
        self.input_data_type_combo_box.currentTextChanged.connect(self.file_type_changed)
        self.use_all_structures_check_box.stateChanged.connect(self._use_all_structures_changed)
        self.run_button.clicked.connect(self.run_selection)

    def _show_dicom_elements(self):
        self.dicom_structures_label.show()
        self.dicom_structures_text_field.show()
        self.dicom_structures_info_label.show()
        self.use_all_structures_check_box.show()

    def _hide_dicom_elements(self):
        self.use_all_structures_check_box.setChecked(False)
        self.dicom_structures_text_field.clear()
        self.dicom_structures_label.hide()
        self.dicom_structures_text_field.hide()
        self.dicom_structures_info_label.hide()
        self.use_all_structures_check_box.hide()

    def _show_nifti_elements(self):
        self.nifti_structures_label.show()
        self.nifti_structures_text_field.show()
        self.nifti_structures_info_label.show()
        self.nifti_image_label.show()
        self.nifti_image_text_field.show()
        self.nifti_image_info_label.show()

    def _hide_nifti_elements(self):
        self.nifti_structures_text_field.clear()
        self.nifti_image_text_field.clear()
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()
        self.nifti_structures_info_label.hide()
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label.hide()

    def _validate_combo_selections(self):
        """Validate combo box selections."""
        required_selections = [
            ('Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Imaging Modality:', self.input_imaging_mod_combo_box)
        ]

        for message, combo_box in required_selections:
            if combo_box.currentText() == message:
                warning_msg = f"Select {message.split(':')[0]}"
                raise InvalidInputParametersError(warning_msg)

    def _use_all_structures_changed(self):
        if self.use_all_structures_check_box.isChecked():
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.dicom_structures_info_label.hide()
        else:
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.dicom_structures_info_label.show()

    def file_type_changed(self, text):
        """Handle changes in the selected input file type."""
        if text == "DICOM":
            self._show_dicom_elements()
            self._hide_nifti_elements()
            if self.use_all_structures_check_box.isChecked():
                self.dicom_structures_label.hide()
                self.dicom_structures_text_field.hide()
                self.dicom_structures_info_label.hide()
        elif text == "NIfTI":
            self._show_nifti_elements()
            self._hide_dicom_elements()
        else:
            self._hide_dicom_elements()
            self._hide_nifti_elements()

    def check_input_parameters(self):
        # Validate combo box selections
        self._validate_combo_selections()
        self.check_common_input_parameters()

        self.input_params["nifti_image_name"] = self.get_text_from_text_field(self.input_params["nifti_image_name"])
        self.input_params["nifti_structures"] = self.get_list_from_text_field(self.input_params["nifti_structures"])
        self.input_params["dicom_structures"] = self.get_list_from_text_field(self.input_params["dicom_structures"])

    def get_input_parameters(self):
        """Collect input parameters from UI elements."""
        input_parameters = {
            'input_directory': self.load_dir_text_field.text(),
            'number_of_threads': self.number_of_threads_combo_box.currentText(),
            'start_folder': self.start_folder_text_field.text(),
            'stop_folder': self.stop_folder_text_field.text(),
            'list_of_patient_folders': self.list_of_patient_folders_text_field.text(),
            'input_data_type': self.input_data_type_combo_box.currentText(),
            'dicom_structures': self.dicom_structures_text_field.text(),
            'nifti_image_name': self.nifti_image_text_field.text(),
            'nifti_structures': self.nifti_structures_text_field.text(),
            'input_imaging_modality': self.input_imaging_mod_combo_box.currentText(),
            'use_all_structures': self.use_all_structures_check_box.checkState(),
        }
        self.input_params = input_parameters
        self.visual_tab = True

    def run_selection(self):
        """Prepare patient data in the background and visualize the results in the main thread."""
        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time + "_Preprocessing")
        self.logger.info("Preprocessing started")

        self.get_input_parameters()

        try:
            self.check_input_parameters()
        except InvalidInputParametersError as e:
            self.logger.error(e)
            CustomWarningBox(str(e)).response()
            return

        list_of_patient_folders = self.get_patient_folders()
        if not list_of_patient_folders:
            CustomWarningBox("No patients to calculate preprocess from.").response()
            return

        # Determine structure set
        structure_set = None
        input_data_type = self.input_params["input_data_type"].lower()

        if input_data_type == "nifti":
            structure_set = self.input_params.get("nifti_structures")
        elif input_data_type == "dicom":
            if not self.input_params.get("use_all_structures", False):
                structure_set = self.input_params.get("dicom_structures")

        # Select backend for joblib
        if IS_FROZEN:
            backend_hint = "threads"
            self.logger.info("Frozen state. Set backend_hint to threads")
        else:
            backend_hint = "processes"
            self.logger.info("Not frozen state. Set backend_hint to processes")

        progress_dialog = ProcessingProgressDialog(
            "Preprocessing Progress",
            len(list_of_patient_folders),
            self,
        )
        progress_dialog.start()

        n_jobs = self.input_params["number_of_threads"]

        def work(progress_callback):
            results = []

            if n_jobs == 1:
                for patient_folder in list_of_patient_folders:
                    result = process_patient_folder(
                        self.input_params,
                        patient_folder,
                        structure_set,
                    )
                    if result is not None:
                        results.append(result)
                    progress_callback(1)
            else:
                with joblib_progress(progress_callback=progress_callback):
                    parallel_results = Parallel(
                        n_jobs=n_jobs,
                        prefer=backend_hint,
                    )(
                        delayed(process_patient_folder)(
                            self.input_params,
                            patient_folder,
                            structure_set,
                        )
                        for patient_folder in list_of_patient_folders
                    )

                results = [result for result in parallel_results if result is not None]

            return results

        worker = ProcessingWorker(work)
        thread = QThread(self)

        def cleanup():
            progress_dialog.finish()
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._processing_thread = None
            self._processing_worker = None

        def handle_finished(patients_to_visualize):
            cleanup()

            if not patients_to_visualize:
                self.logger.warning("No valid patients could be loaded for visualization.")
                CustomWarningBox("No valid patients could be loaded for visualization.").response()
                return

            self.logger.info("Preprocessing finished successfully")
            self.viewer = Visualization(image_sets=patients_to_visualize)
            self.viewer.resize(1500, 900)
            self.viewer.show()
            self.viewer.raise_()
            self.viewer.activateWindow()

        def handle_error(exc):
            cleanup()
            self.logger.error(exc, exc_info=True)
            CustomWarningBox(str(exc)).response()

        worker.moveToThread(thread)
        worker.progress.connect(progress_dialog.update_progress)
        worker.finished.connect(handle_finished)
        worker.error.connect(handle_error)
        thread.started.connect(worker.run)

        self._processing_thread = thread
        self._processing_worker = worker

        thread.start()

    def save_settings(self):
        """
        Update specific fields in the config.json file without overwriting existing data.
        """
        # Data to be updated
        self.get_input_parameters()
        data = {'visual_' + key: value for key, value in self.input_params.items()}
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
        """
        Load input data from a JSON file.
        """
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_text_field.setText(data.get('visual_input_directory', ''))
                self.start_folder_text_field.setText(data.get('visual_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('visual_stop_folder', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('visual_number_of_threads', 'Threads:'))
                self.list_of_patient_folders_text_field.setText(data.get('visual_list_of_patient_folders', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('visual_input_data_type', 'Data Type:'))
                self.dicom_structures_text_field.setText(data.get('visual_dicom_structures', ''))
                self.nifti_image_text_field.setText(data.get('visual_nifti_image_name', ''))
                self.nifti_structures_text_field.setText(data.get('visual_nifti_structures', ''))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('visual_input_imaging_modality', 'Imaging Modality:'))
                self.use_all_structures_check_box.setCheckState(data.get('visual_use_all_structures', 0))

        except FileNotFoundError:
            print("No previous data found!")
