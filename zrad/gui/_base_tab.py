import os
from abc import ABC, ABCMeta, abstractmethod
from multiprocessing import cpu_count

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomInfo, CustomWarningBox
from ..logic.exceptions import InvalidInputParametersError, DataStructureError
from ..logic.image import Image


class BaseTabMeta(ABCMeta, type(QWidget)):
    pass


def get_imaging_filepath(input_dir, patient_folder, filename, imaging_format='dicom'):
    filepath = os.path.join(input_dir, patient_folder, filename)
    if imaging_format == 'dicom':
        if os.path.isfile(filepath + '.dcm'):
            return filepath + '.dcm'
        elif os.path.isfile(filepath):
            return filepath
        else:
            return None
    elif imaging_format == 'nifti':
        if os.path.isfile(filepath + '.nii.gz'):
            return filepath + '.nii.gz'
        elif os.path.isfile(filepath + '.nii'):
            return filepath + '.nii'
        elif os.path.isfile(filepath):
            return filepath
        else:
            return None


def load_images(input_params, patient_folder):
    """Loads image and optional filtered image based on the data type."""
    image = Image()
    filtered_image = None
    input_dir = input_params['input_directory']
    input_data_type = input_params['input_data_type']
    nifti_image_name = input_params['nifti_image_name']
    input_imaging_modality = input_params["input_imaging_modality"]
    try:
        nifti_filtered_image_name = input_params['nifti_filtered_image_name']
    except KeyError:
        nifti_filtered_image_name = None

    if input_data_type.lower() == 'nifti':
        # Construct file path
        image_path = get_imaging_filepath(input_dir, patient_folder, nifti_image_name, imaging_format='nifti')

        if image_path:
            # Read image
            try:
                image.read_nifti_image(image_path)
            except Exception as e:
                error_msg = f"Error reading filtered NIfTI image: {e}"
                raise DataStructureError(error_msg)
        else:
            raise FileNotFoundError(f"{os.path.join(input_dir, patient_folder, nifti_image_name)}")

        # Read filtered image if specified
        if nifti_filtered_image_name:
            filtered_image = Image()
            filtered_image_path = get_imaging_filepath(input_dir, patient_folder, nifti_filtered_image_name,
                                                       imaging_format='nifti')
            if filtered_image_path:
                try:
                    filtered_image.read_nifti_image(filtered_image_path)
                except Exception as e:
                    error_msg = f"Error reading filtered NIfTI image: {e}"
                    raise DataStructureError(error_msg)
            else:
                raise FileNotFoundError(f"{os.path.join(input_dir, patient_folder, nifti_filtered_image_name)}")
    elif input_data_type.lower() == 'dicom':
        # Construct file path for DICOM
        image_path = os.path.join(input_dir, patient_folder)

        # Read image
        image.read_dicom_image(image_path, modality=input_imaging_modality)
    else:
        warning_msg = f"Invalid input data type: {input_data_type}"
        CustomWarningBox(warning_msg).response()
        raise InvalidInputParametersError(warning_msg)

    if filtered_image:
        return image, filtered_image
    else:
        return image


def load_mask(input_params, patient_folder, structure_name, image):
    """Loads a mask based on the data type."""
    input_dir = input_params["input_directory"]
    input_data_type = input_params["input_data_type"]

    mask = Image()

    if input_data_type == 'nifti':
        mask_path = get_imaging_filepath(input_dir, patient_folder, structure_name, imaging_format='nifti')
        if mask_path:
            try:
                mask.read_nifti_mask(image, mask_path)
            except Exception as e:
                error_msg = f"Error reading NIfTI mask: {e}"
                raise DataStructureError(error_msg)
    elif input_data_type == 'dicom':
        try:
            mask.read_dicom_mask(rtstruct_path=input_params['rtstruct_path'], structure_name=structure_name, image=image)
        except Exception as e:
            error_msg = f"Error reading DICOM mask: {e}"
            raise DataStructureError(error_msg)
    else:
        warning_msg = f"Invalid input data type: {input_data_type}"
        CustomWarningBox(warning_msg).response()
        raise InvalidInputParametersError(warning_msg)
    return mask


class BaseTab(QWidget, ABC, metaclass=BaseTabMeta):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize layout and size
        self.setMinimumSize(1220, 640)
        self.layout = QVBoxLayout(self)

        # Initialize UI components
        self.init_ui_components()

        # Initialize parameters
        self.input_params = None

    def init_ui_components(self):
        """Initialize the user interface components for this tab."""
        self.init_io_elements()

        self.input_data_type_combo_box = CustomBox(
            20, 300, 160, 50, self,
            item_list=["Data Type:", "DICOM", "NIfTI"]
        )

        # Number of Threads ComboBox
        no_of_threads = ['Threads:'] + [str(i+1) for i in range(cpu_count())]
        self.number_of_threads_combo_box = CustomBox(
            20, 140, 160, 50, self,
            item_list=no_of_threads
        )

        self.run_button = CustomButton('RUN', 600, 590, 80, 50, self, style=False)

    def init_io_elements(self):
        # Imaging Modality selector
        self.input_imaging_mod_combo_box = CustomBox(
            200, 140, 160, 50, self,
            item_list=[
                "Imaging Modality:", "CT", "MRI", "PET"
            ]
        )

        # Load Directory Button and Label
        self.load_dir_button = CustomButton(
            'Input Directory',
            20, 50, 160, 50, self,
            style=True)
        self.load_dir_text_field = CustomTextField(
            '',
            200, 50, 1000, 50,
            self,
            style=True)
        self.load_dir_text_field.setAlignment(Qt.AlignCenter)
        self.load_dir_button.clicked.connect(lambda: self.open_directory(is_load_dir=True))

        #  Start and Stop Folder TextFields and Labels
        self.start_folder_label = CustomLabel(
            'Start Folder:',
            400, 140, 100, 50, self,
            style="color: white;"
        )
        self.start_folder_text_field = CustomTextField(
            "",
            500, 140, 60, 50, self
        )

        for pos_x in [570, 770, 1150]:
            CustomInfo(
                ' i',
                'Folder Selection Guidelines:\n'
                '• Start and Stop Folders: Use these options only if all folders in the directory have integer names.\n'
                '• List of Folders: You can specify the folders of interest here (e.g. 1, 2a, 3_1), even if their names are non-integer.\n'
                '• Running All Folders: Leave the Start, Stop, and List of Folders fields empty to run all folders in the Input Directory, regardless of whether their names are integers or not.',
                pos_x, 140, 14, 14, self
            )

        self.stop_folder_label = CustomLabel(
            'Stop Folder:',
            600, 140, 100, 50, self,
            style="color: white;")
        self.stop_folder_text_field = CustomTextField(
            "",
            700, 140, 60, 50, self
        )
        # List of Patient Folders TextField and Label
        self.list_of_patient_folders_label = CustomLabel(
            'List of Folders:',
            800, 140, 110, 50, self,
            style="color: white;"
        )
        self.list_of_patient_folders_text_field = CustomTextField(
            "E.g. 1, 5, 10, 34, ...",
            920, 140, 220, 50, self)

        # Save Directory Button and Label
        self.save_dir_button = CustomButton(
            'Output Directory',
            20, 220, 160, 50, self,
            style=True)
        self.save_dir_text_field = CustomTextField(
            '',
            200, 220, 1000, 50,
            self,
            style=True)
        self.save_dir_text_field.setAlignment(Qt.AlignCenter)
        self.save_dir_button.clicked.connect(lambda: self.open_directory(is_load_dir=False))

    def open_directory(self, is_load_dir):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        if is_load_dir:
            initial_directory = self.load_dir_text_field.text() if self.load_dir_text_field.text() else ""
        else:
            initial_directory = self.save_dir_text_field.text() if self.save_dir_text_field.text() else ""

        directory = QFileDialog.getExistingDirectory(self, "Select Directory", initial_directory, options=options)

        if directory and is_load_dir:
            self.load_dir_text_field.setText(directory)
        elif directory and not is_load_dir:
            self.save_dir_text_field.setText(directory)

    def file_type_changed(self, text):
        """Handle changes in the selected input file type."""
        if text == "DICOM":
            self._show_dicom_elements()
            self._hide_nifti_elements()
        elif text == "NIfTI":
            self._show_nifti_elements()
            self._hide_dicom_elements()
        else:
            self._hide_dicom_elements()
            self._hide_nifti_elements()

    def _validate_io_directories(self):
        """Validate that necessary directories are selected."""
        required_dirs = [
            ('Select Load Directory!', self.load_dir_text_field.text().strip()),
            ('Select Save Directory', self.save_dir_text_field.text().strip()),
        ]
        for warning, text in required_dirs:
            if not text:
                warning_msg = warning
                raise InvalidInputParametersError(warning_msg)

    def _get_input_imaging_modality(self):
        """
        Retrieve the selected input imaging modality from the combo box.

        Returns:
            str: The selected input imaging modality.
            None: If no valid selection is made.
        """
        # Retrieve the current text from the combo box
        input_imaging_modality = self.input_imaging_mod_combo_box.currentText()

        # Check if a valid modality has been selected
        if input_imaging_modality == 'Imaging Modality:':
            CustomWarningBox("Select Input Imaging Modality").response()
            return None

        return input_imaging_modality

    def _get_input_data_type(self):
        """
        Retrieve the selected input imaging data type from the combo box.

        Returns:
            str: The selected input imaging modality.
            None: If no valid selection is made.
        """
        # Retrieve the current text from the combo box
        input_data_type = self.input_data_type_combo_box.currentText().strip().lower()

        # Check if a valid modality has been selected
        if input_data_type == 'Data Type:':
            CustomWarningBox("Select Input Imaging Data Type").response()
            return None

        return input_data_type

    def get_patient_folders(self):
        input_dir = self.input_params["input_directory"]
        output_dir = self.input_params["output_directory"]
        start_folder = self.input_params["start_folder"]
        stop_folder = self.input_params["stop_folder"]
        list_of_patient_folders = self.input_params["list_of_patient_folders"]

        def get_list_folders_in_defined_range(folder_to_start, folder_to_stop, directory_path):
            list_of_folders = []
            for folder in os.listdir(directory_path):
                if folder.isdigit():
                    folder = int(folder)
                    if int(folder_to_start) <= folder <= int(folder_to_stop):
                        list_of_folders.append(str(folder))
            return list_of_folders

        if os.path.exists(input_dir):
            input_dir = input_dir
        else:
            warning_msg = f"Load directory '{input_dir}' does not exist."
            CustomWarningBox(warning_msg).response()
            raise InvalidInputParametersError(warning_msg)

        if os.path.exists(output_dir):
            pass
        else:
            os.makedirs(output_dir)

        if start_folder and stop_folder:  # Check if both are non-empty strings
            # List folders in the defined range
            list_of_patient_folders = get_list_folders_in_defined_range(start_folder, stop_folder, input_dir)
        elif list_of_patient_folders and list_of_patient_folders != ['']:  # Check if it is a valid, non-empty list
            list_of_patient_folders = list_of_patient_folders
        elif not list_of_patient_folders and not start_folder and not stop_folder:  # All are None or empty
            list_of_patient_folders = [
                e for e in os.listdir(input_dir)
                if not e.startswith('.') and os.path.isdir(os.path.join(input_dir, e))
            ]
        else:
            warning_msg = "Incorrectly selected patient folders."
            CustomWarningBox(warning_msg).response()
            raise InvalidInputParametersError(warning_msg)
        return list_of_patient_folders

    @staticmethod
    def get_text_from_text_field(text_field):
        """Return text from the field if it is not empty; otherwise, None."""
        return text_field.strip() or None

    @staticmethod
    def get_list_from_text_field(text_field):
        """Convert comma-separated text to a list; return None if empty."""
        text = text_field.strip()
        return [item.strip() for item in text.split(",")] if text else None

    @abstractmethod
    def get_input_parameters(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def run_selection(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def init_dicom_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def init_nifti_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def _show_dicom_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def _hide_nifti_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def _hide_dicom_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def _show_nifti_elements(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def load_settings(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    @abstractmethod
    def save_settings(self):
        """Abstract method that must be implemented by subclasses."""
        pass

    def check_common_input_parameters(self):
        # Validate directories
        self._validate_io_directories()

        self.input_params["input_directory"] = self.get_text_from_text_field(self.input_params["input_directory"])
        self.input_params["output_directory"] = self.get_text_from_text_field(self.input_params["output_directory"])
        self.input_params["start_folder"] = self.get_text_from_text_field(self.input_params["start_folder"])
        self.input_params["stop_folder"] = self.get_text_from_text_field(self.input_params["stop_folder"])
        self.input_params["list_of_patient_folders"] = self.get_list_from_text_field(self.input_params["list_of_patient_folders"])
        self.input_params["input_data_type"] = self._get_input_data_type()
        self.input_params["input_imaging_modality"] = self._get_input_imaging_modality()
        self.input_params["number_of_threads"] = int(self.number_of_threads_combo_box.currentText())
