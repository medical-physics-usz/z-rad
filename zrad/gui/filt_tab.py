import json
from joblib import cpu_count

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog

from .toolbox_gui import CustomButton, CustomLabel, CustomBox, CustomTextField, CustomWarningBox
from ..logic.filtering import Filtering, Mean, LoG, Wavelets2D, Wavelets3D, Laws


class FilteringTab(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = None
        self.load_dir_button = None
        self.load_dir_label = None
        self.input_data_type_combo_box = None
        self.start_folder_label = None
        self.start_folder_text_field = None
        self.stop_folder_label = None
        self.stop_folder_text_field = None
        self.list_of_patient_folders_label = None
        self.list_of_patient_folders_text_field = None
        self.number_of_threads_combo_box = None
        self.save_dir_button = None
        self.save_dir_label = None
        self.input_imaging_mod_combo_box = None
        self.nifti_image_label = None
        self.nifti_image_text_field = None
        self.filter_combo_box = None
        self.padding_type_combo_box = None
        self.mean_filter_support_label = None
        self.mean_filter_support_text_field = None
        self.filter_dimension_combo_box = None
        self.log_filter_sigma_label = None
        self.log_filter_sigma_text_field = None
        self.log_filter_cutoff_label = None
        self.log_filter_cutoff_text_field = None
        self.laws_filter_response_map_label = None
        self.laws_filter_response_map_text_field = None
        self.laws_filter_rot_inv_combo_box = None
        self.laws_filter_distance_label = None
        self.laws_filter_distance_text_field = None
        self.laws_filter_pooling_combo_box = None
        self.laws_filter_energy_map_combo_box = None
        self.wavelet_filter_type_combo_box = None
        self.wavelet_filter_response_map_combo_box = None
        self.wavelet_filter_response_map_connected = False
        self.wavelet_filter_response_map_2d_combo_box = None
        self.wavelet_filter_response_map_3d_combo_box = None
        self.wavelet_filter_response_map_3d_combo_box = None
        self.wavelet_filter_decomposition_level_combo_box = None
        self.wavelet_filter_rot_inv_combo_box = None
        self.run_button = None

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

        if (self.input_imaging_mod_combo_box.currentText() == 'Imaging Mod.:'
                and CustomWarningBox("Select Input Imaging Modality").response()):
            return
        input_imaging_mod = self.input_imaging_mod_combo_box.currentText()

        # Collect values from GUI elements
        load_dir = self.load_dir_label.text()
        save_dir = self.save_dir_label.text()
        start_folder = self.start_folder_text_field.text().strip()
        stop_folder = self.stop_folder_text_field.text().strip()
        number_of_threads = int(self.number_of_threads_combo_box.currentText().split(" ")[0])
        input_data_type = self.input_data_type_combo_box.currentText()
        filter_type = self.filter_combo_box.currentText()
        filter_dimension = self.filter_dimension_combo_box.currentText()
        filter_padding_type = self.padding_type_combo_box.currentText()

        list_of_patient_folders = []
        if self.list_of_patient_folders_text_field.text() != '':
            list_of_patient_folders = [
                int(pat) for pat in str(self.list_of_patient_folders_text_field.text()).split(",")
            ]

        if (not start_folder or not stop_folder) and not list_of_patient_folders:
            CustomWarningBox("Incorrectly selected patient folders!").response()
            return

        if (not self.nifti_image_text_field.text().strip()
                and self.input_data_type_combo_box.currentText() == 'NIFTI'):
            CustomWarningBox("Enter NIFTI image").response()
            return
        nifti_image = self.nifti_image_text_field.text()

        if filter_type == 'Mean' and not self.mean_filter_support_text_field.text().strip():
            CustomWarningBox("Enter Support!").response()
            return
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

        filt_instance = Filtering(load_dir,
                                  list_of_patient_folders,
                                  input_imaging_mod,
                                  input_data_type,
                                  nifti_image,
                                  filter_type,
                                  my_filter,
                                  save_dir,
                                  number_of_threads,
                                  start_folder,
                                  stop_folder
                                  )
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
        data = {
            'Data location': self.load_dir_label.text(),
            'Start folder': self.start_folder_text_field.text(),
            'Stop folder': self.stop_folder_text_field.text(),
            'List of patients': self.list_of_patient_folders_text_field.text(),
            'Data Type': self.input_data_type_combo_box.currentText(),
            'Save directory': self.save_dir_label.text(),
            '# of cores': self.number_of_threads_combo_box.currentText(),
            'Save NII Image': self.nifti_image_text_field.text(),
            'Self Filter Type': self.filter_combo_box.currentText(),
            'Filter Dimension': self.filter_dimension_combo_box.currentText(),
            'Padding Type': self.padding_type_combo_box.currentText(),
            # Mean
            'Mean support': self.mean_filter_support_text_field.text(),
            # LoG
            'LoG sigma': self.log_filter_sigma_text_field.text(),
            'LoG cutoff': self.log_filter_cutoff_text_field.text(),
            # Laws
            'Laws response map': self.laws_filter_response_map_text_field.text(),
            'Laws rot inv': self.laws_filter_rot_inv_combo_box.currentText(),
            'Laws distance': self.laws_filter_distance_text_field.text(),
            'Laws pooling': self.laws_filter_pooling_combo_box.currentText(),
            'Laws energy map': self.laws_filter_energy_map_combo_box.currentText(),
            # Wavelet
            'Wavelet response map 2D': self.wavelet_filter_response_map_2d_combo_box.currentText(),
            'Wavelet response map 3D': self.wavelet_filter_response_map_3d_combo_box.currentText(),
            'Wavelet type': self.wavelet_filter_type_combo_box.currentText(),
            'Wavelet decomp lvl': self.wavelet_filter_decomposition_level_combo_box.currentText(),
            'Wavelet rot inv': self.wavelet_filter_rot_inv_combo_box.currentText()
        }

        with open('zrad/input/last_saved_filt_user_input.json', 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        try:
            with open('zrad/input/last_saved_filt_user_input.json', 'r') as file:
                data = json.load(file)
                self.load_dir_label.setText(data.get('Data location', ''))
                self.start_folder_text_field.setText(data.get('Start folder', ''))
                self.stop_folder_text_field.setText(data.get('Stop folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('List of patients', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('Data Type', ''))
                self.save_dir_label.setText(data.get('Save directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('# of cores', ''))
                self.nifti_image_text_field.setText(data.get('Save NII Image', ''))
                self.filter_combo_box.setCurrentText(data.get('Self Filter Type', ''))
                self.filter_dimension_combo_box.setCurrentText(data.get('Filter Dimension', ''))
                self.padding_type_combo_box.setCurrentText(data.get('Padding Type', ''))
                self.mean_filter_support_text_field.setText(data.get('Mean support', ''))
                self.log_filter_sigma_text_field.setText(data.get('LoG sigma', ''))
                self.log_filter_cutoff_text_field.setText(data.get('LoG cutoff', ''))
                self.laws_filter_response_map_text_field.setText(data.get('Laws response map', ''))
                self.laws_filter_rot_inv_combo_box.setCurrentText(data.get('Laws rot inv', ''))
                self.laws_filter_distance_text_field.setText(data.get('Laws distance', ''))
                self.laws_filter_pooling_combo_box.setCurrentText(data.get('Laws pooling', ''))
                self.laws_filter_energy_map_combo_box.setCurrentText(data.get('Laws energy map', ''))
                self.wavelet_filter_response_map_2d_combo_box.setCurrentText(data.get('Wavelet response map 2D', ''))
                self.wavelet_filter_response_map_3d_combo_box.setCurrentText(data.get('Wavelet response map 3D', ''))
                self.wavelet_filter_type_combo_box.setCurrentText(data.get('Wavelet type', ''))
                self.wavelet_filter_decomposition_level_combo_box.setCurrentText(data.get('Wavelet decomp lvl', ''))
                self.wavelet_filter_rot_inv_combo_box.setCurrentText(data.get('Wavelet rot inv', ''))

        except FileNotFoundError:
            print("No previous data found!")

    def init_tab(self):
        # Create a QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Path to load the files
        self.load_dir_button = CustomButton(
            'Load Directory',
            14, 30, 50, 200, 50, self,
            style=True
        )
        self.load_dir_label = CustomLabel('', 14, 300, 50, 1400, 50, self)
        self.load_dir_label.setAlignment(Qt.AlignCenter)
        self.load_dir_button.clicked.connect(lambda: self.open_directory(key=True))

        # Set used data type
        self.input_data_type_combo_box = CustomBox(
            14, 60, 300, 140, 50, self,
            item_list=[
                "Data Type:", "DICOM", "NIFTI"
                       ]
        )

        self.input_data_type_combo_box.currentTextChanged.connect(
            lambda:
                (self.nifti_image_label.show(),
                 self.nifti_image_text_field.show())
                if self.input_data_type_combo_box.currentText() == "NIFTI"
                else
                (self.nifti_image_label.hide(),
                 self.nifti_image_text_field.hide())
             )

        self.input_imaging_mod_combo_box = CustomBox(
            14, 320, 140, 170, 50, self,
            item_list=[
                "Imaging Mod.:", "CT", "MR", "PT"
            ]
        )

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
        # Set # of used cores
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

        # Set save directory
        self.save_dir_button = CustomButton(
            'Save Directory',
            14, 30, 220, 200, 50, self,
            style=True
        )

        self.save_dir_label = CustomLabel('', 14, 300, 220, 1400, 50, self)
        self.save_dir_label.setAlignment(Qt.AlignCenter)
        self.save_dir_button.clicked.connect(lambda: self.open_directory(key=False))

        # WHAT?
        self.nifti_image_label = CustomLabel(
            'NIFTI image file:',
            18, 320, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT.nii.gz",
            14, 510, 300, 350, 50, self
        )
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()

        # Set output_imaging_type
        self.filter_combo_box = CustomBox(
            14, 60, 380, 140, 50, self,
            item_list=[
                "Filter Type:", "Mean", "Laplacian of Gaussian", "Laws Kernels", "Wavelets"
            ]
        )
        self.filter_combo_box.currentTextChanged.connect(self.filter_combo_box_changed)

        self.padding_type_combo_box = CustomBox(
            14, 480, 380, 150, 50, self,
            item_list=[
                "Padding Type:", "constant", "nearest", "wrap", "reflect"
            ]
        )
        self.padding_type_combo_box.hide()

        self.mean_filter_support_label = CustomLabel(
            'Support:',
            18, 640, 380, 100, 50, self,
            style="color: white;"
        )
        self.mean_filter_support_text_field = CustomTextField(
            "E.g. 15",
            14, 740, 380, 75, 50, self
        )
        self.mean_filter_support_text_field.hide()
        self.mean_filter_support_label.hide()

        self.filter_dimension_combo_box = CustomBox(
            14, 320, 380, 140, 50, self,
            item_list=[
                "Dimension:", "2D", "3D"
            ]
        )
        self.filter_dimension_combo_box.hide()

        self.log_filter_sigma_label = CustomLabel(
            '\u03C3 (in mm):',
            18, 640, 380, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_sigma_text_field = CustomTextField(
            "E.g. 3",
            14, 760, 380, 75, 50, self
        )
        self.log_filter_sigma_label.hide()
        self.log_filter_sigma_text_field.hide()

        self.log_filter_cutoff_label = CustomLabel(
            'Cutoff (in \u03C3):',
            18, 845, 380, 200, 50, self,
            style="color: white;"
        )
        self.log_filter_cutoff_text_field = CustomTextField(
            "E.g. 4",
            14, 990, 380, 75, 50, self)
        self.log_filter_cutoff_label.hide()
        self.log_filter_cutoff_text_field.hide()

        self.laws_filter_response_map_label = CustomLabel(
            'Response Map:',
            18, 650, 380, 200, 50, self,
            style="color: white;"
        )
        self.laws_filter_response_map_text_field = CustomTextField("E.g. L5E5", 14, 830, 380, 100, 50, self)
        self.laws_filter_rot_inv_combo_box = CustomBox(
            14, 950, 380, 190, 50, self,
            item_list=[
                'Pseudo-rot. inv:', 'Enable', 'Disable'
            ]
        )
        self.laws_filter_distance_label = CustomLabel('Distance:', 18, 1160, 380, 200, 50, self, style="color: white;")
        self.laws_filter_distance_text_field = CustomTextField(
            "E.g. 5",
            14, 1270, 380, 75, 50, self
        )
        self.laws_filter_pooling_combo_box = CustomBox(
            14, 1370, 380, 120, 50, self,
            item_list=['Pooling:', 'max'])
        self.laws_filter_energy_map_combo_box = CustomBox(
            14, 1515, 380, 140, 50, self,
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
            14, 820, 380, 170, 50, self,
            item_list=[
                "Wavelet type:", "db3", "db2", "coif1", "haar"
            ]
        )
        self.wavelet_filter_type_combo_box.hide()
        self.wavelet_filter_response_map_combo_box = CustomBox(
            14, 645, 380, 155, 50, self,
            item_list=['Response Map']
        )
        self.wavelet_filter_response_map_combo_box.hide()
        self.wavelet_filter_response_map_2d_combo_box = CustomBox(
            14, 670, 380, 150, 50, self,
            item_list=[
                'Response Map:', 'LL', 'HL', 'LH', 'HH'
            ]
        )
        self.wavelet_filter_response_map_2d_combo_box.hide()
        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            14, 670, 380, 150, 50, self,
            item_list=['Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
                       ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_response_map_3d_combo_box = CustomBox(
            14, 670, 380, 150, 50, self,
            item_list=[
                 'Response Map:', 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', "HLH", "HHH"
             ]
        )
        self.wavelet_filter_response_map_3d_combo_box.hide()

        self.wavelet_filter_decomposition_level_combo_box = CustomBox(
            14, 1010, 380, 200, 50, self,
            item_list=[
                'Decomposition Lvl.:', '1', '2'
            ]
        )
        self.wavelet_filter_decomposition_level_combo_box.hide()

        self.wavelet_filter_rot_inv_combo_box = CustomBox(
            14, 1230, 380, 200, 50, self,
            item_list=[
                'Pseudo-rot. inv:', 'Enable', 'Disable'
            ]
        )
        self.wavelet_filter_rot_inv_combo_box.hide()

        self.run_button = CustomButton(
            'Run',
            20, 910, 590, 80, 50, self,
            style=False
        )
        self.run_button.clicked.connect(self.run_selected_option)

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
