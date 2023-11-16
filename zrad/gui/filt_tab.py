from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFileDialog)
from PyQt5.QtCore import Qt
import multiprocessing
import json

from gui.toolbox_gui import (CustomButton, CustomLabel, CustomBox, CustomTextField)
from logic.filtering import Filtering


class FilteringTab(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = None
        self.is_wavelet_response_map_connected = False
        #self.initUI()

    def run_selected_option(self):
        load_dir = self.LoadDirLabel.text()
        folder_prefix = self.PrefixTextField.text()
        start_folder = self.StartTextField.text()
        stop_folder = self.StopTextField.text()
        save_dir = self.SaveDirLabel.text()
        list_of_patient_folders = [int(pat) for pat in str(self.ListOfPatientsTextField.text()).split(",")]
        nifti_image = self.NiiImageFileTextField.text()
        input_data_type = self.DataTypeComboBox.currentText()
        no_of_threads = self.NoOfThreadsComboBox.currentText().split(" ")[0]
        filter_type = self.FilterComboBox.currentText()
        filter_padding_type = self.PaddingTypeComboBox.currentText()
        support = self.SupportTextField.text()
        filter_dimension = self.FilterDimensionComboBox.currentText()
        LoG_sigma = self.SigmaMmTextField.text()
        LoG_cutoff = self.CutoffTextField.text()
        wavelet_response_map = None
        if filter_dimension == '2D' and filter_type == 'Wavelets':
            wavelet_response_map = self.ResponseMapWavelet2DComboBox.currentText()
        elif filter_dimension == '3D' and filter_type == 'Wavelets':
            wavelet_response_map = self.ResponseMapWavelet3DComboBox.currentText()
        wavelet_type = self.WaveletTypeComboBox.currentText()
        wavelet_decomposition_lvl = None
        if filter_type == 'Wavelets':
            wavelet_decomposition_lvl = int(self.WaveletDecompositionLevelComboBox.currentText())
        wavelet_pseudo_rot_inv = None
        if filter_type == 'Wavelets' and self.WaveletRotInvComboBox.currentText() == 'Enable':
            wavelet_pseudo_rot_inv = True
        elif filter_type == 'Wavelets' and self.WaveletRotInvComboBox.currentText() == 'Disable':
            wavelet_pseudo_rot_inv = False

        print({'Data location': load_dir, 'Folder folder_prefix': folder_prefix, 'Start folder': start_folder, 'Stop folder': stop_folder,
               'Save Directory': save_dir, 'List of patients': list_of_patient_folders,
                 'NIFTI image': nifti_image,
               'Data type': input_data_type, '# of cores': no_of_threads, 'Selected Filter': filter_type,
               'Padding Type': filter_padding_type, 'Mean filter_type mean_filter_support': support, 'Filter dim': filter_dimension,
               'LoG sigma': LoG_sigma, 'LoG cutoff': LoG_cutoff, 'Wavelet response map': wavelet_response_map,
               'Wavelet type': wavelet_type, 'Decomposition lvl': wavelet_decomposition_lvl,
               'Pseudo-rot inv.': wavelet_pseudo_rot_inv
               })


        Filtering(load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders, input_data_type,
                  nifti_image, filter_type, filter_dimension, filter_padding_type, support, LoG_sigma, LoG_cutoff,
                  wavelet_response_map, wavelet_type, wavelet_decomposition_lvl, wavelet_pseudo_rot_inv, save_dir, no_of_threads)


    def open_directory(self, key):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if directory and key == True:
            self.LoadDirLabel.setText(directory)
        elif directory and key == False:
            self.SaveDirLabel.setText(directory)

    def save_input_data(self):
        data = {
            'Data location': self.LoadDirLabel.text(),
            'Folder folder_prefix': self.PrefixTextField.text(),
            'Start folder': self.StartTextField.text(),
            'Stop folder': self.StopTextField.text(),
            'List of patients': self.ListOfPatientsTextField.text(),
            'Data Type': self.DataTypeComboBox.currentText(),
            'Save directory': self.SaveDirLabel.text(),
            '# of cores': self.NoOfThreadsComboBox.currentText(),
            'Save NII Image': self.NiiImageFileTextField.text(),
            'Self Filter Type': self.FilterComboBox.currentText(),
            'Filter Dimension': self.FilterDimensionComboBox.currentText(),
            'Padding Type': self.PaddingTypeComboBox.currentText(),
            'Mean filter_type mean_filter_support': self.SupportTextField.text()

        }
        with open('input/last_saved_filt_user_input.json', 'w') as file:
            json.dump(data, file)

    def load_input_data(self):
        try:
            with open('input/last_saved_filt_user_input.json', 'r') as file:
                data = json.load(file)
                self.LoadDirLabel.setText(data.get('Data location', ''))
                self.PrefixTextField.setText(data.get('Folder folder_prefix', ''))
                self.StartTextField.setText(data.get('Start folder', ''))
                self.StopTextField.setText(data.get('Stop folder', ''))
                self.ListOfPatientsTextField.setText(data.get('List of patients', ''))
                self.DataTypeComboBox.setCurrentText(data.get('Data Type', ''))
                self.SaveDirLabel.setText(data.get('Save directory', ''))
                self.NoOfThreadsComboBox.setCurrentText(data.get('# of cores', ''))
                self.NiiImageFileTextField.setText(data.get('Save NII Image', ''))
                self.FilterComboBox.setCurrentText(data.get('Self Filter Type', ''))
                self.FilterDimensionComboBox.setCurrentText(data.get('Filter Dimension', ''))
                self.PaddingTypeComboBox.setCurrentText(data.get('Padding Type', ''))
                self.SupportTextField.setText(data.get('Mean filter_type mean_filter_support', ''))

        except FileNotFoundError:
            print("No previous data found!")

    def init_tab(self):
        # Create a QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Path to load the files
        self.LoadDirButton = CustomButton('Load Directory', 14, 30, 50, 200, 50, self,
                            style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;")
        self.LoadDirLabel = CustomLabel('', 14, 300, 50, 1400, 50, self)
        self.LoadDirLabel.setAlignment(Qt.AlignCenter)

        self.LoadDirButton.clicked.connect(lambda: self.open_directory(key=True))

        # Set used data type
        self.DataTypeComboBox = CustomBox(14, 60, 300, 140, 50, self, item_list=["Data Type:", "DICOM", "NIFTI"])

        self.DataTypeComboBox.currentTextChanged.connect(self.on_file_type_combo_box_changed)

        # Set folder_prefix
        self.PrefixLabel = CustomLabel('Prefix:', 18, 320, 140, 150, 50, self, style="color: white;")
        self.PrefixTextField = CustomTextField("Enter...", 14, 400, 140, 100, 50, self)

        # Set start_folder folder
        self.StartLabel = CustomLabel('Start:', 18, 520, 140, 150, 50, self, style="color: white;")
        self.StartTextField = CustomTextField("Enter...", 14, 590, 140, 100, 50, self)

        # Set stop_folder folder
        self.StopLabel = CustomLabel('Stop:', 18, 710, 140, 150, 50, self, style="color: white;")
        self.StopTextField = CustomTextField("Enter...", 14, 775, 140, 100, 50, self)

        # Set a list of studied patients
        self.ListOfPatientsLabel = CustomLabel('List of Patients:', 18, 900, 140, 200, 50, self, style="color: white;")
        self.ListOfPatientsTextField = CustomTextField("E.g. 1, 5, 10, 34...", 14, 1080, 140, 350, 50, self)
        # Set # of used cores
        no_of_threads = ['No. of Threads:']
        for core in range(multiprocessing.cpu_count()):
            if core == 0:
                no_of_threads.append(str(core + 1) + " thread")
            else:
                no_of_threads.append(str(core + 1) + " threads")
        self.NoOfThreadsComboBox = CustomBox(14, 1450, 140, 210, 50, self, item_list=no_of_threads)

        # Set save directory
        self.SaveDirButton = CustomButton('Save Directory', 14, 30, 220, 200, 50, self,
                                    style="background-color: #4CAF50; color: white; border: none; border-radius: 25px;")
        self.SaveDirLabel = CustomLabel('', 14, 300, 220, 1400, 50, self)
        self.SaveDirLabel.setAlignment(Qt.AlignCenter)
        self.SaveDirButton.clicked.connect(lambda: self.open_directory(key=False))

        # WHAT?
        self.NiiImageFileLabel = CustomLabel('NIFTI image file:', 18, 320, 300, 200, 50, self, style="color: white;")
        self.NiiImageFileTextField = CustomTextField("E.g. imageCT.nii.gz", 14, 1080-890+320, 300, 350, 50, self)
        self.NiiImageFileLabel.hide()
        self.NiiImageFileTextField.hide()

        # Set output_imaging_type
        self.FilterComboBox = CustomBox(14, 60, 380, 210, 50,
                                             self, item_list=["Filter:", "Mean", "Laplacian of Gaussian", "Laws Kernels","Wavelets"])

        self.FilterComboBox.currentTextChanged.connect(self.filter_combo_box_changed)

        self.PaddingTypeComboBox = CustomBox(14, 500, 380, 150, 50,
                                             self, item_list=["Padding Type:", "constant", "nearest", "wrap", "reflect"])
        self.PaddingTypeComboBox.hide()
        self.SupportLabel = CustomLabel('Support:', 18, 660, 380, 100, 50, self, style="color: white;")
        self.SupportTextField = CustomTextField("E.g. 15", 14, 760, 380, 75, 50, self)
        self.SupportTextField.hide()
        self.SupportLabel.hide()

        self.FilterDimensionComboBox = CustomBox(14, 320, 380, 170, 50,
                                             self, item_list=["Dimension:", "2D", "3D"])
        self.FilterDimensionComboBox.hide()

        self.SigmaMmLabel = CustomLabel('\u03C3 (in mm):', 18, 550 + 110, 380, 200, 50, self, style="color: white;")
        self.SigmaMmTextField = CustomTextField("E.g. 3", 14, 730 + 50, 380, 75, 50, self)
        self.SigmaMmLabel.hide()
        self.SigmaMmTextField.hide()
        self.CutoffLabel = CustomLabel('Cutoff (in \u03C3):', 18, 550 + 120+ 200, 380, 200, 50, self, style="color: white;")
        self.CutoffTextField = CustomTextField("E.g. 4", 14, 730 + 180+40+60, 380, 75, 50, self)
        self.CutoffLabel.hide()
        self.CutoffTextField.hide()

        self.PaddingConstantLabel = CustomLabel('Padding constant:', 18, 550 + 180 + 200 +110, 380, 250, 50, self, style="color: white;")
        self.PaddingConstantTextField = CustomTextField("E.g. 0.0", 14, 730 + 180 + 40 + 290, 380, 75, 50, self)
        self.PaddingConstantLabel.hide()
        self.PaddingConstantTextField.hide()

        self.ResolutionMmLabel = CustomLabel('Resolution (in mm):', 18, 550 + 180 + 70 + 240+290, 380, 250, 50, self,
                                                style="color: white;")
        self.ResolutionMmTextField = CustomTextField("E.g. 2", 14, 730 + 180 + 40 + 290+310, 380, 75, 50, self)
        self.ResolutionMmLabel.hide()
        self.ResolutionMmTextField.hide()

        self.ResopseMapLawsLabel = CustomLabel('Response Map:', 18, 670, 380, 200, 50, self, style="color: white;")
        self.ResopseMapLawsTextField = CustomTextField("E.g. L5E5", 14, 850, 380, 100, 50, self)
        self.LawsRotInvComboBox = CustomBox(14, 970, 380, 200, 50,
                                               self, item_list=['Pseudo-rot. inv:', 'Enable', 'Disable'])
        self.DistanceLawsLabel = CustomLabel('Distance:', 18, 1180, 380, 200, 50, self, style="color: white;")
        self.DustanceLawsTextField = CustomTextField("E.g. 5", 14, 1290, 380, 75, 50, self)
        self.PoolingLawsComboBox = CustomBox(14, 1370, 380, 200, 50,
                                            self, item_list=['Pooling:'])
        self.ResopseMapLawsLabel.hide()
        self.ResopseMapLawsTextField.hide()
        self.LawsRotInvComboBox.hide()
        self.DistanceLawsLabel.hide()
        self.DustanceLawsTextField.hide()
        self.PoolingLawsComboBox.hide()
        self.WaveletTypeComboBox = CustomBox(14, 320 + 350 + 170, 380, 200, 50,
                                        self, item_list=["Wavelet type:", "db3", "db2", "coif1", "haar"])
        self.WaveletTypeComboBox.hide()
        self.ResponseMapWaveletNoneComboBox = CustomBox(14, 320 + 350, 380, 150, 50,
                                                        self, item_list=['Response Map'])
        self.ResponseMapWaveletNoneComboBox.hide()
        self.ResponseMapWavelet2DComboBox = CustomBox(14, 320 + 350, 380, 150, 50,
                                                      self, item_list=['Response Map:', 'LL', 'HL', 'LH', 'HH'])
        self.ResponseMapWavelet2DComboBox.hide()
        self.ResponseMapWavelet3DComboBox = CustomBox(14, 320 + 350, 380, 150, 50,
                                                      self,
                                                      item_list=['Response Map:', 'LLL', 'LLH', 'LHL',
                                                                 'HLL', 'LHH', 'HHL', "HLH", "HHH"])
        self.ResponseMapWavelet3DComboBox.hide()

        self.ResponseMapWavelet3DComboBox = CustomBox(14, 320 + 350, 380, 150, 50,
                                                      self,
                                                      item_list=['Response Map:', 'LLL', 'LLH', 'LHL',
                                                                 'HLL', 'LHH', 'HHL', "HLH", "HHH"])
        self.ResponseMapWavelet3DComboBox.hide()

        self.WaveletDecompositionLevelComboBox = CustomBox(14, 320 + 350+380, 380, 200, 50,
                                                      self, item_list=['Decomposition Lvl:', '1', '2'])
        self.WaveletDecompositionLevelComboBox.hide()

        self.WaveletRotInvComboBox = CustomBox(14, 320 + 350 + 600, 380, 200, 50,
                                               self, item_list=['Pseudo-rot. inv:', 'Enable', 'Disable'])
        self.WaveletRotInvComboBox.hide()

        self.RunButton = CustomButton('Run', 20, 910, 590, 80, 50, self, style=False)
        self.RunButton.clicked.connect(self.run_selected_option)

    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the combo box's value is changed
        if text == 'DICOM':
            self.NiiImageFileLabel.hide()
            self.NiiImageFileTextField.hide()
        elif text == 'NIFTI':
            self.NiiImageFileLabel.show()
            self.NiiImageFileTextField.show()

        else:
            self.NiiImageFileLabel.hide()
            self.NiiImageFileTextField.hide()

    def filter_combo_box_changed(self, text):
        if text == 'Mean':
            if self.is_wavelet_response_map_connected:
                self.FilterDimensionComboBox.currentTextChanged.disconnect()
                self.is_wavelet_response_map_connected = False
            self.PaddingTypeComboBox.show()
            self.SupportTextField.show()
            self.SupportLabel.show()
            self.FilterDimensionComboBox.show()
            self.SigmaMmLabel.hide()
            self.SigmaMmTextField.hide()
            self.CutoffLabel.hide()
            self.CutoffTextField.hide()
            self.PaddingConstantLabel.hide()
            self.PaddingConstantTextField.hide()
            self.ResolutionMmLabel.hide()
            self.ResolutionMmTextField.hide()
            self.WaveletTypeComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.WaveletDecompositionLevelComboBox.hide()
            self.WaveletRotInvComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.ResopseMapLawsLabel.hide()
            self.ResopseMapLawsTextField.hide()
            self.LawsRotInvComboBox.hide()
            self.DistanceLawsLabel.hide()
            self.DustanceLawsTextField.hide()
            self.PoolingLawsComboBox.hide()

        elif text == 'Laplacian of Gaussian':
            if self.is_wavelet_response_map_connected:
                self.FilterDimensionComboBox.currentTextChanged.disconnect()
                self.is_wavelet_response_map_connected = False
            self.PaddingTypeComboBox.show()
            self.SupportTextField.hide()
            self.SupportLabel.hide()
            self.FilterDimensionComboBox.show()
            self.SigmaMmLabel.show()
            self.SigmaMmTextField.show()
            self.CutoffLabel.show()
            self.CutoffTextField.show()
            self.PaddingConstantLabel.hide()
            self.PaddingConstantTextField.hide()
            self.ResolutionMmLabel.hide()
            self.ResolutionMmTextField.hide()
            self.WaveletTypeComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.WaveletDecompositionLevelComboBox.hide()
            self.WaveletRotInvComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.ResopseMapLawsLabel.hide()
            self.ResopseMapLawsTextField.hide()
            self.LawsRotInvComboBox.hide()
            self.DistanceLawsLabel.hide()
            self.DustanceLawsTextField.hide()
            self.PoolingLawsComboBox.hide()

        elif text == "Laws Kernels":
            if self.is_wavelet_response_map_connected:
                self.FilterDimensionComboBox.currentTextChanged.disconnect()
                self.is_wavelet_response_map_connected = False
            self.PaddingTypeComboBox.show()
            self.SupportTextField.hide()
            self.SupportLabel.hide()
            self.FilterDimensionComboBox.show()
            self.SigmaMmLabel.hide()
            self.SigmaMmTextField.hide()
            self.CutoffLabel.hide()
            self.CutoffTextField.hide()
            self.PaddingConstantLabel.hide()
            self.PaddingConstantTextField.hide()
            self.ResolutionMmLabel.hide()
            self.ResolutionMmTextField.hide()
            self.WaveletTypeComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.WaveletDecompositionLevelComboBox.hide()
            self.WaveletRotInvComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()

            self.ResopseMapLawsLabel.show()
            self.ResopseMapLawsTextField.show()
            self.LawsRotInvComboBox.show()
            self.DistanceLawsLabel.show()
            self.DustanceLawsTextField.show()
            self.PoolingLawsComboBox.show()

        elif text == "Wavelets":
            self.PaddingTypeComboBox.show()
            self.SupportTextField.hide()
            self.SupportLabel.hide()
            self.FilterDimensionComboBox.show()
            self.SigmaMmLabel.hide()
            self.SigmaMmTextField.hide()
            self.CutoffLabel.hide()
            self.CutoffTextField.hide()
            self.PaddingConstantLabel.hide()
            self.PaddingConstantTextField.hide()
            self.ResolutionMmLabel.hide()
            self.ResolutionMmTextField.hide()
            self.WaveletTypeComboBox.show()
            if self.FilterDimensionComboBox.currentText() == 'Dimension:':
                self.ResponseMapWaveletNoneComboBox.show()
                self.ResponseMapWavelet3DComboBox.hide()
                self.ResponseMapWavelet2DComboBox.hide()
            elif self.FilterDimensionComboBox.currentText() == '3D':
                self.ResponseMapWaveletNoneComboBox.hide()
                self.ResponseMapWavelet3DComboBox.show()
                self.ResponseMapWavelet2DComboBox.hide()
            elif self.FilterDimensionComboBox.currentText() == '2D':
                self.ResponseMapWaveletNoneComboBox.hide()
                self.ResponseMapWavelet3DComboBox.hide()
                self.ResponseMapWavelet2DComboBox.show()

            self.FilterDimensionComboBox.currentTextChanged.connect(self.wavelet_response_map)
            self.is_wavelet_response_map_connected = True
            self.WaveletDecompositionLevelComboBox.show()
            self.WaveletRotInvComboBox.show()
            self.ResopseMapLawsLabel.hide()
            self.ResopseMapLawsTextField.hide()
            self.LawsRotInvComboBox.hide()
            self.DistanceLawsLabel.hide()
            self.DustanceLawsTextField.hide()
            self.PoolingLawsComboBox.hide()

        else:
            if self.is_wavelet_response_map_connected:
                self.FilterDimensionComboBox.currentTextChanged.disconnect()
                self.is_wavelet_response_map_connected = False
            self.PaddingTypeComboBox.hide()
            self.SupportTextField.hide()
            self.SupportLabel.hide()
            self.FilterDimensionComboBox.hide()
            self.SigmaMmLabel.hide()
            self.SigmaMmTextField.hide()
            self.CutoffLabel.hide()
            self.CutoffTextField.hide()
            self.PaddingConstantLabel.hide()
            self.PaddingConstantTextField.hide()
            self.ResolutionMmLabel.hide()
            self.ResolutionMmTextField.hide()
            self.WaveletTypeComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.WaveletDecompositionLevelComboBox.hide()
            self.WaveletRotInvComboBox.hide()
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
            self.ResopseMapLawsLabel.hide()
            self.ResopseMapLawsTextField.hide()
            self.LawsRotInvComboBox.hide()
            self.DistanceLawsLabel.hide()
            self.DustanceLawsTextField.hide()
            self.PoolingLawsComboBox.hide()

    def wavelet_response_map(self, text):
        if text == '2D':
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.show()
            self.ResponseMapWaveletNoneComboBox.hide()
        elif text == '3D':
            self.ResponseMapWavelet3DComboBox.show()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.hide()
        else:
            self.ResponseMapWavelet3DComboBox.hide()
            self.ResponseMapWavelet2DComboBox.hide()
            self.ResponseMapWaveletNoneComboBox.show()
