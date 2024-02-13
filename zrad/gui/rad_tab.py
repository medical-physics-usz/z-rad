from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFileDialog)
from PyQt5.QtCore import Qt
import multiprocessing
import sys
import json

from zrad.gui.toolbox_gui import (CustomButton, CustomLabel, CustomBox, CustomTextField, CustomCheckBox)
class RadiomicsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = None
        self.LoadDirButton = None
        self.LoadDirLabel = None
        self.fileTypeLabel = None
        self.DataTypeComboBox = None
        self.PrefixLabel = None
        self.PrefixTextField = None
        self.StartLabel = None
        self.StartTextField = None
        self.StopLabel = None
        self.StopTextField = None
        self.ListOfPatientsLabel = None
        self.ListOfPatientsTextField = None
        self.threadNoLabel = None
        self.NoOfThreadsComboBox = None
        self.SaveDirButton = None
        self.SaveDirLabel = None
        self.StructureNamesLabel = None
        self.StructureNamesTextField = None
        self.NiiImageFileLabel = None
        self.NiiImageFileTextField = None
        self.imTypeLabel = None
        self.ImagingTypeComboBox = None
        self.IntensityRangeLabel = None
        self.IntensityRangeTextField = None
        self.ImageInterpolationTypeComboBox = None
        self.ResizeResolutionLabel = None
        self.ResizeResolutionTextField = None
        self.interpTypeLabel = None
        self.ImageInterpolationTypeComboBox = None
        self.resizeDimTypeLabel = None
        self.ResizeDimComboBox = None
        self.RunButton = None
        self.is_wavelet_response_map_connected = False

    def run_selected_option(self):
        if self.LoadDirLabel.text() == '':
            print("Load directory not selected")
            sys.exit()
        else:
            load_dir = self.LoadDirLabel.text()
        folder_prefix = self.PrefixTextField.text()
        start_folder = self.StartTextField.text()
        stop_folder = self.StopTextField.text()
        save_dir = self.SaveDirLabel.text()
        if save_dir == '':
            print("Save directory not selected")
            sys.exit()
        list_of_patient_folders = []
        if self.ListOfPatientsTextField.text() !='':
            list_of_patient_folders = [int(pat) for pat in str(self.ListOfPatientsTextField.text()).split(",")]
        if start_folder == '' and stop_folder == '' and list_of_patient_folders == []:
            print('Studied patients are not specified')
            sys.exit()
        elif (start_folder != '' and stop_folder == '' and list_of_patient_folders == []) or (start_folder == '' and stop_folder != '' and list_of_patient_folders == []):
            print('Studied patients are not completely specified')
            sys.exit()
        number_of_threads = self.NoOfThreadsComboBox.currentText().split(" ")[0]
        print(number_of_threads == 'No.')
        if number_of_threads =='No.':
            print('No. of cores was not specified, default value of 1 core is used')
            number_of_threads = 1
        input_data_type = self.DataTypeComboBox.currentText()
        if input_data_type == "Data Type:":
            print('Input Data Type was not specified')
            sys.exit()
        dicom_structures = [ROI.strip() for ROI in self.StructureNamesTextField.text().split(",")]
        nifti_image = self.NiiImageFileTextField.text()
        nifti_structures = [ROI.strip() for ROI in self.NiiStructureFilesTextField.text().split(",")]
        if input_data_type == 'DICOM' and dicom_structures == []:
            print('DICOM structures are not specified')
        elif (input_data_type == 'NIFTI' and nifti_image == '') or (
                input_data_type == 'NIFTI' and nifti_structures == []):
            print('NIFTI structures or image are not specified')

        print({'Data location': load_dir, 'Folder folder_prefix': folder_prefix, 'Start folder': start_folder, 'Stop folder': stop_folder})

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

        self.StructureNamesLabel = CustomLabel('Studied str.:', 18, 370, 300, 200, 50, self, style="color: white;")
        self.StructureNamesTextField = CustomTextField("E.g. CTV, liver...", 14, 510, 300, 475, 50, self)
        self.StructureNamesLabel.hide()
        self.StructureNamesTextField.hide()

        self.NiiStructureFilesLabel = CustomLabel('NIFTI Str. Files:', 18,
                                                  370, 300, 200, 50, self, style="color: white;")
        self.NiiStructureFilesTextField = CustomTextField("E.g. CTV, liver...", 14,
                                                          540, 300, 250, 50, self)
        self.NiiStructureFilesLabel.hide()
        self.NiiStructureFilesTextField.hide()

        # WHAT?
        self.NiiImageFileLabel = CustomLabel('NIFTI Image File:', 18, 800, 300, 200, 50, self, style="color: white;")
        self.NiiImageFileTextField = CustomTextField("E.g. imageCT.nii.gz", 14, 990, 300, 220, 50, self)
        self.NiiImageFileLabel.hide()
        self.NiiImageFileTextField.hide()

        self.OutlierDetectionCheckBox = CustomCheckBox('Outlier Detection', 18, 375, 460, 250, 50, self)

        self.SigmaLabel = CustomLabel('Confidence Interval (in \u03C3):', 18, 640, 460, 350, 50, self,
                                      style="color: white;")
        self.SigmaTextField = CustomTextField("E.g. 3", 14, 930, 460, 100, 50, self)
        self.SigmaLabel.hide()
        self.SigmaTextField.hide()
        self.OutlierDetectionCheckBox.stateChanged.connect(
            lambda: (self.SigmaLabel.show(), self.SigmaTextField.show()) if self.OutlierDetectionCheckBox.isChecked()
            else (self.SigmaLabel.hide(), self.SigmaTextField.hide()))

        self.IntensityRangeLabel = CustomLabel('Intensity ranges:', 18,
                                               635, 375, 200, 50, self, style="color: white;")
        self.IntensityRangeTextField = CustomTextField("E.g. CTV: 0, 100; liver: 500, 1500", 14,
                                                       820, 375, 350, 50, self)
        self.IntensityRangeLabel.hide()
        self.IntensityRangeTextField.hide()

        self.IntensityRangeCheckBox = CustomCheckBox('Intensity Range', 18, 375, 380, 200, 50, self)
        self.IntensityRangeCheckBox.stateChanged.connect(
            lambda: (self.IntensityRangeLabel.show(), self.IntensityRangeTextField.show())
            if self.IntensityRangeCheckBox.isChecked()
            else (self.IntensityRangeLabel.hide(), self.IntensityRangeTextField.hide())
        )

        self.DiscretizationComboBox = CustomBox(14, 375, 540, 150, 50, self, item_list=["Discretization:", "Disable", "Bin number", "Bin size"])
        self.BinNumberTextField = CustomTextField("E.g. 5", 14, 375+170, 540, 100, 50, self)
        self.BinSizeTextField = CustomTextField("E.g. 50", 14, 375 + 170, 540, 100, 50, self)
        self.BinNumberTextField.hide()
        self.BinSizeTextField.hide()
        self.DiscretizationComboBox.currentTextChanged.connect(self.changed_discretization)

        self.RunButton = CustomButton('Run', 20, 910, 620, 80, 50, self, style=False)
        self.RunButton.clicked.connect(self.run_selected_option)
    def on_file_type_combo_box_changed(self, text):
        # This slot will be called whenever the combo box's value is changed
        if text == 'DICOM':
            self.NiiImageFileLabel.hide()
            self.NiiImageFileTextField.hide()
            self.StructureNamesLabel.show()
            self.StructureNamesTextField.show()
            self.NiiStructureFilesLabel.hide()
            self.NiiStructureFilesTextField.hide()
        elif text == 'NIFTI':
            self.NiiImageFileLabel.show()
            self.NiiImageFileTextField.show()
            self.StructureNamesLabel.hide()
            self.StructureNamesTextField.hide()
            self.NiiStructureFilesLabel.show()
            self.NiiStructureFilesTextField.show()

        else:
            self.StructureNamesLabel.hide()
            self.StructureNamesTextField.hide()
            self.NiiImageFileLabel.hide()
            self.NiiImageFileTextField.hide()
            self.NiiStructureFilesLabel.hide()
            self.NiiStructureFilesTextField.hide()

    def changed_discretization(self, text):
        if text == 'Bin number':
            self.BinNumberTextField.show()
            self.BinSizeTextField.hide()
        elif text == 'Bin size':
            self.BinNumberTextField.hide()
            self.BinSizeTextField.show()
        else:
            self.BinNumberTextField.hide()
            self.BinSizeTextField.hide()