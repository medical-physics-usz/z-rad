"""read data and save texture parameters in txt file"""
import logging
import os
from glob import glob
from pathlib import PurePath

import wx
from numpy import arange
from wx.adv import AboutBox
from wx.adv import AboutDialogInfo

from export import export_excel
from lymph_nodes import LymphNodes
from main_texture_ct import main_texture_ct
from main_texture_mr import main_texture_mr
from main_texture_pet import main_texture_pet
from myinfo import MyInfo
from panel_radiomics import panelRadiomics
from panel_resize import panelResize
from shape import Shape

VERSION = '7.6-dev'


def on_o_programie(evt):
    """info"""
    description = """"""
    licence = """"""
    info = AboutDialogInfo()
    info.SetName('Z-Rad')
    info.SetVersion(VERSION)
    info.SetDescription(description)
    info.SetCopyright('(C) 2017-2023 USZ Radiomics Team')
    info.SetLicence(licence)
    AboutBox(info)


class Radiomics(wx.Frame):
    """Main GUI class plus method on_calculate to start radiomics calculation
        Parent of class Panel"""

    def __init__(self, *a, **b):
        super(Radiomics, self).__init__(size=(1400, 900), pos=(50, 50), title='Z-Rad ' + VERSION, *a, **b)

        self.local = None
        self.sizer = None
        self.p = None
        self.panelHeight = None
        self.panelResize = None
        self.panelRadiomics = None
        self.nb = None
        self.defaultWindowsize = (1400, 1000)
        self.SetMinSize(self.defaultWindowsize)
        logging.basicConfig(format='%(name)-18s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("Main")
        self.logger.info('Initialize GUI and Main')
        self.logger.info("Logger Mode: " + logging.getLevelName(self.logger.getEffectiveLevel()))
        self.init_ui()
        self.Show()

    def init_ui(self):
        self.panelHeight = 24  # height of boxes in GUI, 20 for PC and 40 for lenovo laptop
        self.p = wx.Panel(self, size=self.defaultWindowsize)
        self.nb = wx.Notebook(self.p, size=self.defaultWindowsize)
        self.nb.panelHeight = self.panelHeight
        self.nb.on_calculate = self.on_calculate
        self.panelRadiomics = panelRadiomics(self.nb)  # radiomics panel
        self.panelResize = panelResize(self.nb)  # resize panel
        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panelRadiomics, "Radiomics", select=True)  # select True - first one

        menubar = wx.MenuBar()  # create menu bar
        plik_menu = wx.Menu()
        no = wx.MenuItem(plik_menu, wx.ID_NEW, '&New\tCtrl+N')  # new calculation
        plik_menu.Append(no)
        sv = wx.MenuItem(plik_menu, wx.ID_SAVE, '&Save\tCtrl+S')  # save settings
        plik_menu.Append(sv)
        za = wx.MenuItem(plik_menu, wx.ID_ANY, '&Quit\tCtrl+Q')
        plik_menu.Append(za)
        menubar.Append(plik_menu, 'File')
        program_menu = wx.Menu()
        program = wx.MenuItem(program_menu, wx.ID_ANY, 'About')
        program_menu.Append(program)
        menubar.Append(program_menu, 'About')

        self.SetMenuBar(menubar)

        config = open('config.txt', 'r')  # read the configuration file
        saved_config = []
        for i in config:
            saved_config.append(i)
        self.panelResize.fill(saved_config[:18])  # use the saved configuration
        self.panelRadiomics.fill(saved_config[18:])
        del saved_config
        config.close()

        self.CreateStatusBar(1)

        #  connect menu with methods
        self.Bind(wx.EVT_MENU, on_o_programie, program)
        self.Bind(wx.EVT_MENU, self.on_new, no)
        self.Bind(wx.EVT_MENU, self.on_save, sv)
        self.Bind(wx.EVT_MENU, self.on_quit, za)

        # initialize panel
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)
        self.panelRadiomics.Refresh()
        self.panelRadiomics.Refresh()

    def on_calculate(self, evt):
        """Initialize radiomics calculation"""

        path_save, save_as, structure, pix_nr, bin_size, path_image, n_pref, start, stop = self.panelRadiomics.read()
        self.logger.info("Start: Calculate Radiomics")
        # MyInfo('Test done!')
        '''path_save - save results in, path
        save_as - name of text file to save radiomics
        structure - analysed structures, later converted to list with ',' separation
        pix_nr - number of bin
        bin_size - fixed bin size
        path_image - path to images
        n_pref - prefix in the folder naming eg CTP_x
        start
        stop'''
        stop = int(stop) + 1
        start = int(start)

        self.local = False  # ATTENTION!: if you set True, be aware that you calculate Radiomics in 3D only.

        # file type
        if self.panelRadiomics.FindWindowById(111).GetValue():
            file_type = 'dicom'
            labels = ''
        else:
            file_type = 'nifti'  # nifti file type assumes that conversion to eg HU or SUV has already been performed
            # read the labels for ROIs in the nifti files
            labels = self.panelRadiomics.FindWindowById(1041).GetValue()
            if labels != '':
                labels = labels.split(',')
                for i in arange(0, len(labels)):
                    labels[i] = int(labels[i])

        # convert to a list        
        if structure == '':
            structure = 'none'
            self.panelRadiomics.FindWindowById(104).SetValue(structure)
        else:
            structure = structure.split(',')
            structure = [e.strip() for e in structure]

        # dimensionality
        if self.panelRadiomics.FindWindowById(1061).GetValue():
            dim = '2D'
        elif self.panelRadiomics.FindWindowById(10611).GetValue():
            dim = '2D_singleSlice'
        else:
            dim = '3D'

        # wavelet
        if self.panelRadiomics.FindWindowById(1071).GetValue():
            wv = True
        else:
            wv = False

        if n_pref != '':
            patient_list = [n_pref + '_' + str(i) for i in arange(start, stop)]  # subfolders that you want to analyze
        else:
            pat_range = [str(i) for i in arange(start, stop)]
            pat_dirs = glob(path_image + os.sep + "*[0-9]*")
            list_dir_candidates = [e.split(os.sep)[-1] for e in pat_dirs if
                                   e.split(os.sep)[-1].split("_")[0] in pat_range]
            patient_list = sorted(list_dir_candidates)

        # no. parallel jobs
        n_jobs = int(self.panelRadiomics.FindWindowById(170).GetValue())

        # to be adapted
        export_list = []
        crop_structure = {"crop": False, "ct_path": ""}

        # save parameters of calculation
        dict_parameters = {'path': [path_image],
                           "structure": [str(structure)],
                           "pixelNr": [pix_nr],
                           "bin_size": [bin_size],
                           "Dimension": [dim],
                           "wv": [wv]}

        # modality
        if self.panelRadiomics.FindWindowById(120).GetValue():  # CT
            outlier_corr = self.panelRadiomics.FindWindowById(127).GetValue()
            try:
                hu_min = int(self.panelRadiomics.FindWindowById(125).GetValue())
                hu_max = int(self.panelRadiomics.FindWindowById(126).GetValue())
            except ValueError:  # the input has t be a number
                hu_min = 'none'
                hu_max = 'none'
                self.panelRadiomics.FindWindowById(125).SetValue('none')
                self.panelRadiomics.FindWindowById(126).SetValue('none')

            dict_parameters["image_modality"] = ['CT']
            dict_parameters["HUmin"] = [hu_min]
            dict_parameters["HUmax"] = [hu_max]
            dict_parameters["outlier_corr"] = [outlier_corr]
            main_texture_ct(file_type, path_image, path_save, structure, labels, pix_nr, bin_size,
                            patient_list, save_as,
                            dim, hu_min, hu_max, outlier_corr, wv, self.local, crop_structure, n_jobs)

        elif self.panelRadiomics.FindWindowById(130).GetValue():  # PET
            suv = self.panelRadiomics.FindWindowById(131).GetValue()
            crop_arg = bool(self.panelRadiomics.FindWindowById(133).GetValue())  # if crop
            ct_hu_min = 'none'
            ct_hu_max = 'none'
            ct_path = ""

            if suv and file_type == 'nifti':
                MyInfo('Nifti files with SUV normalization were not tested.')
                raise SystemExit(0)

            if crop_arg and file_type == 'nifti':
                MyInfo('CT cropping and nifit format are not compatible. Use dicom data înstead.')
                raise SystemExit(0)
            elif crop_arg and file_type == 'dicom':
                self.logger.info("CropStructures " + str(crop_arg))
                try:
                    ct_hu_min = int(self.panelRadiomics.FindWindowById(135).GetValue())
                    ct_hu_max = int(self.panelRadiomics.FindWindowById(136).GetValue())
                except ValueError:  # the input has t be a number
                    ct_hu_min = 'none'
                    ct_hu_max = 'none'
                ct_path = self.panelRadiomics.FindWindowById(137).GetValue()  # CT path
                if ct_path == "":
                    print("Error: No CT Path provided!")
                    raise

            crop_structure = {"crop": crop_arg, "hu_min": ct_hu_min, "hu_max": ct_hu_max, "ct_path": ct_path}
            dict_parameters["image_modality"] = ['PET']
            dict_parameters['SUV'] = [suv]
            dict_parameters['CT corrected'] = [crop_arg]
            dict_parameters["HUmin"] = [ct_hu_min]
            dict_parameters["HUmax"] = [ct_hu_max]
            main_texture_pet(self.GetStatusBar(), file_type, path_image, path_save, structure, labels, pix_nr, bin_size,
                             patient_list, save_as,
                             dim, suv, wv, self.local, crop_structure, export_list, n_jobs)

        elif self.panelRadiomics.FindWindowById(150).GetValue():  # MR
            struct_norm1 = self.panelRadiomics.FindWindowById(151).GetValue()
            struct_norm2 = self.panelRadiomics.FindWindowById(152).GetValue()
            norm_roi_advanced = self.panelRadiomics.FindWindowById(153).GetValue()
            path_skull = self.panelRadiomics.FindWindowById(154).GetValue()
            path_template = self.panelRadiomics.FindWindowById(155).GetValue()
            norm_type = self.panelRadiomics.FindWindowById(156).GetValue()  # normalization method

            if norm_type == 'linear' and (struct_norm1 == '' or struct_norm2 == ''):
                MyInfo('To perform the linear interpolation provide the names of two ROIs')
                raise SystemExit(0)
            elif norm_type == 'linear' and file_type == 'nifti':
                MyInfo('MR linear normalization nifit format are not compatible. Use dicom data înstead.')
                raise SystemExit(0)
            elif norm_type == 'histogram matching' and norm_roi_advanced != 'ROI' and file_type == 'nifti':
                MyInfo('MR normalization using brain mask and nifit format are not compatible. Use dicom data înstead.')
                raise SystemExit(0)

            if norm_type == 'histogram matching' and path_template == '':
                MyInfo('To perform the histogram matching provide numpy array of standard ROI')
                raise SystemExit(0)

            dict_parameters["image_modality"] = ['MR']
            dict_parameters['normalization'] = PurePath(norm_type, struct_norm1, struct_norm2, norm_roi_advanced,
                                                        path_template)
            main_texture_mr(self.GetStatusBar(), file_type, path_image, path_save, structure, labels, pix_nr, bin_size,
                            patient_list, save_as, dim, struct_norm1, struct_norm2, norm_roi_advanced, path_skull,
                            norm_type,
                            path_template, wv, self.local, crop_structure, export_list, n_jobs)

        calc_shape = self.panelRadiomics.FindWindowById(1081).GetValue()
        name_shape_pts = self.panelRadiomics.FindWindowById(1083).GetValue()  # name of ROIs defined for shape
        if calc_shape:  # calculate shape
            if file_type == 'nifti':
                MyInfo('Shape parameters cannot be computed from nifti files')
                calc_shape = False
            else:
                name_shape_pt_list = name_shape_pts.split(',')
                name_shape_pt_list = [e.strip() for e in name_shape_pt_list]
                dict_parameters['shape structure'] = [str(name_shape_pt_list)]
                Shape(path_image, path_save, save_as, name_shape_pt_list, start, stop, n_jobs)

        if dim == "3D":
            export_excel(calc_shape, path_save, save_as, dict_parameters)

        # calculate results for LN
        if self.panelRadiomics.FindWindowById(1091).GetValue():
            # name of ROI defined as LN for shape, for example g_LN, searches for g_LN_X
            name_ln = self.panelRadiomics.FindWindowById(1093).GetValue()
            name_shape_pt = name_shape_pts  # name of ROI defined as PT for shape
            path_files_shape = path_image + os.sep + 'resized_1mm' + os.sep
            inp_mypath_results = path_save + os.sep + 'LN_' + '_' + str(start) + '_' + str(stop - 1) + '.txt'
            LymphNodes(name_ln, name_shape_pt, path_files_shape, inp_mypath_results, path_save, start, stop)

        MyInfo('Radiomics done')

    def on_new(self, evt):
        panel_radiomics_save = self.panelRadiomics.save()
        panel_resize_save = self.panelResize.save()
        size = self.panelRadiomics.GetSize()
        self.nb.Destroy()
        self.nb = wx.Notebook(self.p, size=self.defaultWindowsize)

        self.nb.panelHeight = self.panelHeight
        self.nb.on_calculate = self.on_calculate

        self.panelRadiomics = panelRadiomics(self.nb, size=size)
        self.panelRadiomics.fill(panel_radiomics_save)
        self.panelResize = panelResize(self.nb, size=size)
        self.panelResize.fill(panel_resize_save)
        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panelRadiomics, "Radiomics", select=True)

        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)

        self.panelRadiomics.Layout()
        self.panelResize.Layout()

        del panel_radiomics_save
        del panel_resize_save

    def on_quit(self, evt):
        config = open('config.txt', 'w')
        panel_resize_save = self.panelResize.save()
        panel_radiomics_save = self.panelRadiomics.save()
        for i in panel_resize_save:
            config.write('{}\n'.format(i))
        for i in panel_radiomics_save:
            config.write('{}\n'.format(i))
        config.close()
        self.Close()

    def on_save(self, evt):
        config = open('config.txt', 'w')
        panel_resize_save = self.panelResize.save()
        panel_radiomics_save = self.panelRadiomics.save()
        for i in panel_resize_save:
            config.write('{}\n'.format(i))
        for i in panel_radiomics_save:
            config.write('{}\n'.format(i))
        config.close()


# run the app
app = wx.App()
Radiomics(None)
app.MainLoop()
