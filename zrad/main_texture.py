"""read data and save texture parameters in txt file"""
import logging
import os

import wx
from numpy import arange
from wx.adv import AboutBox
from wx.adv import AboutDialogInfo

from exportExcel import ExportExcel
from lymph_nodes import LymphNodes
from main_texture_ct import main_texture_ct
from main_texture_ctp import main_texture_ctp
from main_texture_ivim import main_texture_ivim
from main_texture_mr import main_texture_mr
from main_texture_pet import main_texture_pet
from myinfo import MyInfo
from panel_radiomics import panelRadiomics
from panel_resize import panelResize
from shape import Shape


class Radiomics(wx.Frame):
    """Main GUI class plus method OnCalculate to start radiomics calculation
        Parent of class Panel"""

    def __init__(self, *a, **b):
        super(Radiomics, self).__init__(size=(1075, 725), pos=(100, 100), title='Z-Rad 7.2.1', *a, **b)

        self.defaultWindowsize = (1100, 725)
        self.SetMinSize(self.defaultWindowsize)
        logging.basicConfig(format='%(name)-18s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("Main")
        self.logger.info('Initialize GUI and Main')
        self.logger.info("Logger Mode: " + logging.getLevelName(self.logger.getEffectiveLevel()))
        self.InitUI()
        self.Show()

    def InitUI(self):
        self.panelHeight = 24  # height of boxes in GUI, 20 for PC and 40 for lenovo laptop
        self.p = wx.Panel(self, size=self.defaultWindowsize)
        self.nb = wx.Notebook(self.p, size=self.defaultWindowsize)
        self.nb.panelHeight = self.panelHeight
        self.nb.OnCalculate = self.OnCalculate
        self.panelRadiomics = panelRadiomics(self.nb)  # radiomics panel
        self.panelResize = panelResize(self.nb)  # resize panel
        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panelRadiomics, "Radiomics", select=True)  # select True - first one

        menubar = wx.MenuBar()  # create menu bar
        plikMenu = wx.Menu()
        no = wx.MenuItem(plikMenu, wx.ID_NEW, '&New\tCtrl+N')  # new calculation
        plikMenu.Append(no)
        sv = wx.MenuItem(plikMenu, wx.ID_SAVE, '&Save\tCtrl+S')  # save settings
        plikMenu.Append(sv)
        za = wx.MenuItem(plikMenu, wx.ID_ANY, '&Quit\tCtrl+Q')
        plikMenu.Append(za)
        menubar.Append(plikMenu, 'File')
        programMenu = wx.Menu()
        program = wx.MenuItem(programMenu, wx.ID_ANY, 'About')
        programMenu.Append(program)
        menubar.Append(programMenu, 'About')

        self.SetMenuBar(menubar)

        config = open('config.txt', 'r')  # read the configuration file
        l = []
        for i in config:
            l.append(i)
            # self.logger.debug("list of config " + i )
        self.panelResize.fill(l[:15])  # use the saved configuration
        self.panelRadiomics.fill(l[15:])
        del l
        config.close()

        self.CreateStatusBar(1)

        #  connect menu with methods
        self.Bind(wx.EVT_MENU, self.OnOProgramie, program)
        self.Bind(wx.EVT_MENU, self.OnNew, no)
        self.Bind(wx.EVT_MENU, self.OnSave, sv)
        self.Bind(wx.EVT_MENU, self.OnQuit, za)

        # initialize panel
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)
        self.panelRadiomics.Refresh()
        self.panelRadiomics.Refresh()

    def OnCalculate(self, evt):
        """Initialize radiomics calculation"""

        path_save, save_as, structure, pixNr, binSize, path_image, n_pref, start, stop = self.panelRadiomics.read()
        self.logger.info("Start: Calculate Radiomics")
        # self.logger.info( "Structures found", ', '.join(structure))
        MyInfo('Test done!')
        '''path_save - save results in, path
        save_as - name of text file to save radiomics
        structure - analysed structures, later converted to list with ',' separation
        pixNr - number of bin
        binSize - fixed bin size
        path_image - path to images
        n_pref - prefix in the folder naming eg CTP_x
        start
        stop'''
        stop = int(stop) + 1
        start = int(start)
        
        self.local = self.panelRadiomics.FindWindowById(200).GetValue()  # ATTENTION!: if you set True, be aware that you calculate Radiomics in 3D only. Final implementation to be added

        # convert to a list        
        if structure == '':
            structure = 'none'
            self.panelRadiomics.FindWindowById(104).SetValue(structure)
        else:
            structure = [e.strip() for e in structure]
                    
        multipleNames = self.panelRadiomics.FindWindowById(1041).GetValue()

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
            l_ImName = [n_pref + '_' + str(i) for i in arange(start, stop)]  # subfolders that you want to analyze
        else:
            l_ImName = [str(i) for i in arange(start, stop)]  # subfolders that you want to analyze

        # no. parallel jobs
        n_jobs = int(self.panelRadiomics.FindWindowById(170).GetValue())

        # to be adapted
        exportList = []
        cropStructure = {"crop": False, "ct_path": ""}
        
        # save parameters of calculation
        dict_parameters = {'path': path_image,
                           "structure": structure,
                            "pixelNr": pixNr,
                            "bin_size": binSize,
                            "Dimension": dim,
                            "wv": wv}

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

            dict_parameters["image_modality"] = 'CT'
            dict_parameters["HUmin"] = hu_min
            dict_parameters["HUmax"] = hu_max
            dict_parameters["outlier_corr"] = outlier_corr
            main_texture_ct(self.GetStatusBar(), path_image, path_save, structure, pixNr, binSize, l_ImName, save_as,
                            dim, hu_min, hu_max, outlier_corr, wv, self.local, multipleNames, cropStructure, exportList, n_jobs)

        elif self.panelRadiomics.FindWindowById(130).GetValue():  # PET
            SUV = self.panelRadiomics.FindWindowById(131).GetValue()
            cropArg = bool(self.panelRadiomics.FindWindowById(133).GetValue())  # if crop
            ct_hu_min = 'none'
            ct_hu_max = 'none'
            ct_path = ""

            if cropArg:
                self.logger.info("CropStructures " + str(cropArg))
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

            cropStructure = {"crop": cropArg, "hu_min": ct_hu_min, "hu_max": ct_hu_max, "ct_path": ct_path}
            dict_parameters["image_modality"] = 'PET'
            dict_parameters['SUV'] = SUV
            dict_parameters['CT corrected'] = cropArg
            dict_parameters["HUmin"] = ct_hu_min
            dict_parameters["HUmax"] = ct_hu_max
            main_texture_pet(self.GetStatusBar(), path_image, path_save, structure, pixNr, binSize, l_ImName, save_as,
                             dim, SUV, wv, self.local, cropStructure, exportList, n_jobs)

        elif self.panelRadiomics.FindWindowById(140).GetValue():  # CTP
            outlier_corr = self.panelRadiomics.FindWindowById(141).GetValue()
            dict_parameters["image_modality"] = 'CTP'
            dict_parameters["outlier_corr"] = outlier_corr
            main_texture_ctp(self.GetStatusBar(), path_image, path_save, structure, pixNr, binSize, l_ImName, save_as,
                             dim, outlier_corr, wv, self.local, cropStructure, exportList)

        elif self.panelRadiomics.FindWindowById(150).GetValue():  # MR
            struct_norm1 = self.panelRadiomics.FindWindowById(151).GetValue()
            struct_norm2 = self.panelRadiomics.FindWindowById(152).GetValue()
            dict_parameters["image_modality"] = 'MR'
            dict_parameters['normalization'] = struct_norm1 + ' ' + struct_norm2
            main_texture_mr(self.GetStatusBar(), path_image, path_save, structure, pixNr, binSize, l_ImName, save_as,
                            dim, struct_norm1, struct_norm2, wv, self.local, cropStructure, exportList, n_jobs)

        elif self.panelRadiomics.FindWindowById(160).GetValue():  # IVIM
            dict_parameters["image_modality"] = 'IVIM'
            main_texture_ivim(self.GetStatusBar(), path_image, path_save, structure, pixNr, binSize, l_ImName, save_as,
                              dim, wv, self.local, cropStructure, exportList)

        calc_shape = self.panelRadiomics.FindWindowById(1081).GetValue()
        name_shape_pts = self.panelRadiomics.FindWindowById(1083).GetValue()  # name of ROIs defined for shape
        if calc_shape:  # calculate shape
            name_shape_pt_list = name_shape_pts.split(',')
            name_shape_pt_list = [e.strip() for e in name_shape_pt_list]
            dict_parameters['shape structure'] = name_shape_pt_list
            Shape(path_image, path_save, save_as, name_shape_pt_list, start, stop, n_jobs)

        if dim == "3D":
            ExportExcel(calc_shape, path_save, save_as, dict_parameters)

        # calculate results for LN
        if self.panelRadiomics.FindWindowById(1091).GetValue():
            # name of ROI defined as LN for shape, for example g_LN, searches for g_LN_X
            name_ln = self.panelRadiomics.FindWindowById(1093).GetValue()
            name_shape_pt = name_shape_pts  # name of ROI defined as PT for shape
            path_files_shape = path_image + os.sep + 'resized_1mm' + os.sep
            inp_mypath_results = path_save + os.sep + 'LN_' + '_' + str(start) + '_' + str(stop - 1) + '.txt'
            LymphNodes(name_ln, name_shape_pt, path_files_shape, inp_mypath_results, path_save, start, stop)

        MyInfo('Radiomics done')

    def OnNew(self, evt):
        l = self.panelRadiomics.save()
        lr = self.panelResize.save()
        size = self.panelRadiomics.GetSize()
        self.nb.Destroy()
        self.nb = wx.Notebook(self.p, size=self.defaultWindowsize)

        self.nb.panelHeight = self.panelHeight
        self.nb.OnCalculate = self.OnCalculate

        self.panelRadiomics = panelRadiomics(self.nb, size=size)
        self.panelRadiomics.fill(l)
        self.panelResize = panelResize(self.nb, size=size)
        self.panelResize.fill(lr)
        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panelRadiomics, "Radiomics", select=True)

        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)

        self.panelRadiomics.Layout()
        self.panelResize.Layout()

        del l
        del lr

    def OnQuit(self, evt):
        config = open('config.txt', 'w')
        lr = self.panelResize.save()
        l = self.panelRadiomics.save()
        for i in lr:
            config.write('{}\n'.format(i))
        for i in l:
            config.write('{}\n'.format(i))
        config.close()
        self.Close()

    def OnSave(self, evt):
        config = open('config.txt', 'w')
        lr = self.panelResize.save()
        l = self.panelRadiomics.save()
        for i in lr:
            config.write('{}\n'.format(i))
        for i in l:
            config.write('{}\n'.format(i))
        config.close()

    def OnOProgramie(self, evt):
        """info"""
        description = """"""
        licence = """"""
        info = AboutDialogInfo()
        info.SetName('Z-Rad')
        info.SetVersion('7.2.1')
        info.SetDescription(description)
        info.SetCopyright('(C) 2017-2021 USZ Radiomics Team')
        info.SetLicence(licence)
        AboutBox(info)


# run the app
app = wx.App()
Radiomics(None)
app.MainLoop()
