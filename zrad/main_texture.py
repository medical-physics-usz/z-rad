"""read data and save texture parameters in txt file"""
import logging
from numpy import arange
import wx
from wx.adv import AboutBox
from wx.adv import AboutDialogInfo

import LocalRadiomicsGUI as locRad
from exportExcel import ExportExcel
from lymph_nodes import LymphNodes
from main_texture_pet import main_texture_pet
from main_texture_ct import main_texture_ct
from main_texture_mr import main_texture_mr
from main_texture_ctp import main_texture_ctp
from main_texture_ivim import main_texture_ivim
from myinfo import MyInfo
from panel import MyPanelRadiomics, MyPanelResize
from shape import Shape


class Radiomics(wx.Frame):
    """Main GUI class plus method OnCalculate to start radiomics calculation
        Parent of class Panel"""
    def __init__(self, *a, **b):
        super(Radiomics, self).__init__(size=(1100, 900), title='Z-Rad', *a, **b)
        self.SetMinSize((1000, 800))
        logging.basicConfig(format='%(name)s - %(levelname)s:   %(message)s', level=logging.INFO)
        # filename = "zRad.log",
            
        self.logger = logging.getLogger("Main")
       
        self.logger.info('Initialize GUI and Main')
        self.logger.info("Logger Mode: " + logging.getLevelName(self.logger.getEffectiveLevel()))
        self.InitUI()
        self.Show()
        self.dim = []

    def InitUI(self):
        self.local = False

        self.panelHeight=18 # height of boxes in GUI, 20 for PC and 40 for lenovo laptop

        self.p = wx.Panel(self, size=(1100,900))
        self.nb = wx.Notebook(self.p, size=(1100,900))

        self.nb.panelHeight = self.panelHeight
        self.nb.OnCalculate = self.OnCalculate

        self.panel = MyPanelRadiomics(self.nb)  # radiomics panel
        self.panelResize = MyPanelResize(self.nb)  # resize panel

        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panel, "Radiomics", select=True) # select True - first one
        
        menubar = wx.MenuBar() # create menu bar
        plikMenu = wx.Menu()
        no = wx.MenuItem(plikMenu, wx.ID_NEW, '&New\tCtrl+N') # new calculation
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

        locRadMenu = wx.Menu()
        new = wx.MenuItem(locRadMenu, wx.ID_ANY, 'New')
        locRadMenu.Append(new)
        ab = wx.MenuItem(locRadMenu, wx.ID_ANY, 'About')
        locRadMenu.Append(ab)
        menubar.Append(locRadMenu, 'Local Radiomics')

        self.SetMenuBar(menubar)

        config = open('config.txt', 'r') # read the configuration file
        l = []
        for i in config:
            l.append(i)
            # self.logger.debug("list of config " + i )

        self.panelResize.fill(l[:13])  # use the saved configuration
        self.panel.fill(l[13:])
        del l
        config.close()

        self.CreateStatusBar(1)
                
        #  connect menu with methods
        self.Bind(wx.EVT_MENU, self.OnOProgramie, program)
        self.Bind(wx.EVT_MENU, self.OnNew, no)
        self.Bind(wx.EVT_MENU, self.OnSave, sv)
        self.Bind(wx.EVT_MENU, self.OnQuit, za)
        self.Bind(wx.EVT_MENU, self.OnLocalRadNew, new)
        self.Bind(wx.EVT_MENU, self.OnLocalRadAbout, ab)

        # initialize panel
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)

        self.panel.Refresh()

    def OnLocalRadNew(self, evt):
        path_save, save_as, structure, pixNr, binSize, path_image, n_pref, start, stop = self.panel.read()
        self.logger.info("Start: local radiomics")
        MyInfo('Test done!')
        '''path_save - save results in, path
        save_as - name of text file to save radiomics
        structure - analysed studctures, later converted to list with ',' separation
        pixNr - number of bin
        binSize - fixed bin size
        path_image - path to images
        n_pref - prefix in the folder naming eg CTP_x
        start
        stop'''
        stop = int(stop) + 1
        start = int(start)

        # convert to a list
        if structure == '':
            structure = 'none'
            self.panel.FindWindowById(104).SetValue('none')
        else:
            structure = structure.split(',')
            for i in arange(1, len(structure)):
                if structure[i][0] == ' ':
                    structure[i] = structure[i][1:]

        # dimensionality
        if self.panel.FindWindowById(1061).GetValue():
            self.dim = '2D'
        else:
            self.dim = '3D'

        if n_pref != '':
            l_ImName = [n_pref + '_' + str(i) for i in arange(start, stop)]  # subfolders that you want to analyze
        else:
            l_ImName = [str(i) for i in arange(start, stop)]  # subfolders that you want to analyze

        # modality
        if self.panel.FindWindowById(120).GetValue():  # CT
            outlier_corr = self.panel.FindWindowById(127).GetValue()
            try:
                hu_min = int(self.panel.FindWindowById(125).GetValue())
                hu_max = int(self.panel.FindWindowById(126).GetValue())
            except ValueError:  # the input has t be a number
                hu_min = 'none'
                hu_max = 'none'
                self.panel.FindWindowById(125).SetValue('none')
                self.panel.FindWindowById(126).SetValue('none')

            locRad.CreateGUI(path_image, path_save, structure, pixNr, binSize, l_ImName, hu_min, hu_max,
                                 outlier_corr, False)

    def OnLocalRadAbout(self, evt):
        """info"""
        description = """"""

        licence = """"""

        info = AboutDialogInfo()

        info.SetName('Loc-Rad')
        info.SetVersion('0.1')
        info.SetDescription(description)
        info.SetCopyright('(C) 2018 Andreas Ambrusch')
        info.SetLicence(licence)

        AboutBox(info)

    def OnCalculate(self, evt):
        """initialize radiomics calculaiton"""

        path_save, save_as, structure, pixNr, binSize, path_image, n_pref, start, stop = self.panel.read()
        self.logger.info("Start: Calculate Radiomics")
        # self.logger.info( "Structures found", ', '.join(structure))
        MyInfo('Test done!')
        '''path_save - save results in, path
        save_as - name of text file to save radiomics
        structure - analysed studctures, later converted to list with ',' separation
        pixNr - number of bin
        binSize - fixed bin size
        path_image - path to images
        n_pref - prefix in the folder naming eg CTP_x
        start
        stop'''
        stop = int(stop)+1
        start = int(start)

        # convert to a list        
        if structure == '':
            structure = 'none'
            self.panel.FindWindowById(104).SetValue('none')
        else:
            structure = structure.split(',')
            for i in arange(1, len(structure)):
                if structure[i][0] == ' ':
                    structure[i] = structure[i][1:]

        # dimensionality
        if self.panel.FindWindowById(1061).GetValue():
            self.dim = '2D'
        else:
            self.dim = '3D'

        # wavelet
        if self.panel.FindWindowById(1071).GetValue():
            wv = True
        else:
            wv = False
        
        if n_pref != '':
            l_ImName = [n_pref+'_'+str(i) for i in arange(start, stop)]  # subfolders that you want to analyze
        else:
            l_ImName = [str(i) for i in arange(start, stop)]  # subfolders that you want to analyze
            
        # to be adapted 
        exportList = []
        cropStructure = {"crop": False, "ct_path": ""}

        # modality
        if self.panel.FindWindowById(120).GetValue():  # CT
            outlier_corr = self.panel.FindWindowById(127).GetValue()
            try:
                hu_min = int(self.panel.FindWindowById(125).GetValue())
                hu_max = int(self.panel.FindWindowById(126).GetValue())
            except ValueError: # the input has t be a number
                hu_min = 'none'
                hu_max = 'none'
                self.panel.FindWindowById(125).SetValue('none')
                self.panel.FindWindowById(126).SetValue('none')
            main_texture_ct(self.GetStatusBar(),path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, self.dim, hu_min, hu_max, outlier_corr,wv, self.local, cropStructure, exportList)
        
        elif self.panel.FindWindowById(130).GetValue():  # PET
            SUV = self.panel.FindWindowById(131).GetValue()
            cropArg = bool(self.panel.FindWindowById(133).GetValue())  # if crop
            ct_hu_min = 'none'
            ct_hu_max = 'none'
            ct_path = ""

            if cropArg: 
                self.logger.info("CropStructures " + str(cropArg))
                try:
                    ct_hu_min = int(self.panel.FindWindowById(135).GetValue())
                    ct_hu_max = int(self.panel.FindWindowById(136).GetValue()) 
                except ValueError: # the input has t be a number
                    ct_hu_min = 'none'
                    ct_hu_max = 'none'
                ct_path = self.panel.FindWindowById(137).GetValue()  # CT path
                if ct_path == "":
                    print("Error: No CT Path provided!")
                    raise
            cropStructure = {"crop" : cropArg, "hu_min" : ct_hu_min, "hu_max" : ct_hu_max, "ct_path" : ct_path}
            main_texture_pet(self.GetStatusBar(),path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, self.dim, SUV, wv, self.local, cropStructure, exportList)
        
        elif self.panel.FindWindowById(140).GetValue(): # CTP
            outlier_corr = self.panel.FindWindowById(141).GetValue()
            main_texture_ctp(self.GetStatusBar(),path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, self.dim, outlier_corr, wv,self.local, cropStructure, exportList)
        
        elif self.panel.FindWindowById(150).GetValue(): # MR
            struct_norm1 = self.panel.FindWindowById(151).GetValue()
            struct_norm2 = self.panel.FindWindowById(152).GetValue()
            main_texture_mr(self.GetStatusBar(),path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, self.dim,  struct_norm1, struct_norm2, wv, self.local, cropStructure,exportList)
        
        elif self.panel.FindWindowById(160).GetValue(): # IVIM
            main_texture_ivim(self.GetStatusBar(),path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, self.dim, wv,self.local, cropStructure, exportList)
        
        name_shape_pt = ""
        if self.panel.FindWindowById(1081).GetValue(): # calculate shape
            name_shape_pt = self.panel.FindWindowById(1083).GetValue() # name of ROI defined as PT for shape
            path_files_shape = path_image+'\\resized_1mm\\'+name_shape_pt+'\\'
            inp_mypath_results = path_save + '\\shape_'+name_shape_pt+'_'+str(start)+'_'+str(stop-1)+'.txt'
            Shape(path_files_shape, inp_mypath_results, start, stop)
            
        # calculate results for LN
        
        if self.panel.FindWindowById(1091).GetValue():
            name_ln = self.panel.FindWindowById(1093).GetValue() # name of ROI defined as LN for shape, for example g_LN, searches for g_LN_X
            name_shape_pt = self.panel.FindWindowById(1083).GetValue() # name of ROI defined as PT for shape
            print("name_shape_PT", name_shape_pt)
            path_files_shape = path_image+'\\resized_1mm\\'
            inp_mypath_results = path_save + '\\LN_'+'_'+str(start)+'_'+str(stop-1)+'.txt'
            LymphNodes(name_ln, name_shape_pt, path_files_shape, inp_mypath_results,path_save, start, stop)
        
        ifshape = self.panel.FindWindowById(1081).GetValue()
        ExportExcel(ifshape, path_save, name_shape_pt, start, stop, save_as)

        MyInfo('Radiomics done')
        
    def OnNew(self, evt):
        l = self.panel.save()
        lr = self.panelResize.save()
        size = self.panel.GetSize()
        self.nb.Destroy()
        self.nb = wx.Notebook(self.p, size=(1100, 900))

        self.nb.panelHeight = self.panelHeight
        self.nb.OnCalculate = self.OnCalculate
        
        self.panel = MyPanelRadiomics(self.nb, size=size)
        self.panel.fill(l)
        self.panelResize = MyPanelResize(self.nb, size=size)
        self.panelResize.fill(lr)
        self.nb.AddPage(self.panelResize, "Image and structure resize")
        self.nb.AddPage(self.panel, "Radiomics", select = True)

        self.sizer.Add(self.nb, 1, wx.EXPAND)
        self.p.SetSizer(self.sizer)

        self.panel.Layout()
        self.panelResize.Layout()
        
        del l
        del lr

    def OnQuit(self, evt):
        config = open('config.txt', 'w')
        lr = self.panelResize.save()
        l = self.panel.save()
        for i in lr:
            config.write('{}\n'.format(i))
        for i in l:
            config.write('{}\n'.format(i))
        config.close()
        self.Close()
        
    def OnSave(self, evt):
        config = open('config.txt', 'w')
        lr = self.panelResize.save()
        l = self.panel.save()
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
        info.SetVersion('7.0')
        info.SetDescription(description)
        info.SetCopyright('(C) 2017-2019 USZ Radiomics Team')
        info.SetLicence(licence)

        AboutBox(info)


# run the app
app = wx.App()
Radiomics(None)
app.MainLoop()
