# -*- coding: utf-8 -*-s
import wx
import wx.lib.scrolledpanel as scrolled
import numpy as np
import pylab as py
from pylab import *
from matplotlib.widgets import Slider
from myinfo import MyInfo

from resize_texture import ResizeTexture
from resize_shape import ResizeShape
from check import CheckStructures


class MyPanel(wx.Notebook):  # scrolled.ScrolledPanel):
    """create a notebook"""


class MyPanelResize(scrolled.ScrolledPanel):
    def __init__(self, parent, id=-1, size=wx.DefaultSize, *a, **b):
        super(MyPanelResize, self).__init__(parent, id, (0, 0), size=(800, 400), style=wx.SUNKEN_BORDER, *a, **b)
        self.parent = parent  # class Radiomics from main_texture.py is a parent
        self.maxWidth = 800  # width and hight of the panel
        self.maxHeight = 400

        self.InitUI()

    def InitUI(self):
        """initialize the panel
        the IDs are assigned in a consecutive order and are used later to refer to text boxes etc"""

        h = self.parent.panelHeight  # height of a text box, 20 for PC, 40 for lenovo laptop
        self.SetBackgroundColour('#8AB9F1')  # background color

        # creatignngoxes containing elements of the panel, vbox - vertical box, hbox - horizontal box
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add((-1, 20))

        self.gs_01 = wx.FlexGridSizer(cols=4, vgap=5, hgap=10)  # grid sizes is a box with 3 columns
        # elements I want to put the the grid sizer
        st_org = wx.StaticText(self, label='Original data')  # static text
        tc_org = wx.TextCtrl(self, id=1001, size=(600, h), value="",
                             style=wx.TE_PROCESS_ENTER)  # text box, id is important as I use i later for reading elements from the boxes
        # tc_org- directory with original images
        btn_load_org = wx.Button(self, -1, label='Search')  # button to search
        st_save_resized = wx.StaticText(self, label='Save resized files')
        tc_save_resized = wx.TextCtrl(self, id=1002, size=(600, h), value="",
                                      style=wx.TE_PROCESS_ENTER)  # directory to save resized images
        btn_load_resized = wx.Button(self, -1, label='Search')
        st_name = wx.StaticText(self, label='Structure name')
        tc_name = wx.TextCtrl(self, id=1003, size=(600, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # structures to be resized separated by coma ','
        st_reso = wx.StaticText(self, label='Resolution for texture calculation [mm]')
        tc_reso = wx.TextCtrl(self, id=1004, size=(100, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # resolution for texture calculation
####
        int_type = wx.StaticText(self, label='Interpolation (default = linear):')
        inte_type = wx.ComboBox(self, id=1015, size=(100, 2*h), value='linear', choices=['linear', 'nearest', 'cubic'],
                              style=wx.CB_READONLY)  # Type of Interpolation used -- read only drop down menu to avoid errors
        #inte_type=wx.Choice(self, id=1015, pos, size=(100, h), n, choices=['linear', 'nearest', 'cubic'], style)
        #inte_type=wx.Choice(self, id=1015, pos, size=(100, h), n, choices[], style)
        st_type = wx.StaticText(self, label='Image type')
        tc_type = wx.ComboBox(self, id=1005, size=(100, 2*h), value="", choices=['CT', 'PET', 'MR', 'IVIM'],
                              style=wx.CB_READONLY)  # modality type
        st_start = wx.StaticText(self, label='Start')
        tc_start = wx.TextCtrl(self, id=1006, size=(100, h), value="",
                               style=wx.TE_PROCESS_ENTER)  # patient number to start
        st_stop = wx.StaticText(self, label='Stop')
        tc_stop = wx.TextCtrl(self, id=1007, size=(100, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # patient number to stop

        cb_cropStructure = wx.CheckBox(self, id=1012, label='Use CT Structure')
        st_cropStructure = wx.StaticText(self, label='     CT Path')  # static text
        tc_cropStructure = wx.TextCtrl(self, id=1013, size=(600, h), value="",
                                       style=wx.TE_PROCESS_ENTER)  # text box, id is important as I use i later for reading elements from the boxes
        # tc_org- directory with original images
        btn_cropStructure = wx.Button(self, -1, label='Search')  # button to search

        # cb_texture = wx.CheckBox(self, id=1008, label='Resize texture')
        cb_texture = wx.StaticText(self, id=1008, label='Resize texture')
        cb_texture_none = wx.RadioButton(self, id=10081, label='no texture resizing', style=wx.RB_GROUP)
        cb_texture_dim2 = wx.RadioButton(self, id=10082, label='2D')  # only one option can be selected
        cb_texture_dim3 = wx.RadioButton(self, id=10083, label='3D')

        cb_shape = wx.CheckBox(self, id=1009, label='Resize shape')

        btn_resize = wx.Button(self, id=1010, label='Resize')
        btn_check = wx.Button(self, id=1011, label='Check')

        # fill the grid sizer with elements
        self.gs_01.AddMany([st_org, tc_org, btn_load_org, wx.StaticText(self, label=''),
                            st_save_resized, tc_save_resized, btn_load_resized, wx.StaticText(self, label=''),
                            st_name, tc_name, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            st_reso, tc_reso, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            int_type, inte_type, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            st_type, tc_type, wx.StaticText(self, label=''), wx.StaticText(self, label=''), ##
                            st_start, tc_start, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            st_stop, tc_stop, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            cb_texture, cb_texture_dim2, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), cb_texture_dim3, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), cb_texture_none, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            cb_shape, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''),
                            cb_cropStructure, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''),
                            st_cropStructure, tc_cropStructure, btn_cropStructure, wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), wx.StaticText(self, label=''), btn_check,
                            wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), wx.StaticText(self, label=''), btn_resize,
                            wx.StaticText(self, label='')])

        st01 = wx.StaticLine(self, -1, (10, 1), (900, 3))

        # add grid size to a hbox
        h01box = wx.BoxSizer(wx.HORIZONTAL)
        h01box.Add((10, 10))
        h01box.Add(self.gs_01)
        self.vbox.Add(h01box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        # add hbox to vbox
        h011box = wx.BoxSizer(wx.HORIZONTAL)
        h011box.Add((10, 10))
        h011box.Add(st01)
        self.vbox.Add(h011box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        # add logo
        img = wx.Image('LogoUSZ.png', wx.BITMAP_TYPE_PNG).Scale(220, 60).ConvertToBitmap()
        im = wx.StaticBitmap(self, -1, img)

        h11box = wx.BoxSizer(wx.HORIZONTAL)
        h11box.Add((10, 10))
        h11box.Add(im)
        self.vbox.Add(h11box, flag=wx.RIGHT)
        self.vbox.Add((-1, 10))

        # connect buttons with methods
        self.Bind(wx.EVT_BUTTON, self.resize, btn_resize)  # EVT_BUTTON when button named btn_resize was clicked bind with method self.resize
        self.Bind(wx.EVT_BUTTON, self.OnCheck, btn_check)
        self.Bind(wx.EVT_BUTTON, self.OnOpenOrg, btn_load_org)
        self.Bind(wx.EVT_BUTTON, self.OnOpenP_crop, btn_cropStructure)
        self.Bind(wx.EVT_BUTTON, self.OnOpenR, btn_load_resized)

        self.SetSizer(self.vbox)
        self.Layout()

    def OnOpenOrg(self, evt):
        """dialog box to find path with the original data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(1001).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(1001).SetValue(fop.GetPath() + '\\')

    def OnOpenR(self, evt):
        """dialog box to find path where to save the data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(1002).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(1002).SetValue(fop.GetPath() + '\\')

    def resize(self, evt):  # need and event as an argument
        """main method which calls resize classes"""
        cropArg = False
        ct_path = ""
        inp_resolution = self.FindWindowById(1004).GetValue()  # take the input defined by user
        inp_struct = self.FindWindowById(1003).GetValue()
        inp_mypath_load = self.FindWindowById(1001).GetValue()
        inp_mypath_save = self.FindWindowById(1002).GetValue()
        interpolation_type=self.FindWindowById(1015).GetValue()
        image_type = self.FindWindowById(1005).GetValue()####
        begin = int(self.FindWindowById(1006).GetValue())
        stop = int(self.FindWindowById(1007).GetValue())
        cropArg = bool(self.FindWindowById(1012).GetValue())
        ct_path = self.FindWindowById(1013).GetValue()
        if not cropArg:
            cropInput = {"crop": cropArg, "ct_path": ""}
        else:
            cropInput = {"crop": cropArg, "ct_path": ct_path}
        # divide a string with structures names to a list of names

        if ',' not in inp_struct:
            list_structure = [inp_struct]
        else:
            list_structure = inp_struct.split(',')

        for i in range(0, len(list_structure)):
            list_structure[i] = list_structure[i].strip()

        if self.FindWindowById(10082).GetValue() or self.FindWindowById(10083).GetValue():  # if resizing to texture resolution selected
            if self.FindWindowById(10082).GetValue():  # if 2D chosen
                dimension_resize = "2D"
            else:
                dimension_resize = "3D"

            print(interpolation_type)
            ResizeTexture(inp_resolution, interpolation_type, list_structure, inp_mypath_load, inp_mypath_save, image_type, begin, stop,
                          cropInput, dimension_resize)  # resize images and structure to the resolution of texture
####
        if self.FindWindowById(1009).GetValue():  # if resizing to shape resolution selected
            inp_mypath_save_shape = inp_mypath_save + '\\resized_1mm\\'

            for shape_struct in list_structure:
                ResizeShape(shape_struct, inp_mypath_load, inp_mypath_save_shape, image_type, begin, stop,
                            inp_resolution, interpolation_type,
                            cropInput)  # resize the structure to the resolution of shape, default 1mm unless resolution of texture smaller than 1mm then 0.1 mm

        MyInfo('Resize done')  # show info box

    def OnCheck(self, evt):
        inp_struct = self.FindWindowById(1003).GetValue()
        inp_mypath_load = self.FindWindowById(1001).GetValue()
        begin = int(self.FindWindowById(1006).GetValue())
        stop = int(self.FindWindowById(1007).GetValue())

        CheckStructures(inp_struct, inp_mypath_load, begin, stop)

        MyInfo('Check done: file saved in ' + inp_mypath_load)

    def OnOpenP_crop(self, evt):  # need and event as an argument
        """dialog box to find path with the resized data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(1013).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(1013).SetValue(fop.GetPath() + '\\')

    def fill(self, l):
        """method called by parent to fill the text boxes with save settings
        l - list of elements read from a text file"""
        ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 10081, 10082, 10083, 1009, 1012,
               1013]  # ids of field to fill # if adjust number of ids then also adjust in main_texture in self.panelResize.fill(l[:11])!!!!

        for i in range(0, len(l)):
            try:
                if l[i][-1] == '\n':  # check if there is an end of line sign and remove
                    try:
                        self.FindWindowById(ids[i]).SetValue(l[i][:-1])
                    except TypeError:
                        if l[i][:-1] == 'True':
                            v = True
                        else:
                            v = False
                        self.FindWindowById(ids[i]).SetValue(v)
                else:
                    try:
                        self.FindWindowById(ids[i]).SetValue(l[i])
                    except TypeError:
                        if l[i] == 'True':
                            v = True
                        else:
                            v = False
                        self.FindWindowById(ids[i]).SetValue(v)
            except TypeError:
                self.FindWindowById(ids[i]).SetValue(l[i])
            except IndexError:
                pass
        self.Layout()  # refresh the view

    def save(self):
        """save the last used settings"""
        l = []
        ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 10081, 10082, 10083, 1009, 1012, 1013]
        for i in ids:
            l.append(self.FindWindowById(i).GetValue())
        return l


class MyPanelRadiomics(scrolled.ScrolledPanel):
    def __init__(self, parent, id=-1, size=(800, 400), *a, **b):
        super(MyPanelRadiomics, self).__init__(parent, id, (0, 0), size=size, style=wx.SUNKEN_BORDER, *a, **b)
        self.parent = parent  # class Radiomics from main_texture.py is a parent
        self.maxWidth = 800  # width and hight of the panel
        self.maxHeight = 400

        self.InitUI()

    def InitUI(self):
        self.SetBackgroundColour('#8AB9F1')  # background color
        h = self.parent.panelHeight  # height of a text box, 20 for PC, 40 for lenovo laptop

        # creating boxes containing elements of the panel, vbox - vertical box, hbox - horizontal box
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add((-1, 20))

        st1 = wx.StaticLine(self, -1, (10, 1), (900, 3))  # line to separate sections
        st2 = wx.StaticLine(self, -1, (10, 1), (900, 3))
        st3 = wx.StaticLine(self, -1, (10, 1), (900, 3))
        st7 = wx.StaticLine(self, -1, (10, 1), (900, 3))

        self.gs_3 = wx.FlexGridSizer(cols=3, vgap=5, hgap=10)  # grid size with 3 columns

        st_path = wx.StaticText(self, label='Load')
        tc_path = wx.TextCtrl(self, id=107, size=(900, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # path to resized images
        btn_load_path = wx.Button(self, -1, label='Search')  # button to open dialog box
        st_pref = wx.StaticText(self, label='Prefix')
        tc_pref = wx.TextCtrl(self, id=108, size=(400, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # prefix in the name of patients folders for example HN, goes to HN_X
        st_range_l = wx.StaticText(self, label='Start')
        tc_range_l = wx.TextCtrl(self, id=109, size=(400, h), value="",
                                 style=wx.TE_PROCESS_ENTER)  # patient number start
        st_range_u = wx.StaticText(self, label='Stop')
        tc_range_u = wx.TextCtrl(self, id=110, size=(400, h), value="", style=wx.TE_PROCESS_ENTER)  # patient number end

        # fill grid sizer
        self.gs_3.AddMany([st_path, tc_path, btn_load_path,
                           st_pref, tc_pref, wx.StaticText(self, label=''),
                           st_range_l, tc_range_l, wx.StaticText(self, label=''),
                           st_range_u, tc_range_u, wx.StaticText(self, label='')])

        h3box = wx.BoxSizer(wx.HORIZONTAL)
        h3box.Add((10, 10))
        h3box.Add(self.gs_3)
        self.vbox.Add(h3box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        h31box = wx.BoxSizer(wx.HORIZONTAL)
        h31box.Add((10, 10))
        h31box.Add(st3)
        self.vbox.Add(h31box, flag=wx.LEFT)
        self.vbox.Add((-1, 20))

        self.gs_1 = wx.FlexGridSizer(cols=3, vgap=5, hgap=10)
        st_saver = wx.StaticText(self, label='Save results in')
        tc_saver = wx.TextCtrl(self, id=102, size=(900, h), value="", style=wx.TE_PROCESS_ENTER)  # path to save results
        btn_load_saver = wx.Button(self, -1, label='Search')
        st_saveas = wx.StaticText(self, label='Save as')
        tc_saveas = wx.TextCtrl(self, id=103, size=(400, h), value="",
                                style=wx.TE_PROCESS_ENTER)  # name of the text file to save texture results

        self.gs_1.AddMany([st_saver, tc_saver, btn_load_saver,
                           st_saveas, tc_saveas, wx.StaticText(self, label='')])

        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add((10, 10))
        h1box.Add(self.gs_1)
        self.vbox.Add(h1box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        h11box = wx.BoxSizer(wx.HORIZONTAL)
        h11box.Add((10, 10))
        h11box.Add(st1)
        self.vbox.Add(h11box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        # calculation details
        self.gs_2 = wx.FlexGridSizer(cols=4, vgap=5, hgap=10)

        st_name = wx.StaticText(self, label='Structure name ')
        # name of ROI to be analysed, separate by ',';  take the first structure which can be found in the RS
        tc_name = wx.TextCtrl(self, id=104, size=(200, h), value="", style=wx.TE_PROCESS_ENTER)
        st_disc = wx.StaticText(self, label='Discretization')
        st_bin = wx.StaticText(self, label='Bins No')
        st_size = wx.StaticText(self, label='Bin size')
        tc_bin = wx.TextCtrl(self, id=1051, size=(200, h), value="", style=wx.TE_PROCESS_ENTER)  # size of the bin
        tc_size = wx.TextCtrl(self, id=1052, size=(200, h), value="",
                              style=wx.TE_PROCESS_ENTER)  # number of bins for discretization
        st_dim = wx.StaticText(self, label='Dimension')
        rb_dim2 = wx.RadioButton(self, id=1061, label='2D several slices',
                                 style=wx.RB_GROUP)  # 2D analysis, radiobutton - only one option from the group can be selected
        rb_dim2single = wx.RadioButton(self, id=10611, label='2D single slice')
        rb_dim3 = wx.RadioButton(self, id=1062, label="3D volume")  # 3D analysis
        st_wv = wx.StaticText(self, label='Wavelet transform ')
        rb_wvT = wx.RadioButton(self, id=1071, label='On',
                                style=wx.RB_GROUP)  # wavelet on, radiobutton - only one option from the group can be selected
        rb_wvF = wx.RadioButton(self, id=1072, label="Off")  # wavelet off
        st_s = wx.StaticText(self, label='Shape ')
        rb_sT = wx.RadioButton(self, id=1081, label='On',
                               style=wx.RB_GROUP)  # shape on, radiobutton - only one option from the group can be selected
        rb_sF = wx.RadioButton(self, id=1082, label="Off")  # shape off
        st_name_shape = wx.StaticText(self, label='Structure name shape ')
        tc_name_shape = wx.TextCtrl(self, id=1083, size=(200, h), value="", style=wx.TE_PROCESS_ENTER)
        st_LN = wx.StaticText(self, label='Lymph nodes ')
        rb_LNT = wx.RadioButton(self, id=1091, label='On',
                                style=wx.RB_GROUP)  # analysis of distribution of LN around PT on, radiobutton - only one option from the group can be selected
        rb_LNF = wx.RadioButton(self, id=1092, label="Off")  # analysis of distribution of LN around PT off
        st_name_LN = wx.StaticText(self, label='Name lymph node structure ')
        tc_name_LN = wx.TextCtrl(self, id=1093, size=(200, h), value="",
                                 style=wx.TE_PROCESS_ENTER)  # name of LN structure to be analysed, for example g_LN will search for g_LN_X

        # fill grid sizer
        self.gs_2.AddMany([st_name, tc_name, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                           st_disc, st_bin, st_size, wx.StaticText(self, label=''),
                           wx.StaticText(self, label=''), tc_bin, tc_size, wx.StaticText(self, label=''),
                           st_dim, rb_dim2single, rb_dim2, rb_dim3,
                           st_wv, rb_wvT, rb_wvF, wx.StaticText(self, label=''),
                           st_s, rb_sT, rb_sF, wx.StaticText(self, label=''),
                           wx.StaticText(self, label=''), st_name_shape, tc_name_shape, wx.StaticText(self, label=''),
                           st_LN, rb_LNT, rb_LNF, wx.StaticText(self, label=''),
                           wx.StaticText(self, label=''), st_name_LN, tc_name_LN, wx.StaticText(self, label='')])

        h2box = wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add((10, 10))
        h2box.Add(self.gs_2)
        self.vbox.Add(h2box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        h21box = wx.BoxSizer(wx.HORIZONTAL)
        h21box.Add((10, 10))
        h21box.Add(st2)
        self.vbox.Add(h21box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        # modality dependent details
        self.gs_4 = wx.FlexGridSizer(cols=3, vgap=5, hgap=10)

        st_mod = wx.StaticText(self, label='Modality')

        h4box = wx.BoxSizer(wx.HORIZONTAL)
        h4box.Add((10, 10))
        h4box.Add(st_mod)
        self.vbox.Add(h4box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        self.gs_5 = wx.FlexGridSizer(cols=3, vgap=5, hgap=10)
        rb_ct = wx.RadioButton(self, id=120, label='CT', style=wx.RB_GROUP)  # CT

        st_humin = wx.StaticText(self, label='HU min')
        tc_humin = wx.TextCtrl(self, id=125, size=(200, h), value='', style=wx.TE_PROCESS_ENTER)  # HU range min
        st_humax = wx.StaticText(self, label='HU max')
        tc_humax = wx.TextCtrl(self, id=126, size=(200, h), value='', style=wx.TE_PROCESS_ENTER)  # HU range max

        rb_yes = wx.CheckBox(self, id=127, label='outliers correction')  # outliers corretion, checked if on

        self.gs_5.AddMany([rb_ct, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                           wx.StaticText(self, label=''), st_humin, tc_humin,
                           wx.StaticText(self, label=''), st_humax, tc_humax,
                           wx.StaticText(self, label=''), rb_yes, wx.StaticText(self, label='')])

        h5box = wx.BoxSizer(wx.HORIZONTAL)
        h5box.Add((10, 10))
        h5box.Add(self.gs_5)
        self.vbox.Add(h5box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        self.gs_6 = wx.FlexGridSizer(cols=5, vgap=2.5, hgap=10)
        rb_pet = wx.RadioButton(self, id=130, label='PET')
        st_suv = wx.StaticText(self, label='SUV correction')
        rb_yes = wx.CheckBox(self, id=131, label='yes')  # SUV correciton on
        rb_no = wx.CheckBox(self, id=132, label='no')  # SUV correciton off

        crop_text = wx.StaticText(self, label='Use CT Structure')  # crop PET structure to CT structure set HU range
        crop_yes = wx.CheckBox(self, id=133, label='yes')  # SUV correciton on
        crop_no = wx.CheckBox(self, id=134, label='no')  # SUV correciton on

        crop_st_humin = wx.StaticText(self, label='HU min')
        crop_tc_humin = wx.TextCtrl(self, id=135, size=(200, h), value='', style=wx.TE_PROCESS_ENTER)  # HU range min
        crop_st_humax = wx.StaticText(self, label='HU max')
        crop_tc_humax = wx.TextCtrl(self, id=136, size=(200, h), value='', style=wx.TE_PROCESS_ENTER)  # HU range max
        crop_ctpathlabel = wx.StaticText(self, label='CT Path')
        crop_ctpath = wx.TextCtrl(self, id=137, size=(400, h), value="",
                                  style=wx.TE_PROCESS_ENTER)  # path to resized images
        crop_btn_load_path = wx.Button(self, -1, label='Search')  # button to open dialog box
        self.gs_6.AddMany(
            [rb_pet, wx.StaticText(self, label=''), wx.StaticText(self, label=''), wx.StaticText(self, label=''),
             wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), st_suv, rb_yes, rb_no, wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), crop_text, crop_yes, crop_no, wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), wx.StaticText(self, label=''), crop_st_humin, crop_tc_humin,
             wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), wx.StaticText(self, label=''), crop_st_humax, crop_tc_humax,
             wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), wx.StaticText(self, label=''), crop_ctpathlabel, crop_ctpath,
             crop_btn_load_path])

        h6box = wx.BoxSizer(wx.HORIZONTAL)
        h6box.Add((10, 10))
        h6box.Add(self.gs_6)
        self.vbox.Add(h6box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        self.gs_7 = wx.FlexGridSizer(cols=4, vgap=5, hgap=10)
        rb_ctp = wx.RadioButton(self, id=140, label='CTP')
        st_suv = wx.StaticText(self, label='outliers correction')
        rb_yes = wx.CheckBox(self, id=141, label='yes')  # outlier correction correction on
        rb_no = wx.CheckBox(self, id=142, label='no')

        self.gs_7.AddMany(
            [rb_ctp, wx.StaticText(self, label=''), wx.StaticText(self, label=''), wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), st_suv, rb_yes, rb_no])

        h7box = wx.BoxSizer(wx.HORIZONTAL)
        h7box.Add((10, 10))
        h7box.Add(self.gs_7)
        self.vbox.Add(h7box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        self.gs_8 = wx.FlexGridSizer(cols=4, vgap=5, hgap=10)
        rb_mr = wx.RadioButton(self, id=150, label='MR')
        st_stan = wx.StaticText(self, label='ROI for standardization')
        tc_struct1 = wx.TextCtrl(self, id=151, size=(200, h), value='',
                                 style=wx.TE_PROCESS_ENTER)  # names of structures for linaer function fitting to normalize MR
        tc_struct2 = wx.TextCtrl(self, id=152, size=(200, h), value='', style=wx.TE_PROCESS_ENTER)

        self.gs_8.AddMany(
            [rb_mr, wx.StaticText(self, label=''), wx.StaticText(self, label=''), wx.StaticText(self, label=''),
             wx.StaticText(self, label=''), st_stan, tc_struct1, tc_struct2])

        h8box = wx.BoxSizer(wx.HORIZONTAL)
        h8box.Add((10, 10))
        h8box.Add(self.gs_8)
        self.vbox.Add(h8box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        self.gs_9 = wx.FlexGridSizer(cols=4, vgap=5, hgap=10)
        rb_ivim = wx.RadioButton(self, id=160, label='IVIM')

        self.gs_9.AddMany(
            [rb_ivim, wx.StaticText(self, label=''), wx.StaticText(self, label=''), wx.StaticText(self, label='')])

        h9box = wx.BoxSizer(wx.HORIZONTAL)
        h9box.Add((10, 10))
        h9box.Add(self.gs_9)
        self.vbox.Add(h9box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        h71box = wx.BoxSizer(wx.HORIZONTAL)
        h71box.Add((10, 10))
        h71box.Add(st7)
        self.vbox.Add(h71box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        btn_calculate = wx.Button(self, -1, label='Calculate')  # button to start radiomics
        h10box = wx.BoxSizer(wx.HORIZONTAL)
        h10box.Add((10, 10))
        h10box.Add(btn_calculate)
        self.vbox.Add(h10box, flag=wx.LEFT)
        self.vbox.Add((-1, 10))

        img = wx.Image('LogoUSZ.png', wx.BITMAP_TYPE_PNG).Scale(220, 60).ConvertToBitmap()  # USZ logo
        im = wx.StaticBitmap(self, -1, img)

        h11box = wx.BoxSizer(wx.HORIZONTAL)
        h11box.Add((10, 10))
        h11box.Add(im)
        self.vbox.Add(h11box, flag=wx.RIGHT)
        self.vbox.Add((-1, 10))

        self.Bind(wx.EVT_BUTTON, self.parent.OnCalculate, btn_calculate)  # EVT_BUTTON when button named btn_calculate was clicked bind with method self.parent.OnCalculate
        self.Bind(wx.EVT_BUTTON, self.OnOpenP, btn_load_path)
        self.Bind(wx.EVT_BUTTON, self.OnOpenP_crop, crop_btn_load_path)
        self.Bind(wx.EVT_BUTTON, self.OnOpenSR, btn_load_saver)

        self.SetSizer(self.vbox)  # add vbox to panel
        self.Layout()  # show panel

    def fill(self, l):
        """method called by parent to fill the text boxes with save settings
        l - list of elements read from a text file"""
        # ids of the boxes to be filled
        ids = [107, 108, 109, 110, 102, 103, 104, 1051, 1052, 1061, 10611, 1062, 1071, 1072, 1081, 1082, 1083, 1091, 1092,
               1093, 120, 125, 126, 127, 130, 131, 132, 133, 135, 136, 137, 140, 141, 142, 150, 151, 152, 160]

        for i in range(0, len(l)):
            # print "fill", repr(l[i])
            try:
                if l[i][-1] == '\n':  # check if there is an end of line sign and remove
                    try:
                        self.FindWindowById(ids[i]).SetValue(l[i][:-1])
                    except TypeError:
                        if l[i][:-1] == 'True':
                            v = True
                        else:
                            v = False
                        self.FindWindowById(ids[i]).SetValue(v)
                else:
                    try:
                        self.FindWindowById(ids[i]).SetValue(l[i])
                    except TypeError:
                        if l[i] == 'True':
                            v = True
                        else:
                            v = False
                        self.FindWindowById(ids[i]).SetValue(v)
            except TypeError:
                self.FindWindowById(ids[i]).SetValue(l[i])
            except IndexError:
                pass
        self.Layout()  # refresh view

    def save(self):
        """save the last used settings"""
        l = []

        ids = [107, 108, 109, 110, 102, 103, 104, 1051, 1052, 1061, 10611, 1062, 1071, 1072, 1081, 1082, 1083, 1091, 1092,
               1093, 120, 125, 126, 127, 130, 131, 132, 133, 135, 136, 137, 140, 141, 142, 150, 151, 152, 160]
        for i in ids:
            l.append(self.FindWindowById(i).GetValue())
        return l

    def read(self):
        """method called by parent class to read some of user defined values"""
        l = []
        ids = [102, 103, 104, 1051, 1052, 107, 108, 109, 110]
        for i in ids:
            l.append(self.FindWindowById(i).GetValue())
            # print "read", repr(self.FindWindowById(i).GetValue())
        return l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]

    def OnOpenP(self, evt):  # need and event as an argument
        """dialog box to find path with the resized data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(107).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(107).SetValue(fop.GetPath() + '\\')

    def OnOpenP_crop(self, evt):  # need and event as an argument
        """dialog box to find path with the resized data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(138).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(138).SetValue(fop.GetPath() + '\\')

    def OnOpenSR(self, evt):  # need and event as an argument
        """dialog box to define path to save results"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(102).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(102).SetValue(fop.GetPath() + '\\')
