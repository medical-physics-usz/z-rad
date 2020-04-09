# -*- coding: utf-8 -*-s

# import libraries
import wx
import wx.lib.scrolledpanel as scrolled

class panelRadiomics(scrolled.ScrolledPanel):
    def __init__(self, parent, id=-1, size=(800, 400), *a, **b):
        super(panelRadiomics, self).__init__(parent, id, (0, 0), size=size, style=wx.SUNKEN_BORDER, *a, **b)
        self.parent = parent  # class Radiomics from main_texture.py is a parent
        self.maxWidth = 800  # width and height of the panel
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
        tc_saver = wx.TextCtrl(self, id=102, size=(857, h), value="", style=wx.TE_PROCESS_ENTER)  # path to save results
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

        img = wx.Image('LogoUSZ.png', wx.BITMAP_TYPE_PNG).Scale(220, 40).ConvertToBitmap()  # USZ logo
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
        self.SetupScrolling()

    def fill(self, l):
        """method called by parent to fill the text boxes with save settings
        l - list of elements read from a text file"""
        # ids of the boxes to be filled
        ids = [107, 108, 109, 110, 102, 103, 104, 1051, 1052, 1061, 10611, 1062, 1071, 1072, 1081, 1082, 1083, 1091, 1092,
               1093, 120, 125, 126, 127, 130, 131, 132, 133, 135, 136, 137, 140, 141, 142, 150, 151, 152, 160]

        for i in range(len(l)):
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
