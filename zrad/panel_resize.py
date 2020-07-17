import multiprocessing
import os

import wx
import wx.lib.scrolledpanel as scrolled

from check import CheckStructures
from myinfo import MyInfo
from resize_shape import ResizeShape
from resize_texture import ResizeTexture


class panelResize(scrolled.ScrolledPanel):
    def __init__(self, parent, id=-1, size=wx.DefaultSize, *a, **b):
        super(panelResize, self).__init__(parent, id, (0, 0), size=(800, 400), style=wx.SUNKEN_BORDER, *a, **b)
        self.parent = parent  # class Radiomics from main_texture.py is a parent
        self.maxWidth = 800  # width and height of the panel
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
        # text box, id is important as I use i later for reading elements from the boxes
        tc_org = wx.TextCtrl(self, id=1001, size=(600, h), value="", style=wx.TE_PROCESS_ENTER)
        # tc_org- directory with original images
        btn_load_org = wx.Button(self, -1, label='Search')  # button to search
        # directory to save resized images
        st_save_resized = wx.StaticText(self, label='Save resized files')
        tc_save_resized = wx.TextCtrl(self, id=1002, size=(600, h), value="", style=wx.TE_PROCESS_ENTER)
        btn_load_resized = wx.Button(self, -1, label='Search')
        # structures to be resized separated by coma ','
        st_name = wx.StaticText(self, label='Structure name')
        tc_name = wx.TextCtrl(self, id=1003, size=(600, h), value="", style=wx.TE_PROCESS_ENTER)

        # resolution for texture calculation
        st_reso = wx.StaticText(self, label='Resolution for texture calculation [mm]')
        tc_reso = wx.TextCtrl(self, id=1004, size=(100, h), value="", style=wx.TE_PROCESS_ENTER)

        # interpolation type
        int_type = wx.StaticText(self, label='Interpolation')
        inte_type = wx.ComboBox(self, id=1015, size=(100, 1.25 * h), value='linear', choices=['linear', 'nearest', 'cubic'],
                                style=wx.CB_READONLY)

        # imaging modality
        st_type = wx.StaticText(self, label='Image type')
        tc_type = wx.ComboBox(self, id=1005, size=(100, 1.25 * h), value="", choices=['CT', 'PET', 'MR', 'IVIM'],
                              style=wx.CB_READONLY)  # modality type

        # patient number to start
        st_start = wx.StaticText(self, label='Start')
        tc_start = wx.TextCtrl(self, id=1006, size=(100, h), value="", style=wx.TE_PROCESS_ENTER)
        # patient number to stop
        st_stop = wx.StaticText(self, label='Stop')
        tc_stop = wx.TextCtrl(self, id=1007, size=(100, h), value="", style=wx.TE_PROCESS_ENTER)

        # Number of CPU cores used for parallelization
        n_jobs_st = wx.StaticText(self, label='No. parallel jobs')
        n_jobs_cb = wx.ComboBox(self, id=1016, size=(100, 1.25 * h), value='1',
                                choices=[str(e) for e in range(1, multiprocessing.cpu_count()+1)],
                                style=wx.CB_READONLY)

        cb_cropStructure = wx.CheckBox(self, id=1012, label='Use CT Structure')
        st_cropStructure = wx.StaticText(self, label='     CT Path')  # static text
        # text box, id is important as I use i later for reading elements from the boxes
        tc_cropStructure = wx.TextCtrl(self, id=1013, size=(600, h), value="", style=wx.TE_PROCESS_ENTER)
        # tc_org- directory with original images
        btn_cropStructure = wx.Button(self, -1, label='Search')  # button to search

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
                            st_type, tc_type, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            st_start, tc_start, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            st_stop, tc_stop, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            n_jobs_st, n_jobs_cb, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            cb_texture, cb_texture_dim2, wx.StaticText(self, label=''), wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), cb_texture_dim3, wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''), cb_texture_none, wx.StaticText(self, label=''),
                            wx.StaticText(self, label=''),
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
        img = wx.Image('LogoUSZ.png', wx.BITMAP_TYPE_PNG).Scale(220, 40).ConvertToBitmap()
        im = wx.StaticBitmap(self, -1, img)

        h11box = wx.BoxSizer(wx.HORIZONTAL)
        h11box.Add((10, 10))
        h11box.Add(im)
        self.vbox.Add(h11box, flag=wx.RIGHT)
        self.vbox.Add((-1, 10))

        # connect buttons with methods
        # EVT_BUTTON when button named btn_resize was clicked bind with method self.resize
        self.Bind(wx.EVT_BUTTON, self.resize, btn_resize)
        self.Bind(wx.EVT_BUTTON, self.OnCheck, btn_check)
        self.Bind(wx.EVT_BUTTON, self.OnOpenOrg, btn_load_org)
        self.Bind(wx.EVT_BUTTON, self.OnOpenP_crop, btn_cropStructure)
        self.Bind(wx.EVT_BUTTON, self.OnOpenR, btn_load_resized)

        self.SetSizer(self.vbox)
        self.Layout()
        self.SetupScrolling()

    def OnOpenOrg(self, evt):
        """dialog box to find path with the original data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(1001).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(1001).SetValue(fop.GetPath() + os.sep)

    def OnOpenR(self, evt):
        """dialog box to find path where to save the data"""
        fop = wx.DirDialog(self, style=wx.DD_DEFAULT_STYLE)
        fop.SetPath(self.FindWindowById(1002).GetValue())
        if fop.ShowModal() == wx.ID_OK:
            self.FindWindowById(1002).SetValue(fop.GetPath() + os.sep)

    def resize(self, evt):  # need and event as an argument
        """main method which calls resize classes"""

        cropArg = False
        ct_path = ""
        inp_resolution = self.FindWindowById(1004).GetValue()  # take the input defined by user
        inp_struct = self.FindWindowById(1003).GetValue()
        inp_mypath_load = self.FindWindowById(1001).GetValue()
        inp_mypath_save = self.FindWindowById(1002).GetValue()
        interpolation_type = self.FindWindowById(1015).GetValue()
        image_type = self.FindWindowById(1005).GetValue()
        begin = int(self.FindWindowById(1006).GetValue())
        stop = int(self.FindWindowById(1007).GetValue())
        n_jobs = int(self.FindWindowById(1016).GetValue())
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

        for i in range(len(list_structure)):
            list_structure[i] = list_structure[i].strip()

        if self.FindWindowById(10082).GetValue() or self.FindWindowById(
                10083).GetValue():  # if resizing to texture resolution selected
            if self.FindWindowById(10082).GetValue():  # if 2D chosen
                dimension_resize = "2D"
            else:
                dimension_resize = "3D"

            print(interpolation_type)
            # resize images and structure to the resolution of texture
            ResizeTexture(inp_resolution, interpolation_type, list_structure, inp_mypath_load, inp_mypath_save,
                          image_type, begin, stop, cropInput, dimension_resize, n_jobs)

        if self.FindWindowById(1009).GetValue():  # if resizing to shape resolution selected
            inp_mypath_save_shape = inp_mypath_save + os.sep + 'resized_1mm' + os.sep

            for shape_struct in list_structure:
                # resize the structure to the resolution of shape, default 1mm unless resolution of texture smaller than
                # 1mm then 0.1 mm
                ResizeShape(shape_struct, inp_mypath_load, inp_mypath_save_shape, image_type, begin, stop,
                            inp_resolution, 'linear', cropInput, n_jobs)

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
            self.FindWindowById(1013).SetValue(fop.GetPath() + os.sep)

    def fill(self, l):
        """method called by parent to fill the text boxes with save settings
        l - list of elements read from a text file"""
        # ids of field to fill # if adjust number of ids then also adjust in main_texture in
        # self.panelResize.fill(l[:11])
        ids = [1001, 1002, 1003, 1004, 1015, 1005, 1006, 1007, 1016, 10081, 10082, 10083, 1009, 1012, 1013]

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
        ids = [1001, 1002, 1003, 1004, 1015, 1005, 1006, 1007, 1016, 10081, 10082, 10083, 1009, 1012, 1013]
        for i in ids:
            l.append(self.FindWindowById(i).GetValue())
        return l
