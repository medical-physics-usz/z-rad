import wx

class MyInfo(Exception):
    def __init__(self, info):
        self.info = info
        self.box()
    def box(self):
        wx.MessageBox(self.info, 'Info', wx.OK)
