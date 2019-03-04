import wx

class MyException(Exception):
    '''dialog box for messages
    info -message in the box'''
    def __init__(self, info):
        self.info = info
        self.box()
    def box(self):
        wx.MessageBox(self.info, u'Error', wx.OK|wx.ICON_EXCLAMATION)
