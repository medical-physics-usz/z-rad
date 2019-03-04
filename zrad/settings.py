# -*- coding: utf-8 -*-
import wx

class Set(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(Set, self).__init__(*args, **kwargs) 
            
        self.InitUI()
        
    def InitUI(self):    

        self.panel = wx.Panel(self, size=(300,300))
        self.panel.SetBackgroundColour('#8AB9F1') #kolor tła

        l = wx.ComboBox(self, id = 100, choices=['PC', 'lenovo'])
        b = wx.Button(self, -1, label = 'OK')
         
        self.SetTitle('Settings')

        self.panel.vbox = wx.BoxSizer(wx.VERTICAL)
        h1box = wx.BoxSizer(wx.HORIZONTAL)
        h1box.Add((10, 10))
        h1box.Add(l)
        h2box = wx.BoxSizer(wx.HORIZONTAL)
        h2box.Add((10, 10))
        h2box.Add(b)
        self.panel.vbox.Add(h1box, flag=wx.LEFT)
        self.panel.vbox.Add(h2box, flag=wx.LEFT)
        self.panel.vbox.Add((-1, 10))

        self.Bind(wx.EVT_BUTTON, self.Return, b)
        self.SetSizer(self.panel.vbox)

        self.panel.Layout()
        
        self.Show()

    def Return(self, evt):
        t = self.panel.FindWindowById(100).GetValue()
        if t == 'PC':
            h = 20
        elif t == 'lenovo':
            h = 40
        else:
            h = 30
        self.h = h


##ex = wx.App()
##Set(None)
##ex.MainLoop()    

 
