# -*- coding: utf-8 -*-

# import libraries
import pydicom as dc # dicom library
from pydicom.filereader import InvalidDicomError
import numpy as np
from os import path, makedirs, listdir, rmdir
from os.path import isfile, join
import xlsxwriter as xlsx

# own classes
from exception import MyException

class CheckStructures(object):
    '''Class called by MyPanelResize
    It reads all the patients folders, searches for RS and checks if structures to be resized are in the RS. Returns an excel file with a list of structures in RS and indication if a user-define ROI is one of the structures
    Type: object
    Attributes:
    inp_struct - string containing the structures to be resized separated by ‘,’
    inp_mypath_load – path to all the patients directory, also the path where the final excel file is saved
    begin – start number
    stop – stop number
    '''

    def __init__(self, inp_struct, inp_mypath_load, begin, stop):
        
        #divide a string with structures names to a list of names
        if ',' not in inp_struct: 
            self.list_structure = [inp_struct]
        else:
            self.list_structure =inp_struct.split(',')
        for i in range(len(self.list_structure)):
            if self.list_structure[i][0]==' ':
                self.list_structure[i] = self.list_structure[i][1:]
            if self.list_structure[i][-1]==' ':
                self.list_structure[i] = self.list_structure[i][:-1]
                
        dict_check = self.CheckNames(inp_mypath_load, begin, stop)
        self.Save(inp_mypath_load, dict_check)
        
                
    def CheckNames(self, inp_mypath_load, begin, stop):
        '''checks if the names specifeid by usser are present in the structure set'''
        RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage'] #structure set
        
        #create a dictornary to save results        
        dict_check = {}
        dict_check['pat_nr'] = [] #patient number
        dict_check['ROI_rs'] = [] #list of structures in the RS
        for s in self.list_structure: #create a key for each user defined structure
            dict_check[s] = []
          
        for n in range(begin, stop+1): 
            mypath_file = inp_mypath_load + str(n) + '\\'
            try:              
                rs=[]
                for f in listdir(mypath_file):
                    try:
                        if isfile(join(mypath_file,f)) and dc.read_file(mypath_file+f).SOPClassUID in RS_UID: #read only dicoms of certain modality
                            rs.append(f)
                    except InvalidDicomError: #not a dicom file   
                        pass
                    
                if len(rs)!=1: #more than one RS or missing RS
                    dict_check['pat_nr'].append(n)
                    dict_check['ROI_rs'].append(['RS number != 1'])
                    for s in self.list_structure:
                        dict_check[s].append('')
                
                else:
                    rs_name = mypath_file+rs[0]
                    rs = dc.read_file(rs_name) #read rs

                    list_organs_names = [] #ROI names
                    for j in range(len(rs.StructureSetROISequence)):
                        list_organs_names.append(rs.StructureSetROISequence[j].ROIName)
                    
                    dict_check['pat_nr'].append(n)
                    dict_check['ROI_rs'].append(list_organs_names)
                    
                    for s in self.list_structure:
                        dict_check[s].append(s in list_organs_names)
                                        
            except WindowsError: #no directory
                pass
            except IOError: #no directory
                pass 
            
        return dict_check
        
    def Save(self, inp_mypath_load, dict_check):
        '''saves results to an excel file'''
        try:
            wb = xlsx.Workbook(inp_mypath_load+'ROI_check.xlsx') #new excel file to save results 
            
            #cell formating type, green or red background
            formatTrue = wb.add_format()
            formatTrue.set_bg_color('green')
            formatFalse = wb.add_format()
            formatFalse.set_bg_color('red')
            
            ws = wb.add_worksheet()
            ws.name = 'ROI_check'
                    
            ws.write(0, 0, 'patient number')
            for s in range(len(self.list_structure)):
                ws.write(s+1, 0, self.list_structure[s])
            ws.write(0, 0, 'ROI in RS')
            
            max_pat_nr = len(dict_check['pat_nr'])
            for n in range(max_pat_nr):
                ws.write(0, n+1, dict_check['pat_nr'][n])
                for i, s in enumerate(self.list_structure):
                    if dict_check[s][n] == '':
                        ws.write(i+1, n+1, dict_check[s][n])
                    elif dict_check[s][n]: #if structure exists in RS
                        ws.write(i+1, n+1, dict_check[s][n], formatTrue)
                    elif not dict_check[s][n]: #if structure exists in RS
                        ws.write(i+1, n+1, dict_check[s][n], formatFalse)
                for ri, r in enumerate(dict_check['ROI_rs'][n]):
                    ws.write(ri+len(self.list_structure)+1, n+1, r)            
            
            wb.close()
        except IOError:
            info= 'Close the ROI_check.xlsx file and rerun the check'
            MyException(info)