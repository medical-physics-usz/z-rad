# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:42:53 2018

@author: vuodi
"""

from DicomReader import dcmReader
from glob import glob

def renamePatientName(dirDicom, newName):
    print(dirDicom + r"\\*dcm")
    files = glob(dirDicom + r"\\*dcm")
    
    for iFile in files:
        d = dcmReader(iFile)
        d.dcm.PatientName = newName
        d.dcm.PatientID = newName
        #d.dcm.save_as(iFile)
    
    
if __name__== "__main__":
    path = r"K:\RAO_Physik\Research\1_FUNCTIONAL IMAGING\7_SAKK_Lung_Study\TP1_CT_input"
    renamePatientName(path + "\\ccc_94 - Kopie", "new")