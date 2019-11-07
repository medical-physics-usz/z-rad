# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:17:28 2018

@author: Physics
"""

from numpy import loadtxt, arange
import numpy as np
import pandas as pd

class ExportExcel(object):
    '''combines the txt files for texture and shape into one excel file'''
    def __init__(self, ifshape, path_save, name_shape_pt, start, stop, save_as):
        
        if ifshape:
            shape = open(path_save + '\\shape_'+name_shape_pt+'_'+str(start)+'_'+str(stop-1)+'.txt', 'r')
        else:
            shape = ''
        
        texture = open(path_save+save_as+'.txt', 'r')
        
        path = path_save+save_as+'.xlsx'
        
        combine_matrix = self.createMatrix(shape, texture)
        combine_matrix = self.select(combine_matrix, shape)
        self.saveExcel(combine_matrix, path)
        
    def createMatrix(self, shape, texture):
        lines = texture.readlines()
        
        l1 = lines[1].split('\t')
        l1.pop(-1)
        matrix = [l1]
        for l in lines[2:]:
            elements = l.split('\t')
            e = elements[2].split(',')[0]
            elements[2] = e[1:]
            e = elements[3].split(',')[0]
            elements[3] = e[1:]
            e = elements[4].split(',')[0]
            elements[4] = e[1:]
            elements.pop(-1)
            for ei, e in enumerate(elements): #check for the complex numbers which may come from the MCC
                if '(' in e:
                    if round(complex(e).imag,3) == 0:
                        elements[ei] = str(complex(e).real)
                    else:
                        elements[ei] = ''
            matrix.append(elements)
          
        matrix = np.array(matrix)
        
        if shape != '':       
            lines = shape.readlines()
            
            l1 = lines[0].split('\t')
            l1[-1] = l1[-1][:-1]
            matrix_shape = [l1]
            for l in lines[1:]:
                elements = l.split('\t')
                elements.pop(-1)
                matrix_shape.append(elements)
              
            matrix_shape = np.array(matrix_shape)
        
            #find union of indexes
            pat_id = set(matrix_shape[1:,0])
            pat_id.union(set(matrix[1:,0]))
            
            pat_id = np.array(list(pat_id), dtype = np.int64)
            pat_id = list(pat_id)
            pat_id.sort()
            
            #combine matricies
            combine_matrix = np.zeros(((len(pat_id)+1), (len(matrix[0])+len(matrix_shape[0])-1)), dtype = '|S30')
            combine_matrix[0] = list(matrix[0,0:2])+[matrix[0][4]]+list(matrix_shape[0,1:])+list(matrix[0,2:4])+list(matrix[0,5:])
            
            
            for i, pid in enumerate(pat_id):
                ind_t = np.where(matrix[:,0]==str(pid))[0]
                ind_s = np.where(matrix_shape[:,0]==str(pid))[0]
                combine_matrix[i+1][0] = str(pid)
                if len(ind_s)==1:
                    combine_matrix[i+1,3:len(matrix_shape[0])+2] = matrix_shape[ind_s[0],1:]
                if len(ind_t)==1:
                    combine_matrix[i+1, 1] = matrix[ind_t[0], 1]
                    combine_matrix[i+1, 2] = matrix[ind_t[0], 4]
                    combine_matrix[i+1, len(matrix_shape[0])+2:] = list(matrix[ind_t[0],2:4])+list(matrix[ind_t[0],5:])
                    
        else:
            pat_id = np.array(list(matrix[1:,0]), dtype = np.int64)
            pat_id = list(pat_id)
            pat_id.sort()
            
            #combine matricies
            combine_matrix = np.zeros(((len(pat_id)+1), len(matrix[0])), dtype = '|S30')
            combine_matrix[0] = list(matrix[0,0:2])+[matrix[0][4]]+list(matrix[0,2:4])+list(matrix[0,5:])
            
            
            for i, pid in enumerate(pat_id):
                ind_t = np.where(matrix[:,0]==str(pid))[0]
                combine_matrix[i+1][0] = str(pid)
                if len(ind_t)==1:
                    combine_matrix[i+1, 1] = matrix[ind_t[0], 1]
                    combine_matrix[i+1, 2] = matrix[ind_t[0], 4]
                    combine_matrix[i+1, 3:] = list(matrix[ind_t[0],2:4])+list(matrix[ind_t[0],5:])

                
        ind_empty = np.where(combine_matrix == b'')
        combine_matrix[ind_empty] = np.nan
        ind_empty = np.where(combine_matrix == b"''")
        combine_matrix[ind_empty] = np.nan
        ind_bracket = np.where(combine_matrix == b"[")
        combine_matrix[ind_bracket] = np.nan
        ind_bracket = np.where(combine_matrix == b"]")
        combine_matrix[ind_bracket] = np.nan
        ind_nan = np.where(combine_matrix == b'nan')
        combine_matrix[ind_nan] = np.nan
        ind_inf = np.where(combine_matrix == b'inf')
        combine_matrix[ind_inf] = np.nan
        ind_inf = np.where(combine_matrix == b'-inf')
        combine_matrix[ind_inf] = np.nan
        
        return combine_matrix
                
    def select(self, combine_matrix, shape):
        delete_names = [ b'fractal_dim',b'center_mass_shift', b'MTV20%', b'MTV30%', b'MTV40%', b'MTV50%', b'MTV60%', b'MTV70%']#'vmin', 'vmax', 'organ', 'voxels',
        
        ind_delete = []
        for dn in delete_names:
            ind_delete = np.concatenate((ind_delete, np.where(combine_matrix[0]==dn)[0]))
        
        ind_delete = list(np.array(ind_delete, dtype = np.int))
        if shape != '':
            ind_delete = ind_delete + [4,6]
        ind_delete.sort()
        ind_delete.reverse()
        
        #safe features
        ind_save = []
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV20%')[0][0]))
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV30%')[0][0]))
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV40%')[0][0]))
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV50%')[0][0]))
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV60%')[0][0]))
        ind_save.append(int(np.where(combine_matrix[0]==b'MTV70%')[0][0]))
        
        for inds in ind_save:
            ind_delete.remove(inds)
        
        
        if shape != '':
            ind_save = []
            ind_save.append(int(np.where(combine_matrix[0]==b'fractal_dim')[0][0]))
            ind_save.append(int(np.where(combine_matrix[0]==b'center_mass_shift')[0][0]))
            
            for inds in ind_save:
                ind_delete.remove(inds)
        
        combine_matrix = np.delete(combine_matrix, ind_delete, 1)
        
        #reorginize
        if shape != '':
            ind_copy = int(np.where(combine_matrix[0]==b'fractal_dim')[0][0])
            ind_text = int(np.where(combine_matrix[0]==b'vmin')[0][0])
            
            temp = combine_matrix[:,ind_copy:ind_copy+8].copy()
            combine_matrix[:,(ind_text+8):(ind_copy+8)] = combine_matrix[:,ind_text:ind_copy]
            combine_matrix[:,ind_text:(ind_text+8)] = temp
        else:
            ind_copy = int(np.where(combine_matrix[0]==b'MTV20%')[0][0])
            ind_text = int(np.where(combine_matrix[0]==b'vmin')[0][0])
            
            temp = combine_matrix[:,ind_copy:ind_copy+6].copy()
            combine_matrix[:,(ind_text+6):(ind_copy+6)] = combine_matrix[:,ind_text:ind_copy]
            combine_matrix[:,ind_text:(ind_text+6)] = temp
              
        return combine_matrix
                
    def saveExcel(self, combine_matrix, path):
        ## convert your array into a dataframe
        
        names = list(combine_matrix[1:,1])
        combine_matrix[1:,1] = 0
        
        columns = list(combine_matrix[0])
        combine_matrix[0] = 0

        
        df = pd.DataFrame(combine_matrix, dtype = np.float64)
        
        
        df.loc[0,0:len(columns)] = columns
        df.loc[1:len(names),1] = names
                       
        
        df.to_excel(path, index=False, header = False, sheet_name = 'radiomics') 

                
                
                