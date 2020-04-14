# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import pandas as pd

class ExportExcel(object):
    '''combines the txt files for texture and shape into one excel file'''
    def __init__(self, ifshape, path_save, name_shape_pt, start, stop, save_as):
        
        if ifshape:
            shape = pd.read_csv(path_save + '\\shape_'+name_shape_pt+'_'+str(start)+'_'+str(stop-1)+'.txt', 
                                sep="\t", header=0, index_col=False)
            shape.set_index("patient", inplace=True)
        else:
            shape = ''
        texture = pd.read_csv(path_save + "\\" + save_as+'.txt', sep="\t", header=0, index_col=False)
        texture.set_index("patient", inplace=True)
        if ifshape:
            df = shape.join(texture)
        else:
            df = texture
        df = self.cleanup(df)
        df = self.reorder(df)
        path = path_save + "\\" + save_as+'.xlsx'
        df.to_excel(path, index=True, header=True, sheet_name='radiomics') 
        
    def cleanup(self, df):
        def replaceList(df, featName):
            for i in range(len(df.index)):
                if type(df[featName][df.index[i]]) == str:
                    df[featName][df.index[i]] = float(df[featName][df.index[i]][1:df[featName][df.index[i]].find(",")])
                else:
                    df[featName][df.index[i]] = np.nan
            return df
        
        # clean up shape
        delShape = ["nonzero_Points", "Clusters"]
        df = df.iloc[:,~df.columns.str.contains('|'.join(delShape))]
        df = replaceList(df, "voxels")
        df = replaceList(df, "vmin")
        df = replaceList(df, "vmax")
        # clean up texture
        wavelet_types = ['HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL']
        delete_names = [ 'fractal_dim','center_mass_shift','MTV20%', 'MTV30%', 'MTV40%', 'MTV50%', 'MTV60%', 'MTV70%']#'vmin', 'vmax', 'organ', 'voxels',
        for iDeleteCol in delete_names:
            delWavelets = [s + "_" + iDeleteCol for s in wavelet_types]
            df = df.iloc[:,~df.columns.str.contains('|'.join(delWavelets))]
        df.rename(columns={"Initial_fractal_dim": "fractal_dim", 
                            "Initial_center_mass_shift": 'center_mass_shift',
                            'Initial_MTV20%': "MTV20%", 
                            'Initial_MTV30%' :'MTV30%', 
                            'Initial_MTV40%': 'MTV40%', 
                            'Initial_MTV50%': 'MTV50%', 
                            'Initial_MTV60%': 'MTV60%', 
                            'Initial_MTV70%': 'MTV70%'}, inplace=True)
        return df
    
    def reorder(self, df):
        df.insert(0,'organ', df.pop("organ"))
        df.insert(1,'voxels', df.pop("voxels"))
        df.insert(2,'vmin', df.pop("vmin"))
        df.insert(3,'vmax', df.pop("vmax"))
        # fractal dims and center of mass shift
        df.insert(20,'fractal_dim', df.pop("fractal_dim"))
        df.insert(21,'center_mass_shift', df.pop("center_mass_shift"))
        df.insert(22,'MTV20%', df.pop("MTV20%"))
        df.insert(23,'MTV30%', df.pop("MTV30%"))
        df.insert(24,'MTV40%', df.pop("MTV40%"))
        df.insert(25,'MTV50%', df.pop("MTV50%"))
        df.insert(26,'MTV60%', df.pop("MTV60%"))
        df.insert(27,'MTV70%', df.pop("MTV70%"))
        
        return df