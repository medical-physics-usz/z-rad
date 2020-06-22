import os

import numpy as np
import pandas as pd


class ExportExcel(object):
    """combines the txt files for texture and shape into one excel file"""

    def __init__(self, ifshape, path_save, save_as):
        if ifshape:
            shape = pd.read_csv(path_save + 'shape_' + save_as + '.csv', index_col=0)
        else:
            shape = ''
        texture = pd.read_csv(path_save + 'texture_' + save_as + '.txt', sep="\t", header=0, index_col=False)
        if ifshape:
            df = shape.merge(texture, on=['patient', 'organ'], how='outer')
        else:
            df = texture
        df = self.cleanup(df)
        df = self.reorder(df)
        path = path_save + os.sep + save_as + '.xlsx'
        df.to_excel(path, sheet_name='radiomics')

    def cleanup(self, df):
        # change MCC features to real numbers
        MCC_features = df.columns[df.columns.str.contains("MCC")]
        for iMCC in MCC_features:
            df[iMCC] = df.loc[:, iMCC].apply(complex).apply(np.real)

        # clean up shape
        delShape = ["nonzero_Points", "Clusters"]
        df = df.iloc[:, ~df.columns.str.contains('|'.join(delShape))]

        for featName in ['voxels', 'vmin', 'vmax']:
            df[featName] = df[featName].apply(lambda x: float(x[1:len(x)].split(",")[0]) if isinstance(x, str) else np.nan)

        # clean up texture
        wavelet_types = ['HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL']
        delete_names = ['fractal_dim', 'center_mass_shift', 'MTV20%', 'MTV30%', 'MTV40%', 'MTV50%', 'MTV60%',
                        'MTV70%']  # 'vmin', 'vmax', 'organ', 'voxels',
        for iDeleteCol in delete_names:
            delWavelets = [s + "_" + iDeleteCol for s in wavelet_types]
            df = df.iloc[:, ~df.columns.str.contains('|'.join(delWavelets))]
        return df

    def reorder(self, df):
        df.insert(1, 'organ', df.pop("organ"))
        df.insert(2, 'voxels', df.pop("voxels"))
        # fractal dims and center of mass shift
        df.insert(21, 'fractal_dim', df.pop("fractal_dim"))
        df.insert(22, 'center_mass_shift', df.pop("center_mass_shift"))
        df.insert(23, 'MTV20%', df.pop("MTV20%"))
        df.insert(24, 'MTV30%', df.pop("MTV30%"))
        df.insert(25, 'MTV40%', df.pop("MTV40%"))
        df.insert(26, 'MTV50%', df.pop("MTV50%"))
        df.insert(27, 'MTV60%', df.pop("MTV60%"))
        df.insert(28, 'MTV70%', df.pop("MTV70%"))
        df.insert(29, 'vmin', df.pop("vmin"))
        df.insert(30, 'vmax', df.pop("vmax"))

        return df
