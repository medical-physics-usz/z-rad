import os

import pandas as pd


class ExportExcel(object):
    """combines the txt files for texture and shape into one excel file"""

    def __init__(self, ifshape, path_save, save_as, dict_parameters):
        texture = pd.read_csv(path_save + 'texture_' + save_as + '.txt', sep="\t", header=0, index_col=False)
        if ifshape:
            shape = pd.read_csv(path_save + 'shape_' + save_as + '.csv', index_col=0)
            texture['patient'] = texture['patient'].astype(str)
            shape['patient'] = shape['patient'].astype(str)
            df = shape.merge(texture, on=['patient', 'organ'], how='outer')
        else:
            df = texture
        df = self.cleanup(df)
        df = self.reorder(df)
        df = df.sort_values(['patient', 'organ']).reset_index(drop=True)

        path = path_save + os.sep + save_as + '.xlsx'
        df_parameters = pd.DataFrame.from_dict(dict_parameters)
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, index=True, header=True, sheet_name="radiomics")
            df_parameters.to_excel(writer, index=False, header=True, sheet_name="parameters")

    def cleanup(self, df):
        # clean up shape
        delShape = ["nonzero_Points", "Clusters"]
        df = df.iloc[:, ~df.columns.str.contains('|'.join(delShape))]
        pd.options.mode.chained_assignment = None  # default='warn'
        for featName in ['voxels', 'vmin', 'vmax']:
            df[featName] = df[featName].apply(lambda x: eval(x)[0] if isinstance(x, str) and eval(x) else '')
        pd.options.mode.chained_assignment = 'warn'

        # clean up texture
        wavelet_types = ['HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL']
        delete_names = ['fractal_dim', 'center_mass_shift', 'MaxIntensityTumorVolume20%', 'MaxIntensityTumorVolume30%', 
                        'MaxIntensityTumorVolume40%', 'MaxIntensityTumorVolume50%', 'MaxIntensityTumorVolume60%', 'MaxIntensityTumorVolume70%']  # 'vmin', 'vmax', 'organ', 'voxels',
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
        df.insert(23, 'MaxIntensityTumorVolume20%', df.pop("MaxIntensityTumorVolume20%"))
        df.insert(24, 'MaxIntensityTumorVolume30%', df.pop("MaxIntensityTumorVolume30%"))
        df.insert(25, 'MaxIntensityTumorVolume40%', df.pop("MaxIntensityTumorVolume40%"))
        df.insert(26, 'MaxIntensityTumorVolume50%', df.pop("MaxIntensityTumorVolume50%"))
        df.insert(27, 'MaxIntensityTumorVolume60%', df.pop("MaxIntensityTumorVolume60%"))
        df.insert(28, 'MaxIntensityTumorVolume70%', df.pop("MaxIntensityTumorVolume70%"))
        df.insert(28, 'vmin', df.pop("vmin"))
        df.insert(28, 'vmax', df.pop("vmax"))
        return df
