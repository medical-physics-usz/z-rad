# -*- coding: utf-8 -*-
#import libraries
import pydicom as dc # dicom library
import numpy as np # numerical computation 
import pylab as py # drawing plots
import scipy.stats as st
from os import makedirs # managing files
from os.path import isdir

#own classes
#import class to calculate texture parameters
from texture import Texture
from read import ReadImageStructure
from export import Export

class main_texture_ctp(object):
    '''
    Main class to handle CTP images, reads images and structures, remove outliers (could be improved) and calls radiomics calculation and export class to export results
    Type: object
    Attributes: 
    sb – Status bar in the frame 
    path_image – path to the patients subfolders
    path_save – path to save radiomics results
    structure – list of structures to be analysed
    pixNr number of analyzed bins, if not specified  = none
    binSize – bin size for the analysis, if not specified = none
    l_ImName – list of patients subfolders (here are data to be analysed)
    save_as – name of text files to save the radiomics results
    Dim – string variable of value 2D or 3D for dimensionality of calculation
    outlier – bool, correct for outliers
    wv – bool, calculate wavelet  
    exportList – list of matrices/features to be calculated and exported
    '''
    def __init__(self, sb, path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, dim, outlier_corr,wv,local, cropStructure, exportList):
        final=[] # list with results
        image_modality = ['BV', 'MTT', 'BF']
        dicomProblem = []
        for ImName in l_ImName:
            print('patient', ImName)
            try:
                sb.SetStatusText('Load '+ImName)
                
                mypath_image = path_image+ImName+'\\'
                UID = ['CTP']
                
                read = ReadImageStructure(UID, mypath_image, structure, wv, dim, local, image_modality)  # none for dimension

                dicomProblem.append([ImName, read.listDicomProblem])   

                l_IM_matrix = [] # list containing different perfusion maps
                for m_name in np.arange(0, len(image_modality)):
                    IM_matrix = [] #list containing the images matrix
                    for f in read.onlyfiles[m_name]:
                        data = dc.read_file(mypath_image + f).PixelData
                        data16 = np.array(np.fromstring(data, dtype=np.float16)) #converitng to decimal
                        # recalculating for rows x columns
                        a = []
                        for j in np.arange(0, read.rows):#(0, rows):
                            a.append(data16[j*read.columns:(j+1)*read.columns])
                        a = np.array(a)
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)

                    l_IM_matrix.append(IM_matrix)

                #pre-processing - remove points outside 3 sigma
                points_remove = []
                if outlier_corr == True:
                    self.Xcontour = read.Xcontour
                    self.Xcontour_W = read.Xcontour_W
                    self.Ycontour = read.Ycontour
                    self.Ycontour_W = read.Ycontour_W
                    for m in np.arange(0, len(l_IM_matrix)):
                        p_remove, norm_points = self.MatrixRemove(l_IM_matrix[m], image_modality[m], ImName, path_save) #returns list of indicies for outliers and the number of points left with gray value != nan
                        points_remove.append(p_remove)
                    for m in np.arange(0, len(l_IM_matrix)):
                        l_IM_matrix[m] = self.remove_points(l_IM_matrix[m], points_remove) #place NaN in the place of outliers

                    del points_remove
                    del p_remove
                    del norm_points
                    
            except WindowsError: #error if there is not directory
                continue
            except IndexError: #empty folder
                continue

                #for m_names in arange(0, len(prefix)):
                    #Texture(arguments).ret() -> function for texture calculation
                    #arguments: image, stucture name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                    #function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
            sb.SetStatusText('Calculate '+ImName)
            stop_calc = '' #in case someting would be wrong with the image tags
            lista_results = Texture(sb,l_IM_matrix, read.structure_f, read.columns, read.rows, read.xCTspace, read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, cropStructure, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour, read.Ycontour_W).ret()

            #final list contains of the sublist for each patient, sublist contains of [patient number, structure used for calcualtions, list of texture parameters, number of points used for calculcations]
            final.append([ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]])
                
        #save the results
        Export(final, exportList, wv,path_save, save_as, image_modality, path_image)        
    
    
    def MatrixRemove(self, imap, perf, ImName, path):
        '''read the gray values in the ROI, remove the NaN and fir gaussian function '''
        v = [] #read gray values in ROI
        for i in np.arange(0, len(self.Xcontour)): #slices
            for j in np.arange(0, len(self.Xcontour[i])): #sub-structres in the slice
                for k in np.arange(0, len(self.Xcontour[i][j])):
                    v.append(imap[i][self.Ycontour[i][j][k]][self.Xcontour[i][j][k]])

        #remove NaN
        ind = np.where(np.isnan(np.array(v)))[0]
        for j in np.arange(1, len(ind)+1):
            v.pop(ind[-j])
        #fitting gaussian function and deleting ouliers.
        remove = self.histfit(v, 15, perf, ImName, path)
        p_remove=[]
        for j in np.arange(0, len(remove)):
            p_remove.append(np.where(np.array(imap) == remove[j])) #which indices corresponds to the values that should be removed
        return p_remove, len(v)

    def histfit(self,x,N_bins, name, ImName, path):
        ''' 
        x - data
        N_bins - number of bins in the histogram
         
        plot a histrogram and a guassian funcation with mean and SD calcualte from data
        '''
        
        x = np.array(x, dtype = np.float64)
        py.figure(figsize=(20,20))
        py.subplot(121)
        n, bins, patches = py.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75)     
     
        bincenters = 0.5*(bins[1:]+bins[:-1])
     
        y = st.norm.pdf(bincenters, loc = np.mean(x), scale = np.std(np.array(x)))
     
        py.plot(bincenters, y, 'r--', linewidth=1, label = 'std: '+str(round(np.std(x),2)))
        py.plot([np.mean(x)+3*np.std(x), np.mean(x)+3*np.std(x)], [0,0.1], 'b--')
        py.plot([np.mean(x)-3*np.std(x), np.mean(x)-3*np.std(x)], [0,0.1], 'b--')
        #check which values are outside the range of 3 sigma
        ind1 = np.where(np.array(x)>(np.mean(x)+3*np.std(x)))[0]
        ind2 = np.where(np.array(x)<(np.mean(x)-3*np.std(x)))[0]
        
        v = [] #outliers
        for i in ind1:
            v.append(x[i])
        for i in ind2:
            v.append(x[i])
        x= list(x)
        leg=str(len(v)*100./len(x))
        for i in v: #remove outlier from data vector
            x.remove(i)
        py.legend()
        #plot a histogram after the removal
        py.subplot(122)
        n, bins, patches = py.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75, label = 'removed points: '+leg)
        py.legend()
        #save the figure
        try:
            makedirs(path+'gauss\\')
        except OSError:
            if not isdir(path+'gauss\\'):
                raise
        py.savefig(path+'gauss\\'+'hist_'+name+'_'+ImName+'.png')
        py.close()
        return v #retruns outliers values

    def remove_points(self, M, p):
        '''insert NaN in the place of the ouliers'''
        for i in p:
            if i !=[]:
                for j in i:
                    for k in np.arange(0, len(j[0])):
                        M[j[0][k]][j[1][k]][j[2][k]] = np.nan
        return M

