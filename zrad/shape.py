# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import vtk
import os.path
from vtk import *
from os import listdir, remove # managing files
from os.path import isfile, join, exists
##from mayavi import mlab
#from mayavi.mlab import triangular_mesh
from vtk.numpy_interface import dataset_adapter as dsa
import scipy
from scipy import spatial
import scipy.ndimage as ndi
#import mahotas
import matplotlib.cm as cm
import sys
from datetime import datetime

from sklearn.decomposition import PCA

class Shape(object):
    '''Calculate shape features
    Type: object
    Attributes:
    inp_mypath_load – path to text files with saved points inside ROI
    inp_mypath_results – save shape results to that path
    low - start number
    high - stop number
    '''
    def __init__(self, inp_mypath_load, inp_mypath_results, low, high):
        path_results = inp_mypath_results #'F:\\HN PET data\\validation_shapePT.txt'
        self.path_load = inp_mypath_load+"\\"
        #nlist = [52,66,74,77, 88,108,109,115,122,129,141,147,175,184,187,214,228,233,245,260,265]
        nlist =  [str(i) for i in np.arange(low, high+1)]#[4,5,13,20,32,35,47,48,54,59,60]

        #maximum euclidian distance, one needs 
        scalefactor = 0.3 #0.4 #0.1 => 8 s /file, 0.2 => 10 s/file, 0.3 => 1 min/file, 0.4 => MemoryOverload
        
        #in which resoltuion files were saved
        ind_f = inp_mypath_load[:-1].rfind('\\') #get into one folder up
        path_set = inp_mypath_load[:ind_f+1]
        f = open(path_set+'shape_resolution.txt', 'r')
        savedResolution = float(f.readlines()[0])
        f.close()
        del ind_f
        del path_set

        #calculate parameters of all GTVs
        if exists(path_results):
            remove(path_results)

        fGTV = open(path_results, "w") #use "a" to append
        fGTV.write("Name\tMC-Volume\tnonzero_Points\tMC-Surface\tClusters\tCompactness_1\tCompactness_2\tDispr.\tSphericity\tAsphericity\tA/V\tthickness_median\tthickness_SD\teuclidian_distance\tmajor_axis\tminor_axis\tleast_axis\telongation\tflatness" "\n")

        print('Start_0: ', datetime.now().strftime('%H:%M:%S'))

        for nn in nlist:
            try:
                if exists(self.path_load+nn) and not listdir(self.path_load+nn) == []: 
                    print('processing: ',nn + ', start: ', datetime.now().strftime('%H:%M:%S'))
                    pic3d, pnz, extent = self.fill(nn)
                    num_cluster = self.clust(pic3d,0)
                    dst_median,dst_std,dst_mean = self.thickness(nn,pic3d,0)
                    vtkvol, vtksur, comp1, comp2, ar, dispr, spher, aspher, AtoV = self.marching_cubes(pic3d,extent,0, len(pnz[0]))
                    
                    maxeucl = self.maxeuclid(pic3d,scalefactor)
                    major_axis, minor_axis, least_axis, elong, flat = self.PCA_analysis(pnz)#for big structres self.PCA_analysis(pic3d, scalefactor)
                    if savedResolution != 1.0: #to adapt for the resolution of readin points
                        vtkvol = 0.001*vtkvol
                        vtksur = 0.01*vtksur
                        dst_median = 0.1*dst_median
                        dst_std = 0.1*dst_std
                        maxeucl = 0.1*maxeucl
                        major_axis = 0.1*major_axis
                        minor_axis = 0.1*minor_axis
                        least_axis = 0.1*least_axis
                                               
                    fGTV.write(nn + "\t" + str(vtkvol) + "\t" + str(len(pnz[0])) + "\t" + str(vtksur) + "\t" + str(num_cluster) + "\t"+ str(comp1) +"\t" + str(comp2) + "\t" + str(dispr) +"\t"+ str(spher)+ "\t"+ str(aspher) + "\t"+ str(AtoV) +'\t' )
                    fGTV.write(str(dst_median)+ '\t' +str(dst_std)+'\t')
                    fGTV.write(str(maxeucl)+ '\t')
                    fGTV.write(str(major_axis) + '\t' +str(minor_axis)+ '\t' + str(least_axis)+ '\t' + str(elong) + '\t' +str(flat) + '\t')
                    fGTV.write("\n")
                    sys.stdout.flush()
                else:
                    print('directory %s does not exist or is empty'%(nn))
            except WindowsError:
                pass

        fGTV.close()

    def fill(self, fname):
        #Start with empty 3d-array of zeros
        width = 600
        files = [f for f in listdir(self.path_load+fname) if isfile(join(self.path_load+fname,f))]
        #what slices are in the files
        l_slices = []
        for z in files:
            l_slices.append(int(z[6:]))
        pic3d = np.zeros([width,width,(max(l_slices)-min(l_slices)+10)],dtype=np.int8)
        #fill the 3d-array with ones inside the contour
        for z in range(min(l_slices),max(l_slices)+1):
            try:
                zi = z - min(l_slices)
                data = np.loadtxt(self.path_load+fname+'\\slice_'+str(z))
                data = data.astype(np.int32)
                try:
                    rangex = len(data[:,1])
                except IndexError:
                    if len(data) == 2:
                        rangex = 0
                        xp = data[0]
                        yp = data[1]
                        pic3d[xp+3][yp+3][zi+3]=1
                    elif len(data) == 0: #empty file, for example for lymph node if there is a break
                        rangex = 0
                    else:
                        raise IndexError
                for x in range (0,rangex):
                    xp = data[x,0]
                    yp = data[x,1]
                    pic3d[xp+3][yp+3][zi+3]=1
            except IOError:
                pass

        
        # select the cuboid subarray with nonzero elements
        x, y, z = np.nonzero(pic3d)
        xmi = x.min()-2
        ymi = y.min()-2
        zmi = z.min()-2
        xma = x.max()+2
        yma = y.max()+2
        zma = z.max()+2
        
        pic3d = pic3d[xmi:xma, ymi:yma, zmi:zma]
        pic3d[3]
        pnz = np.nonzero(pic3d)
     
        print('Number of nonzero points = ',len(pnz[0]))
        #extent for vtk - analysis:
        extent = (0, zma-zmi-1, 0, yma-ymi-1, 0, xma-xmi-1)
        return pic3d, pnz, extent

    def maxeuclid(self, vol,n_e):
        #Parameter 17, Maximum euclidean 3D-Distance (max. 3D-diameter)
        #scaling of the array with factor n_e necessary to avoid a too large array
        try:
            psmall = ndi.interpolation.zoom(vol,n_e)
    #        print psmall
            psmall[np.where(psmall>0.0001)]=1
            psmall[np.where(psmall<0.0001)]=0
            pnzs = np.nonzero(psmall)
            D = spatial.distance.pdist(np.transpose(pnzs),'euclidean')    #creates the condensed distance matrix
            D.sort()
            maxeucl = np.max(D)/n_e                                       #takes the max and rescales it
        except ValueError:
            psmall = vol
            pnzs = np.nonzero(psmall)
            D = spatial.distance.pdist(np.transpose(pnzs),'euclidean')    #creates the condensed distance matrix
            maxeucl = np.max(D)   
    #        
        print('Longest euclidean distance = ',np.round(maxeucl))
        sys.stdout.flush()
        return maxeucl
        
    def clust(self, vol,rend):    #Number of Clusters
        s = ndi.generate_binary_structure(3,3)
        la, num_features = ndi.label(vol, structure=s)#la = labeled_array
        print("Number of Clusters:", num_features)
        
        #Coloring of the Clusters
    ##    la1 = la>0 #
    ##    z,x,y =la.nonzero() #indices of nonzero entries
    ##    la1s = la[la1] #array with values of nonzero entries
    ##    dtval, counts = np.unique(la1s, return_counts = True)
        #print(dtval)
        #print(counts)
    ##    if rend == 1:
    ##        mlab.points3d(x,y,z,la1s,mode = 'point', colormap = 'Blues',scale_factor = 1,vmin = 0)#ohne scale_factor wird kleinster Wert = 0 Durchmesser
    ##        mlab.show()
        return num_features
        
    def thickness(self,fnm,vol,rend):
        # Distance Transform to determine average thickness
        #try:
        #    remove('/users/xwuerms/documents/Thesis/Results/Thickness_%s.txt'%fnm)
        #except OsError:
        #    pass
    ##    fth = open(path_results+'Thickness_%s.txt'%fnm, 'w')
        dtpoints = scipy.ndimage.morphology.distance_transform_edt(vol)
        dtpflat = np.ndarray.flatten(dtpoints[np.where(dtpoints != 0)])
        dst_mean = np.mean(dtpflat)
        dst_std = np.std(dtpflat)
        dst_median = np.median(dtpflat)
    #    print(dst_median,dst_std,dst_mean)
        
    ##    dtval, counts = np.unique(dtpflat, return_counts = True)
        #print(dtval)
        #print(counts)
        
        if rend == 1:
            fig = plt.figure(1)
            ax = fig.add_subplot(211)
            plt.xlim(0,15)
            #plt.hist(dtpflat,dtval)
            ax.bar(dtval,counts, width = 0.1)
            ax = fig.add_subplot(212)
            plt.xlim(0,15)
            plt.ylim(0.8,1.2)
            ax.axes.get_yaxis().set_ticks([])
            bp = ax.boxplot(dtpflat, patch_artist = True, manage_xticks = False,showfliers = False, vert = False, notch = True)
            for box in bp['boxes']:
                box.set( color='#7570b3', linewidth=2)
                box.set( facecolor = '#1b9e77' )
            for whisker in bp['whiskers']:
                whisker.set(color='#7570b3', linewidth=2)
            for cap in bp['caps']:
                cap.set(color='#7570b3', linewidth=2)
            for median in bp['medians']:
                median.set(color='#b2df8a', linewidth=2)
            for flier in bp['fliers']:
                flier.set(marker='o', color='#e7298a', alpha=0.5)
            plt.show()
        
    ##    fth.write(str(dtval) + "\n" + str(counts))
    ##    fth.close()
        return dst_median,dst_std,dst_mean

    def marching_cubes(self, vol,extent,rend, pnz):
        dataImporter = vtk.vtkImageImport()
        data_string = vol.tostring()
        
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataSpacing(1,1,1)
        dataImporter.SetDataExtent(extent)
        dataImporter.SetWholeExtent(extent)
        isoSurface = vtkMarchingCubes()
        #isoSurface = vtkDiscreteMarchingCubes() # works, looks more like scipy.marching cubes
        isoSurface.SetInputConnection(dataImporter.GetOutputPort())
        isoSurface.SetValue(0,1)
        
        # Have VTK calculate the Mass (volume) and surface area
        Mass = vtk.vtkMassProperties()
        Mass.SetInputConnection(isoSurface.GetOutputPort())
        Mass.Update()
            
        vtkvol = Mass.GetVolume()/1000.			#Parameter Nr. 22 
        vtksur = Mass.GetSurfaceArea()/100.	 		#Parameter Nr. 20

        #adaptation for Oncoray
##        vol = vtkvol
##        vtkvol = pnz
##        vtksur = vtksur*100
        
        comp1 = vtkvol/(np.pi**(1/2.)*vtksur**(3/2.))	#Parameter Nr. 15
        comp2 = 36*np.pi*vtkvol**2./(vtksur**3.)		#Parameter Nr. 16
        ar = (vtkvol/(4/3.*np.pi))**(1/3.)			#Radius of equiv. sphere
        dispr = vtksur/(4.*np.pi*ar**2.)			#Parameter Nr. 18
        spher = np.pi**(1/3.)*(6.*vtkvol)**(2./3.)/vtksur	#Parameter Nr. 19
        aspher = (vtksur**3/(36 *np.pi*vtkvol**2))**(1./3)-1
        AtoV = vtksur/vtkvol				#Parameter Nr. 21

##        vtkvol = vol*1000
        
        #    print "VTK-MC-Volume = ",vtkvol 
        #    print'Number of Non-Zero points = ',len(pnz[0])
        #    print "VTK-MC-Surface = ",vtksur


        print((vtkvol, vtksur, comp1, comp2, ar, dispr, spher, AtoV))
        sys.stdout.flush()

        # 3d-Render-Block
        if rend == 1:
            surfaceMapper = vtkPolyDataMapper()
            surfaceMapper.SetInputConnection(isoSurface.GetOutputPort())    
            surfaceActor = vtkActor()
            #surfaceMapper.ScalarVisibilityOff()
            surfaceActor.SetMapper(surfaceMapper)    
            outlineData = vtkOutlineFilter()
            outlineData.SetInputConnection(dataImporter.GetOutputPort())    
            outlineMapper = vtkPolyDataMapper()
            outlineMapper.SetInputConnection(outlineData.GetOutputPort())    
            outline = vtkActor()
            outline.SetMapper(outlineMapper)
            outline.GetProperty().SetColor(0.5, 0.5, 0.5)
            camera = vtkCamera()
            camera.SetViewUp(0, 0, -1)
            camera.SetPosition(-2, -2, -2)
            ren = vtkRenderer()
            ren.AddActor(outline)
            ren.AddActor(surfaceActor)
            ren.SetBackground(0.2, 0.2, 0.2)
            ren.SetActiveCamera(camera)
            ren.ResetCamera()    
            renWin = vtkRenderWindow()
            renWin.AddRenderer(ren)
            renWin.SetWindowName("IsoSurface/MarchingCubes")
            renWin.SetSize(700, 700);    
            iren = vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)
            iren.Initialize()
            iren.Start()
        
        return vtkvol, vtksur, comp1, comp2, ar, dispr, spher, aspher, AtoV

    def PCA_analysis(self, pnz):
##        uncomment for big structures, change the function to PCA_analysis(self, vol, n_e)
##        pnz = ndi.interpolation.zoom(vol,n_e)
##    #        print psmall
##        pnz[np.where(pnz>0.0001)]=1
##        pnz[np.where(pnz<0.0001)]=0
##
##        print pnz.shape
##
##        pnz = np.nonzero(pnz)
        
        pca = PCA()
        pca.fit(np.transpose(pnz))
        eigen_value = pca.explained_variance_
        eigen_value.sort()
        major_axis = 4*eigen_value[2]**0.5 
        minor_axis = 4*eigen_value[1]**0.5
        least_axis = 4*eigen_value[0]**0.5 
        elong = minor_axis/major_axis
        flat = least_axis/major_axis
        print('axis', major_axis, minor_axis, least_axis)
        return major_axis, minor_axis, least_axis, elong, flat

    ###################
    ###################




    #calculate Maximum euclidean distance for all GTVs separately

    ##if exists('/users/xwuerms/documents/Thesis/Results/Results_MaxEucl_%s.txt'%scalefactor):
    ##    remove('/users/xwuerms/documents/Thesis/Results/Results_MaxEucl_%s.txt'%scalefactor)
    ##
    ##fMECL = open("/users/xwuerms/documents/Thesis/Results/Results_MaxEucl_%s.txt"%scalefactor, "w") #use "a" to append
    ##fMECL.write('Name, maxeucl, pnz, ' +'\n')
    ##
    ##for nn in ['5','10','11','12','13','17','19']:
    ##    for ak in ['AC','KM']:
    ##        for aa in ['GTV_xw','GTV_sg','GTV_jr','GTV_le','LS','LH']: 
    ###for nn in ['5']:
    ###    for ak in ['AC']:
    ###        for aa in ['xw']:      
    ##            fname = "M%s_%s_%s"%(nn,ak,aa)
    ##            if exists('/users/xwuerms/documents/Thesis/contoursM/%s/'%(fname)):
    ##                print 'processing: ',fname + ', start: ', datetime.now().strftime('%H:%M:%S')
    ##                pic3d, pnz, extent = fill(fname)
    ##                maxeucl = maxeuclid(pic3d,scalefactor)
    ##                fMECL.write(fname + ", " +  str(maxeucl)+ ", " + str(len(pnz[0])))
    ##                fMECL.write("\n")
    ##                sys.stdout.flush()
    ##            else: 
    ##                print 'directory %s does not exist'%(fname)


