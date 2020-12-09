import os
from os import listdir, makedirs
from os.path import isfile, join, isdir

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import vtk
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    from sklearn.metrics import calinski_harabaz_score
except ImportError:
    from sklearn.metrics import calinski_harabasz_score


class LymphNodes(object):
    """
    Calculate features related to distribution of lymph nodes around the primary tumor
    Attributes: 
    name_ln – name of LN for example gLN would then calculate features for gLN_x  
    name_sh – name of the PT shape structure
    path_files_shape – path where shape files are saved
    path_save – path to save final txt file
    path_save_dir – path to save produced plots for minimum distance distribution and clusters
    start – patient number to start
    stop – patient number to stop
    """

    def __init__(self, name_ln, name_sh, path_files_shape, path_save, path_save_dir, start, stop):
        final = []  # list with results
        for ImName in range(start, stop + 1):
            ImName = str(ImName)
            print('patient', ImName)
            tumorMass, ptVol, ptPoints = self.FindMassCenter(name_sh, path_files_shape,
                                                             ImName)  # calculates the tumor center of the mass
            print('PT', ptVol)
            if tumorMass == "":  # no such a file or directory
                final.append((ImName, "", "", "", "", "", "", "", "", ""))
                continue
            # returns list of lm center of the mass, volumes and all LN points
            listLnMass, listLnVol, listLnPoints = self.FindLN(name_ln, path_files_shape, ImName)
            if len(listLnMass) == 0:  # if no LN
                final.append((ImName, "", "", "", "", "", "", "", "", ""))
                continue
            # calculate features
            largestDist, meanDist, sumDist, distances = self.DistanceToTumor(tumorMass, listLnMass)
            weightedDist = self.WeightDist(sumDist, listLnMass)
            massPtLn = self.MassPtLn(ptPoints, listLnPoints, tumorMass)
            volLargestWeightedDist, volMeanWeightedDist, normVolLargestWeightedDist, normVolMeanWeightedDist = self.VolWeightedDist(
                distances, listLnVol, ptVol)
            minDistMean, minDistVar, triPoints = self.SmallestDist(tumorMass, listLnMass, path_save_dir, ImName)
            nrClusters = self.Clusters(triPoints, path_save_dir, ImName)
            elong, flat = self.PCAcharacteristic(listLnPoints, ptPoints)
            # add features to the list
            final.append((ImName, largestDist, meanDist, sumDist, weightedDist, volLargestWeightedDist,
                          volMeanWeightedDist, normVolLargestWeightedDist, normVolMeanWeightedDist, massPtLn,
                          minDistMean, minDistVar, nrClusters, elong, flat))
        # save final file
        self.Save(final, path_save)

    def FindMassCenter(self, name_sh, path_files_shape, ImName):
        """calculates the tumor center of the mass"""
        try:
            files = [f for f in listdir(path_files_shape + name_sh + os.sep + ImName) if
                     isfile(join(path_files_shape + name_sh + os.sep + ImName, f))]
            print(path_files_shape + name_sh + os.sep + ImName)

            # what slices are in the files
            l_slices = []
            for z in files:
                l_slices.append(int(z[6:]))  # files are called slice_X where X is a slice number, so l_slice is a list containing slice numbers.
            width = 1000  # everythng is resized to 1mm so tzpically 700 voxels would be enough
            print(max(l_slices))
            pic3d = np.zeros([max(l_slices) + 10, width, width], dtype=np.uint8)
            # fill the 3d-array with ones inside the contour
            for z in range(min(l_slices), max(l_slices) + 1):
                try:
                    data = np.loadtxt(
                        path_files_shape + name_sh + os.sep + ImName + os.sep + 'slice_' + str(z))  # if the file exists
                    data = data.astype(np.int32)
                    try:
                        rangex = len(data[:, 1])  # if only one voxel in a slice
                    except IndexError:
                        if len(data) == 2:
                            rangex = 0
                            xp = data[0]
                            yp = data[1]
                            pic3d[z][yp][xp] = 1
                        elif len(data) == 0:  # empty file, for example for lymph node if there is a break
                            rangex = 0
                        else:
                            raise IndexError
                    for x in range(0, rangex):
                        xp = data[x, 0]
                        yp = data[x, 1]
                        pic3d[z][yp][xp] = 1
                except IOError:  # if the files does not exists
                    pass
            # calcuate center of the mass
            ind = list(np.where(pic3d == 1))  # voxel size 1mm
            print(ind[0])
            print(ind[1])
            print(ind[2])
            MassCenter = np.array([np.sum(ind[0]), np.sum(ind[1]), np.sum(ind[2])])
            MassCenter = MassCenter / float(len(ind[0]))

            # calculate volume
            w, d, h = pic3d.shape
            extent = (0, h - 1, 0, d - 1, 0, w - 1)
            dataImporter = vtk.vtkImageImport()
            data_string = pic3d.tostring()

            dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            dataImporter.SetDataScalarTypeToUnsignedChar()
            dataImporter.SetNumberOfScalarComponents(1)
            dataImporter.SetDataSpacing(1, 1, 1)
            dataImporter.SetDataExtent(extent)
            dataImporter.SetWholeExtent(extent)
            isoSurface = vtk.vtkMarchingCubes()
            isoSurface.SetInputConnection(dataImporter.GetOutputPort())
            isoSurface.SetValue(0, 1)

            # Have VTK calculate the Mass (volume) 
            Mass = vtk.vtkMassProperties()
            Mass.SetInputConnection(isoSurface.GetOutputPort())
            Mass.Update()

            volume = Mass.GetVolume()

        except OSError:  # no such a directory (in case this patient has less LN than the others)
            MassCenter = ''
            volume = ''
            ind = ''

        return MassCenter, volume, ind  # ind - all the points

    def FindLN(self, name_ln, path_files_shape, ImName):  # returns list of lm center of the mass
        """iterate through all LN folders and find their center of mass"""
        onlydirs = [f for f in listdir(path_files_shape) if f.startswith(name_ln) and isdir(
            join(path_files_shape, f))]  # find only folder corresponding to lymph nodes
        listLnMass = []
        listLnPoints = []
        listLnVol = []
        for dire in onlydirs:
            m, vol, p = self.FindMassCenter(dire, path_files_shape, ImName)
            listLnMass.append(m)
            listLnPoints.append(p)
            listLnVol.append(vol)
        # remove "" corresponing to nonexcistig files
        remove = True

        while remove:
            try:
                listLnMass.remove("")
                listLnPoints.remove("")
                listLnVol.remove("")
            except ValueError:
                remove = False
        return listLnMass, listLnVol, listLnPoints

    def DistanceToTumor(self, tumorMass, listLnMass):
        """distance of LN to PT"""
        distance = []
        for ln in listLnMass:
            dist = np.sum((tumorMass - ln) ** 2)
            distance.append(np.sqrt(float(dist)))
        largestDist = np.max(distance)
        meanDist = np.mean(distance)
        sumDist = np.sum(distance)

        return largestDist, meanDist, sumDist, distance

    def WeightDist(self, sumDist, listLnMass):
        """sum of PT and LN distances normalized by the sum of distances between the LN"""
        listSumDist = []
        if len(listLnMass) > 1:  # if more than one LN
            for mainLN in listLnMass:  # distances between LNs
                distance = []
                for ln in listLnMass:
                    dist = np.sum((mainLN - ln) ** 2)
                    distance.append(np.sqrt(float(dist)))
                listSumDist.append(np.sum(distance) / (
                            len(listLnMass) - 1))  # divide by the number of LN -1 to noramlize the sum of distances
            weightedDist = sumDist / np.sum(listSumDist)
        else:
            weightedDist = 1.

        return weightedDist

    def MassPtLn(self, ptPoints, listLnPoints, massCenterPt):
        """distance between tumor center of the mass and center of the mass of tumor plus LN"""
        for ln in listLnPoints:
            ptPoints[0] = np.concatenate((ptPoints[0], ln[0]))
            ptPoints[1] = np.concatenate((ptPoints[1], ln[1]))
            ptPoints[2] = np.concatenate((ptPoints[2], ln[2]))
        MassCenter = np.array([np.sum(ptPoints[0]), np.sum(ptPoints[1]), np.sum(ptPoints[2])])
        MassCenter = MassCenter / float(len(ptPoints[0]))
        dist = np.sum((MassCenter - massCenterPt) ** 2)
        dist = np.sqrt(float(dist))
        return dist

    def VolWeightedDist(self, distances, listLnVol, ptVol):
        """distance between LN and PT weighted by LN volume and additionally normalized by the PT volume"""
        wDist = np.array(distances) * np.array(listLnVol)
        volLargestWeightedDist = np.max(wDist)
        volMeanWeightedDist = np.mean(wDist)
        normVolLargestWeightedDist = volLargestWeightedDist / ptVol
        normVolMeanWeightedDist = volMeanWeightedDist / ptVol

        return volLargestWeightedDist, volMeanWeightedDist, normVolLargestWeightedDist, normVolMeanWeightedDist

    def SmallestDist(self, tumorMass, listLnMass, path, nr):
        """Find smallest distances between objects, using the Delanuay for defining the traingulation and the minimum spanning tree for searching the undirect graph
        path to save a histogram of distances
        nr which patient is it"""
        points = np.array(listLnMass + [tumorMass])

        if len(points) > 4:  # convex hull needs at least 5 points
            tri = Delaunay(points)  # in 3D it returns a pyramid

            distMatrix = np.zeros((len(points), len(points)))
            for triangle in tri.simplices:
                for corner in range(len(triangle)):
                    dist = np.sqrt(float(np.sum((points[triangle[corner]] - points[triangle[corner - 1]]) ** 2)))
                    distMatrix[min(triangle[corner], triangle[corner - 1])][
                        max(triangle[corner], triangle[corner - 1])] = dist

            # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html
            Tcsr = minimum_spanning_tree(distMatrix)
            distanceDistribution = Tcsr.toarray()
            distanceDistribution = distanceDistribution[distanceDistribution.nonzero()]
            varDist = np.var(distanceDistribution)
            meanDist = np.mean(distanceDistribution)

        elif len(points) > 2:
            # one hase to find the distance matrix separately because the Delumay uses convex hull and needs at least 5 points
            # distance matrix - rows correspond to a node and in the matrix we have distanaces to nodes in the columns, only two the shortes distances are included
            distMatrix = np.zeros((len(points), len(points)))
            for pi, p1 in enumerate(points):
                for pj, p2 in enumerate(points):
                    if pj > pi:
                        dist = np.sqrt(float(np.sum((p1 - p2) ** 2)))
                        distMatrix[pi][pj] = dist
            print(distMatrix)

            if len(points) == 4:
                # saving only two the smallest
                vi = np.where(distMatrix[0] == np.max(distMatrix[0]))[0]
                distMatrix[0][vi] = 0

            Tcsr = minimum_spanning_tree(distMatrix)
            distanceDistribution = Tcsr.toarray()
            distanceDistribution = distanceDistribution[distanceDistribution.nonzero()]
            varDist = np.var(distanceDistribution)
            meanDist = np.mean(distanceDistribution)

        elif len(points) == 2:
            distanceDistribution = [np.sqrt(float(np.sum((points[0] - points[1]) ** 2)))]
            varDist = 0
            meanDist = distanceDistribution[0]
        else:
            distanceDistribution = []
            varDist = 0
            meanDist = 0

        # save histogram
        print('dist ', distanceDistribution)
        matplotlib.rcParams.update({'font.size': 12})
        try:
            fig = plt.figure(200, figsize=(15, 15))
            if len(distanceDistribution) == 1:
                plt.hist(distanceDistribution, bins=[distanceDistribution[0] - 0.1, distanceDistribution[0] + 0.1])
            else:
                plt.hist(distanceDistribution)
            plt.title(nr)
            try:
                makedirs(path + os.sep + 'LN_PT_min_distance_distribution' + os.sep)
            except OSError:
                if not isdir(path + os.sep + 'LN_PT_min_distance_distribution' + os.sep):
                    raise
            fig.savefig(path + os.sep + 'LN_PT_min_distance_distribution' + os.sep + nr + '.png')
            plt.close()
        except ValueError:  # if distanceDistribtuion is empty
            pass

        return meanDist, varDist, points

    ##    def Area(self, tri, points):
    ##        '''The area of the figure where centers of the mass (lymph nodes and primary tumor) are the corners'''
    ##        #tri result of Delaunay triangulation
    ##        if len(tri) != 0:
    ##            area = 0
    ##            for t in tri:
    ##                lengths = []
    ##                for ind in np.arange(len(t)):
    ##                    #caculate area with herons equation
    ##                    lengths.append(np.sqrt(float(np.sum((points[t[ind]]-points[t[ind-1]])**2))))
    ##                p = 0.5*np.sum(lengths)
    ##                area += np.sqrt(p*(p-lengths[0])*(p-lengths[1])*(p-lengths[2]))
    ##        else:
    ##            area = 0
    ##
    ##        return area

    def Clusters(self, points, path, nr):
        """define number of clusters using kmeans and calinski harabaz score
        points list of the centers of the masses LN and PT (taken from function SmallestDist)
        path - save a graph
        nr - patient number"""
        if len(points) == 1:  # only primary
            optClust = 0
        elif len(points) <= 3:  # primary plus two nodes
            optClust = 1
        else:
            ch = []  # list of calinski_harabaz scores for different numbers of clusters
            for i in range(2, len(points)):
                kmeans_model = KMeans(n_clusters=i, random_state=1).fit(points)
                labels = kmeans_model.labels_
                ch.append((calinski_harabaz_score(points, labels), i))
            limitedCH = ch[:int(len(points) / 2. + 1) - 2]  # limit number of cluters to N/2 +1
            limitedCH.sort()

            optClust = limitedCH[-1][1]  # take the smallest value of CH

            ch = np.array(ch)
            fig = plt.figure(300, figsize=(15, 15))
            plt.plot(ch[:, 1], ch[:, 0], 'o')
            plt.title(nr)
            try:
                makedirs(path + os.sep + 'LN_PT_clustering' + os.sep)
            except OSError:
                if not isdir(path + os.sep + 'LN_PT_clustering' + os.sep):
                    raise
            fig.savefig(path + os.sep + 'LN_PT_clustering' + os.sep + nr + '.png')
            plt.close()

        return optClust

    def PCAcharacteristic(self, listPointsLN, pointsPT):
        """Uses the PCA of all LN and PT points to calculate elongation and flatness"""
        pnz = np.transpose(pointsPT)
        for n in listPointsLN:
            pnz = np.concatenate((pnz, np.transpose(n)))
        #        #remove possible overlapping points
        spnz = np.unique(pnz, axis=0)
        del pnz

        pca = PCA()
        pca.fit(spnz)
        eigen_value = pca.explained_variance_
        eigen_value.sort()
        major_axis = 4 * eigen_value[-1] ** 0.5
        minor_axis = 4 * eigen_value[-2] ** 0.5
        least_axis = 4 * eigen_value[-3] ** 0.5
        elong = minor_axis / major_axis
        flat = least_axis / major_axis

        return elong, flat

    def Save(self, final, path_save):
        """export results
        final - list with results"""
        names = ['patient', 'largestDist', 'meanDist', 'sumDist', 'weightedDist', 'volLargestWeightedDist',
                 'volMeanWeightedDist', 'normVolLargestWeightedDist', 'normVolMeanWeightedDist', 'massPtLn',
                 'minDistMean', 'minDistVar', 'optClust', 'elongation', 'flateness']
        result = open(path_save, "w")
        for i in names:
            result.write(i + '\t')
        result.write('\n')
        for pat in final:
            for i in pat:
                result.write(str(i) + '\t')
            result.write('\n')
        result.close()
