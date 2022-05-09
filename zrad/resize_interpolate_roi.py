import logging

import cv2
import numpy as np
import pydicom as dc
from scipy.interpolate import interp1d
from scipy.ndimage.morphology import distance_transform_edt


class InterpolateROI(object):

    def structures(self, rs, structure, slices, x_ct, y_ct, xCTspace, yCTspace, l_IM, round_factor):
        """find the contour points for a given ROI,
        calls the getPoints methods with returns a M - 3D matrix with -1 outside, 0 on the border and 1 inside
        """
        self.logger = logging.getLogger("InterpolROI")

        rs = dc.read_file(rs)  # read RS file
        list_organs = []  # list of organs defined in the RS file

        for j in range(len(rs.StructureSetROISequence)):
            list_organs.append([rs.StructureSetROISequence[j].ROIName, rs.StructureSetROISequence[j].ROINumber])
        self.logger.info("Structures in structure set file\t" + ", ".join(map(str, np.array(list_organs)[:, 0])))

        organs = [structure]  # define the structure you're interested in

        contours = []  # list with structure contours

        # search for organ I'm interested in
        for i in range(len(organs)):  # organs defined by user
            for j in range(len(list_organs)):  # organ in RS
                if list_organs[j][0] == organs[i]:  # if the same name
                    for k in range(len(rs.ROIContourSequence)):  # search by ROI number
                        if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:  # check the ROI number
                            st_nr = k  # ROI number
                            try:
                                lista = []  # z position of the slice
                                # contours in dicom are saved as a list with sequence x1, y1, zi, x2, y2, zi, ... xn, yn, zi
                                # where zi is the slice position
                                # if there are sub-contours in the slice then these are two different sequences with the same zi
                                for l in range(len(rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([round(
                                        float(rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]),
                                        round_factor), rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3],
                                        rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista)  # sub-contours in the slice
                                for m in range(len(lista)):
                                    index.append(lista[m][0])
                                self.logger.info('z positions contour \t' + ", ".join(map(str, index)))
                                self.logger.info('z positions image \t' + ", ".join(map(str, slices)))
                                if len(index) != 1:  # if more than one slice
                                    diffI = round(index[1] - index[0], 3)  # double check if the orientation is ok
                                    diffS = round(slices[1] - slices[0], 3)
                                    self.logger.info('resolution image: {} mm, ROI: {} mm'.format(diffI, diffS))
                                    # if different orientation then reverse the contour points
                                    if np.sign(diffI) != np.sign(diffS):
                                        index.reverse()
                                        lista.reverse()
                                    # check for slices without contour in between other contour slices
                                    diff = abs(np.array(index[1:]) - np.array(index[:-1])) / diffS
                                    self.logger.info('difference in z position between slices normalized to slice spacing \t' + ", ".join(map(str, diff)))
                                    dk = 0
                                    for d in range(len(diff)):
                                        # if no empty slice in between then abs(int(round(diff[d],0))) = 1
                                        for di in range(1, int(round(abs(diff[d]), 0))):
                                            # if not add empty slices to index and lista
                                            index.insert(d + dk + 1, index[d + dk] + diffS)
                                            lista.insert(d + dk + 1, [[], [[], []]])
                                            dk += 1
                                    # include empty list to slices where structure was not contour, so in the end
                                    # lista and index has the same length as image
                                    sliceB = index[-1]
                                    sliceE = index[0]

                                    indB = np.where(np.array(slices) == sliceB)[0][0]
                                    indE = np.where(np.array(slices) == sliceE)[0][0]
                                    self.logger.info("Index of first and last slice {}, {}".format(indE, indB))
                                    if indE != 0:
                                        for m in range(abs(indE - 0)):
                                            lista.insert(0, [[], [[], []]])
                                    if indB != (len(slices) - 1):
                                        for m in range(abs(indB - (len(slices) - 1))):
                                            lista.append([[], [[], []]])
                                    for n in range(len(lista)):
                                        lista[n] = lista[n][1:]
                                    contours.append(lista)  # list of contours for all user defined structures
                                else:  # if only one slice of contour
                                    ind = np.where(np.array(slices) == index[0])[0][0]
                                    self.logger.info('contour only in slice {}'.format(ind))
                                    if ind != 0:
                                        for m in range(abs(ind - 0)):
                                            lista.insert(0, [[], [[], []]])
                                    if ind != (len(slices) - 1):
                                        for m in range(abs(ind - (len(slices) - 1))):
                                            lista.append([[], [[], []]])
                                    for n in range(len(lista)):
                                        lista[n] = lista[n][1:]
                                    contours.append(lista)
                                break
                            except AttributeError:
                                self.logger.info('no contours for: ' + organs[i])

        # recalculating for pixels the points into pixels
        contours = np.array(contours, dtype=object)

        # recalculate contour points from mm to pixels
        for i in range(len(contours)):  # contours
            for j in range(len(contours[i])):  # slice
                for n in range(len(contours[i][j])):  # number of contours per slice
                    if np.size(contours[i][j][n][0]) != 0:
                        contours[i][j][n][0] = np.array(abs(contours[i][j][n][0] - x_ct) / xCTspace)
                        contours[i][j][n][1] = np.array(abs(contours[i][j][n][1] - y_ct) / yCTspace)
                        for k in range(len(contours[i][j][n][0])):
                            contours[i][j][n][0][k] = int(round(contours[i][j][n][0][k], 0))
                            contours[i][j][n][1][k] = int(round(contours[i][j][n][1][k], 0))
                        contours[i][j][n][0] = np.array(contours[i][j][n][0], dtype=int)
                        contours[i][j][n][1] = np.array(contours[i][j][n][1], dtype=int)

        x_c_min = []  # x position of contour points to define the region of interest where we look for the structure
        x_c_max = []
        y_c_min = []
        y_c_max = []
        for i in range(len(contours)):  # contours
            for j in range(len(contours[i])):  # slice
                for n in range(len(contours[i][j])):  # number of contours per slice
                    if np.size(contours[i][j][n][0]) != 0:
                        x_c_min.append(np.min(contours[i][j][n][0]))
                        x_c_max.append(np.max(contours[i][j][n][0]))
                        y_c_min.append(np.min(contours[i][j][n][1]))
                        y_c_max.append(np.max(contours[i][j][n][1]))

        x_min = np.min(x_c_min)
        x_max = np.max(x_c_max) + 1
        y_min = np.min(y_c_min)
        y_max = np.max(y_c_max) + 1

        self.logger.info('xmin {}, xmax {}, ymin {}, ymax {}'.format(x_min, x_max, y_min, y_max))

        # finding points inside the contour, M - 3D matrix with -1 outside, 0 on the border and 1 inside
        M = self.getPoints(contours[0], x_min, x_max, y_min, y_max, l_IM)

        return M, x_min, y_min, st_nr

    def multiContour(self, lista):
        """account for multi-contours in one slice,
        checks z positions in each sublist of the list and if the have the same z then creates a new sublist
        for example input l = [[z1, [x1, x2],[y1,y2]], [z1, [x3, x4, x5],[y3, y4, y5]], [z2, [x1, x2],[y1,y2]]] - 3 contours on 2 slices
        output l = [[z1, [[x1, x2],[y1,y2]], [[x3, x4, x5],[y3, y4, y5]]], [z2, [[x1, x2],[y1,y2]]]]"""

        listap = []
        lista_nr = []
        for i in lista:
            listap.append(i[0])
            if i[0] not in lista_nr:
                lista_nr.append(i[0])
        counts = []
        for i in lista_nr:
            counts.append(listap.count(i))  # how many times a certain z position occurs on the list

        listka = []
        nr = 0
        kontur = []
        for i in range(len(lista)):
            if lista[i][0] not in listka:
                m = [lista[i][0]]
                for j in range(counts[nr]):
                    m.append([np.array(lista[i + j][1], dtype=float), np.array(lista[i + j][2], dtype=float)])
                    listka.append(lista[i][0])
                kontur.append(m)
                nr += 1
        return kontur

    def getPoints(self, segment, xmin, xmax, ymin, ymax, l_IM):
        """get points inside the contour
        segment - contour points"""
        cnt_all = []
        # print 'slices in image: ', l_IM
        # print 'slices in structure: ', len(segment)
        for k in range(l_IM):
            cnt = []
            for i in range(len(segment[k])):
                c = []
                for j in range(len(segment[k][i][0])):
                    c.append([segment[k][i][0][j], segment[k][i][1][j]])
                cnt.append(c)
            if cnt == []:
                cnt = [[], []]
            cnt_all.append(cnt)

        M = []
        for k in range(l_IM):
            if cnt_all[k] != [[]]:
                m = np.ones((ymax + 1 - ymin, xmax + 1 - xmin))  # initialize  the 2D matrix with 1
                for n in range(len(cnt_all[k])):  # sub-contours
                    for i in range(ymin, ymax + 1):
                        for j in range(xmin, xmax + 1):
                            # check if the point in inside the polygon defined by contour points, 0 - on contour, 1 - inside, -1 -outside
                            m[i - ymin][j - xmin] = m[i - ymin][j - xmin] * cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j, i), False)
                m = m * (-1) ** (len(
                    cnt_all[k]) + 1)  # to account for multiple sub-contours, especially the holes in the contour
                M.append(m)
            else:
                M.append([])
        M = np.array(M, dtype=object)

        # adjust if there is a contour only in one slice, add slice filled with -1 before and after
        ind = []
        for k in range(len(M)):
            if np.size(M[k]) != 0:
                ind.append(k)

        if len(ind) == 1:
            if ind[0] != 0:
                M[ind[0] - 1] = np.ones((ymax - ymin + 1, xmax - xmin + 1)) * -1
            if ind[0] != len(M):
                M[ind[0] + 1] = np.ones((ymax - ymin + 1, xmax - xmin + 1)) * -1

        return M  # M - 3D matrix with -1 outside, 0 on the border and 1 inside

    def interpolate(self, interp_algo, s1, s2, znew, output_type):
        """interpolate structure between slices"""
        s1 = np.array(s1)  # ROI points in 2D matrix, -1 out, 0 border, 1 in
        s2 = np.array(s2)  # ROI points in 2D matrix, -1 out, 0 border, 1 in
        s1a = s1.copy()
        s2a = s2.copy()
        s1a[np.where(s1 == -1)] = 1  # copy of ROI matrix with 0 on the border and 1 otherwise
        s2a[np.where(s2 == -1)] = 1  # copy of ROI matrix with 0 on the border and 1 otherwise
        im1 = distance_transform_edt(s1a, return_distances=True)  # calculate distance to border abs value
        im2 = distance_transform_edt(s2a, return_distances=True)
        del s1a
        del s2a

        out1 = np.ones((len(s1), len(s1[0])), dtype=int)
        out2 = np.ones((len(s2), len(s2[0])), dtype=int)
        out1[np.where(s1 != -1)] = 0  # a matrix with 1 inside ROI and 0 outside
        out2[np.where(s2 != -1)] = 0
        del s1
        del s2

        im1 = -im1 * out1 + im1 * (~out1 + 2)  # to transform distance as negative inside the ROI and positive outside
        im2 = -im2 * out2 + im2 * (~out2 + 2)

        con = np.array([im1, im2])  # stack slices to make a 3D matrix for interpolation

        z = [0, 1]
        con_m = np.zeros((len(znew), len(im1), len(im1[0])))  # interpolated 3D matrix
        for i in range(len(im1)):  # interpolate each voxel in z direction
            for j in range(len(im1[0])):
                f = interp1d(z, con[:, i, j], kind='linear')
                con_m[:, i, j] = f(znew)
        del con
        del im1
        del im2

        Xfin = []
        Yfin = []
        # for shape return all the points in the structure
        if output_type == 'shape':
            for n in range(len(con_m)):
                indx = np.where(con_m[n] >= 0)[0]
                indy = np.where(con_m[n] >= 0)[1]

                Xfin.append(indx)
                Yfin.append(indy)
            del con_m
        # for texture find polygon encompassing the structure
        elif output_type == 'texture':
            for n in range(len(con_m)):  # slice by  slice
                a = np.zeros((len(con_m[n]), len(con_m[n][0])), dtype=np.uint8)
                a[np.where(con_m[n] >= 0)] = 1  # everything inside contour equal 1
                try:
                    contour, hier = cv2.findContours(a, mode=1, method=2)  # find a contour for the structure
                except:
                    _, contour, hier = cv2.findContours(a, mode=1, method=2)  # find a contour for the structure

                Xf = []
                Yf = []
                for i in range(len(contour)):  # for substructures, like holes
                    Xf.append(contour[i][:, 0, 0])
                    Yf.append(contour[i][:, 0, 1])
                del contour
                Xfin.append(Xf)  # append given slice contour
                Yfin.append(Yf)
                del Xf, Yf
            del con_m

        return Xfin, Yfin  # returns list of contour points, if a slice contrain a sub-structures then the element for slice is list of lists
