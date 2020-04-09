# -*- coding: utf-8 -*-s

# import libraries
import cv2
import numpy as np
import copy
import pydicom as dc  # dicom library
import logging


class Structures(object):
    """Class to extract each point in the image grid which belongs to the defined ROI
    rs - to the structure set including structure set name
    structure – list of organs to be analyzed, always choose first existing organ for this list, for example [GTV_art, GTV], if GTV_art  is not defined in the rs then choose GTV, if GTV not defined then /AttributeError with a message: missing structure
    slices – list of positions of slice in z
    x_ct - x coordinate of image top left corner
    y_ct – y coordinate of image top left corner
    xCTspace – pixel spacing in xy
    position – patient position on the table for example HFS (head first supine)
    len_IM – number of slices
    wv - bool, calculate wavelet, to see if we need contours in wavelet space
        """

    def __init__(self, rs, structure, slices, x_ct, y_ct, xCTspace, len_IM, wv, dim, local):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start Reading in StructureSet")
        self.slices = slices
        self.x_ct = x_ct
        self.y_ct = y_ct
        self.xCTspace = xCTspace
        self.len_IM = len_IM
        self.wv = wv
        self.dim = dim

        self.Xcontour_W = []
        self.Ycontour_W = []
        self.slices_w = []
        self.Xcontour = []
        self.Ycontour = []
        self.Xcontour_Rec = []
        self.Ycontour_Rec = []

        self.find(rs, structure, local)

    def find(self, rs, structure, local):
        self.logger.debug("Structure set filename " + rs)
        self.rs = dc.read_file(rs)  # read RS file
        list_organs = []  # list of organs defined in the RS file
        self.logger.info('structures in RS file: ')
        for j in range(len(self.rs.StructureSetROISequence)):  # find structure name and number
            list_organs.append([self.rs.StructureSetROISequence[j].ROIName, self.rs.StructureSetROISequence[j].ROINumber])
            self.logger.info("Structures in Structure Set File: " + self.rs.StructureSetROISequence[j].ROIName)

        organs = structure  # define the structure you're interested in

        self.contours = []  # list with structure contours

        # search for organ I'm interested in
        for i in range(len(organs)):  # organs defined by user
            for j in range(len(list_organs)):  # organ in RS
                if list_organs[j][0] == organs[i]:  # if the same name
                    for k in range(len(self.rs.ROIContourSequence)):  # search by ROI number
                        if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:  # double check the ROI number
                            try:
                                lista = []  # z position of the slice
                                # contours in dicom are save as a list with sequence x1, y1, zi, x2, y2, zi, ... xn, yn, zi
                                # where zi is the slice position
                                # if there are subcontours in the slice then these re two different sequences with the same zi
                                for l in range(len(self.rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([round(
                                        float(self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]), 3),
                                                  self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3],
                                                  self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista)  # subcontours in the slice
                                for m in range(len(lista)):
                                    index.append(lista[m][0])
                                self.logger.info('z positions contour \n' + ", ".join(map(str, index)))
                                self.logger.info('z positions image \n' + ", ".join(map(str, self.slices)))
                                if len(self.slices) == 1:
                                    slice_count = False  # only one slice
                                else:
                                    slice_count = True  # True is more than one slice
                                try:
                                    diffI = round(index[1] - index[0], 2)  # double check if the orientation is ok
                                except IndexError:
                                    info = 'only one slice'
                                    slice_count = False

                                if slice_count:  # if more than one slice
                                    diffS = round(self.slices[1] - self.slices[0], 2)
                                    self.logger.info("resolution image, ROI " + ", ".join(map(str, (diffI, diffS))))
                                    if np.sign(diffI) != np.sign(diffS):  # if different orientation then reverse the contour points
                                        index.reverse()
                                        lista.reverse()
                                    # check for slices without contour in between other contour slices
                                    diff = abs(np.array(index[1:]) - np.array(index[:-1])) / diffS
                                    self.logger.info(
                                        "difference in t position between slices normalized to slice spacing  " + ", ".join(
                                            map(str, diff)))
                                    dk = 0
                                    for d in range(len(diff)):
                                        for di in range(1, abs(int(round(diff[d], 0)))):  # if no empty slice in between then abs(int(round(diff[d],0))) = 1
                                            index.insert(d + dk + 1, index[d + dk] + diffS)  # if not add empty slices to index and lista
                                            lista.insert(d + dk + 1, [[], [[], []]])
                                            dk += 1
                                    # include empty list to slices where structure was not contour, so in the end lista and index has the same length as image
                                    sliceB = index[-1]  # first slice with contour (Begin)
                                    sliceE = index[0]  # last slice with contour (End)
                                    indB = np.where(np.array(self.slices) == sliceB)[0][0]
                                    indE = np.where(np.array(self.slices) == sliceE)[0][0]
                                    if indE != 0:
                                        for m in range(abs(indE - 0)):
                                            lista.insert(0, [[], [[], []]])
                                    if indB != (len(self.slices) - 1):
                                        for m in range(abs(indB - (len(self.slices) - 1))):
                                            lista.append([[], [[], []]])
                                    for n in range(len(lista)):
                                        lista[n] = lista[n][1:]
                                    self.contours.append(lista)  # list of contours for all user defined structures
                                    break
                                else:   # check if this is also true for several contours in one slice!!!!!!!!!!!!!!!!!!!!!!!
                                    # if only one slice of contour (but also many other slices) or if only one slice
                                    self.logger.info("contour only in slice")
                                    if len(self.slices) == 1 and len(index) != 1:  # if one slice but contour of 3D volume
                                        ind = np.where(np.array(self.slices) == index)[0][0]
                                        lista[ind] = lista[ind][1:]  # get rid of slice position in lista for ind
                                        self.contours.append([lista[ind]])
                                    else:  # if several or only one slice(s), and only one slice with contour
                                        ind = np.where(np.array(self.slices) == index[0])[0][0]
                                        if ind != 0:
                                            for m in range(abs(ind - 0)):
                                                lista.insert(0, [[], [[], []]])
                                        if ind != (len(self.slices) - 1):
                                            for m in range(abs(ind - (len(self.slices) - 1))):
                                                lista.append([[], [[], []]])
                                        for n in range(len(lista)):
                                            lista[n] = lista[n][1:]
                                        self.contours.append(lista)
                            except AttributeError:
                                self.logger.info("no contours for: " + organs[i])

        if self.wv and self.dim == "2D" or self.wv and self.dim == "2D_singleSlice":  # contours not scaled in slice-direction for 2D wavelet calculation
            contours_wv = copy.deepcopy(self.contours)

        # recalculating for pixels the points into pixels
        self.cnt = []
        if local:
            self.contours = np.array(self.contours)
            self.organs = organs[0]
        else:
            try:
                if len(self.contours) != 1:
                    self.contours = np.array([self.contours[0]])
                    self.organs = organs[0]
                else:
                    if self.dim != "2D_singleSlice":
                        self.contours = np.array(self.contours)
                    self.organs = organs[-1]
            except IndexError:
                #            info = "Check structure names" #for Lucas
                #            MyException(info)
                raise IndexError

        if list(self.contours[0]) == ['one slice']:  # stop the calculation if it's only one slice  % does it ever stop ?????????????????????????
            self.Xcontour = 'one slice'
            self.Ycontour = 'one slice'
            self.Xcontour_W = 'one slice'
            self.Ycontour_W = 'one slice'
        else:  # continue, recalculate contour points from mm to pixels
            print(self.x_ct, self.xCTspace)
            for i in range(len(self.contours)):  # contours
                for j in range(len(self.contours[i])):  # slice
                    for n in range(len(self.contours[i][j])):  # number of contours per slice
                        if list(self.contours[i][j][n][0]):  # if list (with x values) not empty
                            self.contours[i][j][n][0] = np.array(abs(self.contours[i][j][n][0] - self.x_ct) / self.xCTspace)
                            self.contours[i][j][n][1] = np.array(abs(self.contours[i][j][n][1] - self.y_ct) / self.xCTspace)
                            for k in range(len(self.contours[i][j][n][0])):
                                self.contours[i][j][n][0][k] = int(round(self.contours[i][j][n][0][k], 0))
                                self.contours[i][j][n][1][k] = int(round(self.contours[i][j][n][1][k], 0))
                            self.contours[i][j][n][0] = np.array(self.contours[i][j][n][0], dtype=np.int)
                            self.contours[i][j][n][1] = np.array(self.contours[i][j][n][1], dtype=np.int)

            x_c_min = []  # x position of contour points to define the region of interest where we look for the structure
            x_c_max = []
            y_c_min = []
            y_c_max = []
            for i in range(len(self.contours)):  # contours
                for j in range(len(self.contours[i])):  # slice
                    for n in range(len(self.contours[i][j])):  # number of contours per slice
                        if list(self.contours[i][j][n][0]):  # if contour values in that slice
                            x_c_min.append(np.min(self.contours[i][j][n][0]))
                            x_c_max.append(np.max(self.contours[i][j][n][0]))
                            y_c_min.append(np.min(self.contours[i][j][n][1]))
                            y_c_max.append(np.max(self.contours[i][j][n][1]))

            try:
                x_min = np.min(x_c_min)
                x_max = np.max(x_c_max) + 1
                y_min = np.min(y_c_min)
                y_max = np.max(y_c_max) + 1

                self.logger.info("xmin, xmax, ymin, ymax " + ", ".join(map(str, (x_min, x_max, y_min, y_max))))

                del x_c_min
                del x_c_max
                del y_c_min
                del y_c_max

                # finding points inside the contour
                self.Xcontour, self.Ycontour, cnt = self.getPoints(self.contours[0], x_min, x_max, y_min, y_max, self.len_IM)

                if local:
                    self.Xcontour_Rec, self.Ycontour_Rec, cnt = self.getPoints(self.contours[-1], x_min, x_max, y_min, y_max, self.len_IM)

                del self.contours

                # wavelets ---------------------------------------------------------------------------------------------
                # finding the points for transformed images
                if self.wv:
                    self.contours = []
                    # slices position in the transformed image
                    self.slices_w = list(np.array(self.slices).copy())

                    if self.dim == "2D" or self.dim == "2D_singleSlice":  # slice position doesn't change - use self.contour as previously calculated
                        self.contours = contours_wv  # slice positions will be the same

                    if self.dim == "3D":  # slice position changed
                        # boundary conditions
                        if self.slices_w[0] - self.slices_w[1] < 0:
                            self.slices_w.insert(0, self.slices[0] - 2 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.insert(1, self.slices[0] - abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] + abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] + 2 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] + 3 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] + 4 * abs(self.slices[0] - self.slices[1]))
                        else:
                            self.slices_w.insert(0, self.slices[0] + 2 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.insert(1, self.slices[0] + abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] - abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] - 2 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] - 3 * abs(self.slices[0] - self.slices[1]))
                            self.slices_w.append(self.slices[-1] - 4 * abs(self.slices[0] - self.slices[1]))
                        for i in range(len(self.slices_w)):  # round list elements
                            self.slices_w[i] = round(self.slices_w[i], 3)

                        # get the points in the contour, as previously
                        self.logger.info("Calculate Wavelets")
                        # same as above for a original image ROI
                        for i in range(len(organs)):  # organ defined by user
                            for j in range(len(list_organs)):  # organ in RS
                                if list_organs[j][0] == organs[i]:
                                    for k in range(len(self.rs.ROIContourSequence)):
                                        if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                                            try:
                                                lista = []
                                                for l in range(len(self.rs.ROIContourSequence[k].ContourSequence)):
                                                    lista.append([round(float(
                                                        self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]), 3),
                                                        self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3],
                                                        self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                                lista.sort()
                                                index = []
                                                lista = self.multiContour(lista)  # subcontours in the slice
                                                for m in range(len(lista)):
                                                    index.append(round(lista[m][0], 3))
                                                slice_count = True  # True is more than one slice
                                                try:
                                                    diffI = round(index[1] - index[0], 3)  # double check if the orientation is ok
                                                except IndexError:
                                                    info = 'only one slice'
                                                    slice_count = False
                                                if slice_count:  # if more than one slice
                                                    diffS = round(self.slices_w[1] - self.slices_w[0], 3)
                                                    if np.sign(diffI) != np.sign(diffS):
                                                        index.reverse()
                                                        lista.reverse()
                                                    # empty slices
                                                    diff = abs(np.array(index[1:]) - np.array(index[:-1])) / diffS
                                                    dk = 0
                                                    for d in range(len(diff)):
                                                        for di in range(1, abs(int(diff[d]))):
                                                            index.insert(d + dk + 1, index[d + dk] + diffS)
                                                            lista.insert(d + dk + 1, [[], [[], []]])
                                                            dk += 1
                                                    sliceB = index[-1]
                                                    sliceE = index[0]
                                                    indB = np.where(np.array(self.slices_w) == sliceB)[0][0]
                                                    indE = np.where(np.array(self.slices_w) == sliceE)[0][0]
                                                    if indE != 0:
                                                        for m in range(abs(indE - 0)):
                                                            lista.insert(0, [[], [[], []]])
                                                    if indB != (len(self.slices_w) - 1):
                                                        for m in range(abs(indB - (len(self.slices_w) - 1))):
                                                            lista.append([[], [[], []]])
                                                    lista = lista[::2]  # adjust resolution drops down by 2
                                                    for n in range(len(lista)):
                                                        lista[n] = lista[n][1:]
                                                    self.contours.append(lista)
                                                    break
                                            except AttributeError:
                                                print('no contours for: ' + organs[i])
                        self.slices_w = self.slices_w[::2]  # adjust resolution drops down by 2

                    # recalculating for pixels
                    self.cnt = []
                    x_ct = self.x_ct - 2 * self.xCTspace  # adjust resolution drops down by 2
                    y_ct = self.y_ct - 2 * self.xCTspace
                    for i in range(len(self.contours)):  # contours
                        for j in range(len(self.contours[i])):  # slice
                            for n in range(len(self.contours[i][j])):  # number of contours per slice
                                if list(self.contours[i][j][n][0]):
                                    self.contours[i][j][n][0] = np.array(abs(self.contours[i][j][n][0] - x_ct) / (2 * self.xCTspace))
                                    self.contours[i][j][n][1] = np.array(abs(self.contours[i][j][n][1] - y_ct) / (2 * self.xCTspace))
                                    for k in range(len(self.contours[i][j][n][0])):
                                        self.contours[i][j][n][0][k] = int(round(self.contours[i][j][n][0][k], 0))
                                        self.contours[i][j][n][1][k] = int(round(self.contours[i][j][n][1][k], 0))
                                    self.contours[i][j][n][0] = np.array(self.contours[i][j][n][0], dtype=np.int)
                                    self.contours[i][j][n][1] = np.array(self.contours[i][j][n][1], dtype=np.int)

                    x_c_min = []  # x position of contour points to define the region of interest where we look for the structure
                    x_c_max = []
                    y_c_min = []
                    y_c_max = []
                    for i in range(len(self.contours)):  # contours
                        for j in range(len(self.contours[i])):  # slice
                            for n in range(len(self.contours[i][j])):  # number of contours per slice
                                if list(self.contours[i][j][n][0]):
                                    x_c_min.append(np.min(self.contours[i][j][n][0]))
                                    x_c_max.append(np.max(self.contours[i][j][n][0]))
                                    y_c_min.append(np.min(self.contours[i][j][n][1]))
                                    y_c_max.append(np.max(self.contours[i][j][n][1]))

                    x_min = np.min(x_c_min)
                    x_max = np.max(x_c_max)
                    y_min = np.min(y_c_min)
                    y_max = np.max(y_c_max)

                    del x_c_min
                    del x_c_max
                    del y_c_min
                    del y_c_max
                    self.logger.info(
                        "Wavelet xmin, xmax, ymin, ymax " + ", ".join(map(str, (x_min, x_max, y_min, y_max))))

                    # get all point inside the contour
                    if self.dim == "2D" or self.dim == "2D_singleSlice":
                        self.Xcontour_W, self.Ycontour_W, cnt = self.getPoints(self.contours[0], x_min, x_max, y_min, y_max, self.len_IM)
                    elif self.dim == "3D":
                        self.Xcontour_W, self.Ycontour_W, cnt = self.getPoints(self.contours[0], x_min, x_max, y_min, y_max, int(np.floor((self.len_IM + 5) / 2.)))

            except ValueError:  # ValueError
                self.Xcontour_W = ''
                self.Ycontour_W = ''
                self.logger.warn('too small structure')
                pass
            # raise ValueError

    def multiContour(self, lista):
        """account for multicontours in one slice,
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
                    m.append([np.array(lista[i + j][1], dtype=np.float), np.array(lista[i + j][2], dtype=np.float)])
                    listka.append(lista[i][0])
                kontur.append(m)
                nr += 1
        return kontur

    def getPoints(self, segment, xmin, xmax, ymin, ymax, nr_slices):
        """get points inside the contour (and for resolution of wavelet transform)
        segment - contour points
        nr_slices: self.len_IM.
        If points calculated for wavelets in 3D, slices are reduced, therefore,
        int(np.floor((self.len_IM + 5) / 2.)) must be used as input for nr_slices!"""
        cnt_all = []
        # print 'slices in image: ', self.len_IM
        # print 'slices in structure: ', len(segment)
        for k in range(nr_slices):
            cnt = []
            for i in range(len(segment[k])):
                c = []
                for j in range(len(segment[k][i][0])):
                    c.append([segment[k][i][0][j], segment[k][i][1][j]])
                cnt.append(c)
            if cnt == []:
                cnt = [[], []]
            cnt_all.append(cnt)

        Xp = []
        Yp = []
        for k in range(nr_slices):
            if cnt_all[k] != [[]]:
                M = []
                for n in range(len(cnt_all[k])):
                    m = np.zeros((int(ymax + 1 - ymin), int(xmax + 1 - xmin)))
                    for i in range(ymin, ymax + 1):
                        for j in range(xmin, xmax + 1):  # check if the point in inside the polygon defined by contour points, 0 - on contour, 1 - inside, -1 -outside
                            m[int(i - ymin)][int(j - xmin)] = cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j, i), False)
                    M.append(m)
                for n in range(1, len(M)):  # to account for multiple subcontours in a slice, including holes in a contour
                    M[0] = M[0] * M[n]
                M[0] = M[0] * (-1) ** (len(M) + 1)
                ind = np.where(M[0] >= 0)
                xp = ind[1] + xmin
                yp = ind[0] + ymin
                Xp.append([xp])
                Yp.append([yp])
            else:
                Xp.append([])
                Yp.append([])
        return Xp, Yp, cnt_all
