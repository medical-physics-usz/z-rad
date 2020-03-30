# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import copy


class NanOptimizer:
    raw_img = []        # real image matrix
    raw_voi_data = []   # the real 3d matrix
    raw_rec_data = []   # the 3d matrix of the recurrence
    img_indices = []    # contains the indices of the roi
    nan_img = []        # NH NAN image
    tt_img = []         # tumor tissue in neighborhood
    tt_img_2D = []
    slc_nof_tt = []     # number of tumor tissue pixels per slice
    raw_shape = 0       # shape of the image matrix
    dic_grid = {}       # contains all possible grid points
    dic_grid_full = {}  # contains the grid points spanned over the neighboured slices
    nan_sums = {}       # contains the sum of nans for each voxel
    tt_sums = {}        # contains the sum of tumor tissue parts
    dic_tot_tumcov = {}       # contains the total tumor coverage for all possible grid points
    rec_grid = {}


    def __init__(self, image_matrix=[], VOI_matrix=[], REC_matrix=[], VOI_path='', max_tumor_coverage=0, dynamic_grid=0):
        self.max_tumor_coverage = max_tumor_coverage
        self.dynamic_grid = dynamic_grid
        print(self.max_tumor_coverage)
        if len(image_matrix):
            NanOptimizer.raw_img = copy.deepcopy(image_matrix)
            NanOptimizer.raw_shape = NanOptimizer.raw_img.shape
        if len(VOI_matrix):
            NanOptimizer.raw_voi_data = copy.deepcopy(VOI_matrix)
            NanOptimizer.raw_shape = NanOptimizer.raw_voi_data.shape
            print("RAW IMAGE MATRIX")
            for slice in VOI_matrix:
                print(slice)
                nof_tt = len(np.argwhere(~np.isnan(slice)))
                NanOptimizer.slc_nof_tt.append(nof_tt)
                print(nof_tt)
            print("END")
            print(NanOptimizer.slc_nof_tt)
        if len(REC_matrix):
            NanOptimizer.raw_rec_data = copy.deepcopy(REC_matrix)
            print("RAW RECURRENCE MATRIX")
            for slice in REC_matrix:
                print(slice)
            print("END")
        if len(VOI_path):
            self.read_VOI_file(VOI_path)

    def read_VOI_file(self, file_path):
        f = open(file_path, mode="r")
        roi = f.readlines()
        contour_data_found = False

        # create numpy array from text file
        for line in roi:
            line = line.replace('\n', '')
            temp = []

            if line.__contains__(';'):
                NanOptimizer.voi_raw_data.append(copy.deepcopy(NanOptimizer.raw_img))
                NanOptimizer.raw_img = []
                continue

            for el in line.replace('[', '').replace(']', '').split(','):
                if el == 'nan':
                    temp.append(np.nan)
                else:
                    temp.append(float(el))

            NanOptimizer.raw_img.append(temp)

        NanOptimizer.voi_raw_data.append(copy.deepcopy(NanOptimizer.raw_img))

        print(np.asarray(NanOptimizer.voi_raw_data))

    # Get the raw image via file
    def read_file(self, file_path):
        f = open(file_path, mode="r")
        roi = f.readlines()
        contour_data_found = False

        # create numpy array from text file
        for line in roi:
            line = line.replace('\n', '')
            temp = []

            for el in line.replace('[','').replace(']','').split(','):
                if el == 'nan':
                    temp.append(np.nan)
                else:
                    temp.append(float(el))

            NanOptimizer.raw_img.append(np.asarray(temp))

        NanOptimizer.raw_img = np.asarray(NanOptimizer.raw_img)


    def minimize_nan_contribtion(self):

        if len(NanOptimizer.raw_img):
            self.get_img_indices_2D()
            self.plot()
            self.get_nan_img_2D()
            optimal_index = self.gridding_2D()
            print(optimal_index)
        else:
            self.get_img_indices_3D()
            #self.plot()
            self.get_nan_img_3D()
            if self.dynamic_grid == 1:
                self.gridding_3D_dynamic()
            else:
                optimal_index = self.gridding_3D_static()
                print(optimal_index)

    def plot(self, img=[]):
        # Plotting
        if not len(img):
            img = NanOptimizer.raw_img
        plt.subplot(111)
        print(img)
        plt.imshow(img, cmap='gray')
        plt.show()

    def set_contour_to_nan_3D(self):

        bin_img_masks = []
        for img in NanOptimizer.voi_raw_data:
            temp = copy.deepcopy(np.asarray(img))
            ind_img = np.argwhere(~np.isnan(np.asarray(img)))
            ind_nan = np.argwhere(np.isnan(np.asarray(img)))
            temp[ind_img[:,0], ind_img[:,1]] = 1
            temp[ind_nan[:,0], ind_nan[:,1]] = 0
            bin_img_masks.append(copy.deepcopy(temp))

        self.img_indices = self.get_img_borders(bin_img_masks)

        #self.cont_ind = np.argwhere(Optimize2D.raw_img == 3000)


        # Set contour to NAN
        #Optimize2D.raw_img[self.cont_ind[:, 0], self.cont_ind[:, 1]] = np.nan

    def get_img_borders(self, image_masks, allow_nan_center=False):
        """
        :param allow_nan_center: if false only the indices where the logic AND operation over 3 slices results in 1
                                 if true the greatest image area of these 3 slices is returned which could then lead to center pixels which contain NAN
        :return: the image indices where center pixels are allowed
        """
        and_op = []

        if allow_nan_center:
            return []
        else:
            and_op = image_masks[0]*image_masks[1]*image_masks[2]

        print(and_op)

    def get_img_indices_2D(self):
        NanOptimizer.img_indices = np.argwhere(~np.isnan(NanOptimizer.raw_img))

    def get_img_indices_3D(self):
        NanOptimizer.img_indices = np.argwhere(~np.isnan(NanOptimizer.raw_voi_data))
        print("NANOPT IMG_INDICES")
        for arr in NanOptimizer.img_indices:
            print(arr)
        print("END")

    def get_nan_img_2D(self):
        """
        Converts the image data to a special image matrix where each pixel contains information
        about the number of NANs in the neighborhood
        :return:
        """
        y_dim = len(NanOptimizer.raw_img[:, 0])
        x_dim = len(NanOptimizer.raw_img[0])

        # Enlarge original image area
        # Necessary to calculate the number of NANs in the edge-regions of the whole image matrix
        temp = np.zeros(shape=(y_dim + 2, x_dim + 2))
        temp.fill(np.nan)

        # NAN- filled mask which contains the neighborhood information instead of the original image data
        nh_nan_img = np.zeros(shape=(y_dim + 2, x_dim + 2))
        nh_nan_img.fill(np.nan)

        img_indices = []

        # Transfer raw image data to NAN- matrix
        # This results in a raw image enlarged by 1 row and 1 column filled with NANs
        for y in range(y_dim):
            for x in range(x_dim):
                temp[y+1, x+1] = NanOptimizer.raw_img[y, x]
                img_indices.append([x+1, y+1])

        # Calculate the number of NANs in the neighborhood for the original pixels of the raw image
        for x,y in img_indices:
            sub_voxel = temp[y-1 : y+2, x-1 : x+2]

            # if np.isnan(sub_voxel[1, 1]):
            #     number_of_nans = len(np.argwhere(np.isnan(sub_voxel))) - 1
            # else:
            number_of_nans = len(np.argwhere(np.isnan(sub_voxel)))

            # if number_of_nans == -1:
            #     number_of_nans = 0

            nh_nan_img[y, x] = number_of_nans


        # Cut the temporary NAN- image to the same size as the original image
        NanOptimizer.nan_img = nh_nan_img[np.argwhere(~np.isnan(nh_nan_img))[:, 0],
                                  np.argwhere(~np.isnan(nh_nan_img))[:,1]].reshape(len(NanOptimizer.raw_img[:, 0]),
                                                                                   len(NanOptimizer.raw_img[0]))
        print(NanOptimizer.nan_img)

    def get_nan_img_3D(self):
        """
        Converts the image data to a special image matrix where each pixel contains information
        about the number of NANs in the neighborhood
        :return:
        """
        # sh = NanOptimizer.raw_voi_data.shape
        z_dim = int(NanOptimizer.raw_shape[0])
        y_dim = int(NanOptimizer.raw_shape[1])
        x_dim = int(NanOptimizer.raw_shape[2])

        # Enlarge original image area
        # Necessary to calculate the number of NANs in the edge-regions of the whole image matrix
        temp = np.zeros(shape=(z_dim + 2, y_dim + 2, x_dim + 2))
        temp.fill(np.nan)

        # NAN- filled mask which contains the neighborhood information instead of the original image data
        nh_nan_img = np.zeros(shape=(z_dim + 2, y_dim + 2, x_dim + 2))
        nh_nan_img.fill(np.nan)

        nh_tt_img = np.zeros(shape=(z_dim + 2, y_dim + 2, x_dim + 2))
        nh_tt_img_2D = np.zeros(shape=(z_dim + 2, y_dim + 2, x_dim + 2))

        img_indices = []

        # Transfer raw image data to NAN- matrix
        # This results in a raw image enlarged by 1 row and 1 column filled with NANs
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    temp[z + 1, y + 1, x + 1] = NanOptimizer.raw_voi_data[z, y, x]
                    img_indices.append([z + 1, x + 1, y + 1])

        print("IMG INDICES")
        print(img_indices)
        # Calculate the number of NANs in the neighborhood for the original pixels of the raw image
        for z, x, y in img_indices:
            sub_voxel = temp[z - 1: z + 2, y - 1: y + 2, x - 1: x + 2]
            sub_area = temp[z, y - 1: y + 2, x - 1: x + 2]
            # print sub_voxel
            number_of_nans = len(np.argwhere(np.isnan(sub_voxel)))
            nh_nan_img[z, y, x] = number_of_nans
            nh_tt_img[z, y, x] = 27 - number_of_nans
            nh_tt_img_2D[z, y, x] = 9 - len(np.argwhere(np.isnan(sub_area)))

        print("------------------------------- DIMENSIONS ---------------------------------------------")
        print(nh_nan_img.shape)
        NanOptimizer.nan_img = nh_nan_img[1:z_dim+1, 1:y_dim+1, 1:x_dim+1]
        NanOptimizer.tt_img = nh_tt_img[1:z_dim+1, 1:y_dim+1, 1:x_dim+1]
        NanOptimizer.tt_img_2D = nh_tt_img_2D[1:z_dim+1, 1:y_dim+1, 1:x_dim+1]
        print(NanOptimizer.nan_img.shape)
        print(NanOptimizer.raw_shape)
        print("-----------------------------------------END---------------------------------------------")
        # for sl in NanOptimizer.nan_img[:]:
        #     print sl

        print ("TUMOR TISSUE 3D")
        for sl in NanOptimizer.tt_img[:]:
            print(sl)
        print("END TT")

        print ("TUMOR TISSUE 2D")
        for sl in NanOptimizer.tt_img_2D[:]:
            print(sl)
        print("END TT")


    def gridding_3D_static(self):

        temp = copy.deepcopy(NanOptimizer.nan_img)

        percentage = 0
        offset = 0
        temp_arr = []

        start_indices_x = [0, 1, 2]
        start_indices_y = [0, 1, 2]
        start_indices_z = [0, 1, 2]

        z_dim = int(NanOptimizer.raw_shape[0])
        y_dim = int(NanOptimizer.raw_shape[1])
        x_dim = int(NanOptimizer.raw_shape[2])

        NanOptimizer.dic_grid = {}
        NanOptimizer.nan_sums = {}

        for z in start_indices_z:
            for x in start_indices_x:
                for y in start_indices_y:
                    temp2 = copy.deepcopy(temp)
                    # print "---------  x=%d y=%d" % (x, y)
                    nan_sum = 0
                    tt_sum = 0
                    lst_grid = []
                    # print ("Start: " + str(Optimize2D.nan_img[x, y]))
                    print("START")
                    print(np.arange(z, z_dim, 3))
                    print(np.arange(y, y_dim, 3))
                    print(np.arange(x, x_dim, 3))
                    for slc in np.arange(z, z_dim, 3):
                        for col in np.arange(x, x_dim, 3):
                            for row in np.arange(y, y_dim, 3):
                                for arr in NanOptimizer.img_indices[np.where(NanOptimizer.img_indices[:,0] == slc)]:# NanOptimizer.img_indices[range(len(NanOptimizer.img_indices))]:
                                    el = np.asarray([slc, row, col])
                                    if (el == arr).all():
                                        nan_sum += NanOptimizer.nan_img[slc, row, col]
                                        lst_grid.append([slc, row, col])
                                        temp2[slc, row, col] = np.max(temp) + 1  # just to get some contrast to the center pixels

                    NanOptimizer.dic_grid[(z, x, y)] = copy.deepcopy(lst_grid)
                    NanOptimizer.nan_sums[(z, x, y)] = nan_sum
                    percentage += (100/24)
                    print(str(percentage) + "%")

        # plt.show()
        print("NAN_SUMS")
        print(NanOptimizer.nan_sums)
        print(len(NanOptimizer.nan_sums))
        return_value = 0

        min_val = min(NanOptimizer.nan_sums.values())
        print("MINVAL")
        print(min_val)
        for key, val in NanOptimizer.nan_sums.items():
            if val == min_val:
                return_value = key

        # Optimal start pixel returned
        x, y, z = return_value

        print("DIC GRID")
        print(NanOptimizer.dic_grid[return_value])
        print(return_value)

        # fig = plt.figure()
        # ax = Axes3D(fig)
        #
        # print np.asarray(NanOptimizer.dic_grid[return_value])[:, 0]
        # print np.asarray(NanOptimizer.dic_grid[return_value])[:, 1]
        # print np.asarray(NanOptimizer.dic_grid[return_value])[:, 2]
        #
        # x_coords = np.asarray(NanOptimizer.dic_grid[return_value])[:, 2]
        # y_coords = np.asarray(NanOptimizer.dic_grid[return_value])[:, 1]
        # z_coords = np.asarray(NanOptimizer.dic_grid[return_value])[:, 0]
        #
        # ax.scatter(x_coords, y_coords, z_coords)
        #
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #
        # plt.show()

        self.create_full_grid()
        self.get_grid_points_recurrence()
        return return_value


    def gridding_3D_dynamic(self):

        percentage = 0

        start_indices_x = [0, 1, 2]
        start_indices_y = [0, 1, 2]
        start_indices_z = [0, 1, 2]

        z_dim = int(NanOptimizer.raw_shape[0])
        y_dim = int(NanOptimizer.raw_shape[1])
        x_dim = int(NanOptimizer.raw_shape[2])

        NanOptimizer.dic_grid = {}
        NanOptimizer.nan_sums = {}
        NanOptimizer.tt_sums = {}

        dic_slc_sum = {}    #sum of tumor tissue or NANs per slice
        dic_opt = {}

        for z in start_indices_z:
            dic_slc_sum[z] = 0
            dic_opt[z] = []
            # print "NEW z: " + str(z)
            # print np.arange(z, z_dim, 3)
            for slc in np.arange(z, z_dim, 3):
                for x in start_indices_x:
                    for y in start_indices_y:

                        tt_sum = 0
                        nan_sum = 0
                        lst_grid = []

                        for col in np.arange(x, x_dim, 3):
                            for row in np.arange(y, y_dim, 3):
                                for arr in NanOptimizer.img_indices[np.where(NanOptimizer.img_indices[:,0] == slc)]: #NanOptimizer.img_indices[range(len(NanOptimizer.img_indices))]:
                                    el = np.asarray([slc, row, col])
                                    if (el == arr).all():
                                        # print Optimize2D.nan_img[row, col]
                                        nan_sum += NanOptimizer.nan_img[slc, row, col]
                                        tt_sum += NanOptimizer.tt_img[slc, row, col]
                                        lst_grid.append([slc, row, col])

                        if len(lst_grid):
                            NanOptimizer.dic_grid[(slc, x, y)] = copy.deepcopy(lst_grid)
                            NanOptimizer.nan_sums[(slc, x, y)] = nan_sum
                            NanOptimizer.tt_sums[(slc, x, y)] = tt_sum

                if len(NanOptimizer.nan_sums):
                    print(NanOptimizer.dic_grid)
                    print(NanOptimizer.nan_sums)
                    print(NanOptimizer.tt_sums)

                    if self.max_tumor_coverage == 0:
                        # Find lowest number of NANs in this slice
                        opt_val = min(NanOptimizer.nan_sums.values())
                        print("MINVAL")
                        print(opt_val)
                        for key, val in NanOptimizer.nan_sums.items():
                            if val == opt_val:
                                dic_opt[z].append(NanOptimizer.dic_grid[key])
                                # print "Slice min: "
                                # print slice_min
                                # print NanOptimizer.dic_grid[key]
                                break
                    else:
                        opt_val = max(NanOptimizer.tt_sums.values())
                        print("MAXVAL")
                        print(opt_val)
                        for key, val in NanOptimizer.tt_sums.items():
                            if val == opt_val:
                                dic_opt[z].append(NanOptimizer.dic_grid[key])
                                # print "Slice min: "
                                # print slice_min
                                # print NanOptimizer.dic_grid[key]
                                break

                    NanOptimizer.nan_sums.clear()
                    NanOptimizer.tt_sums.clear()

                    dic_slc_sum[z] += opt_val

            percentage += (100 / 3)
            print(str(percentage) + "%")

        print("OPT DIC")
        print(dic_opt)
        print("DIC_SLC_SUM")
        print(dic_slc_sum)

        print("ARR")
        NanOptimizer.dic_grid = []
        if self.max_tumor_coverage == 0:
            # Find lowest number of NANs
            print(min(dic_slc_sum, key=dic_slc_sum.get))
            for slc in dic_opt[min(dic_slc_sum, key=dic_slc_sum.get)]:
                for arr in slc:
                    print(arr)
                    NanOptimizer.dic_grid.append(arr)
        else:
            print(max(dic_slc_sum, key=dic_slc_sum.get))
            for slc in dic_opt[max(dic_slc_sum, key=dic_slc_sum.get)]:
                for arr in slc:
                    print(arr)
                    NanOptimizer.dic_grid.append(arr)


        print(NanOptimizer.dic_grid)
        self.create_full_grid(dynamic=True)
        self.get_grid_points_recurrence(dynamic=True)
        # NanOptimizer.dic_grid = dic_opt[min(dic_slc_nan)]

    def gridding_2D(self):

        #print np.arange(0, len(Optimize2D.raw_img[:, 0]), 3)
        #print np.arange(0, len(Optimize2D.raw_img[0]), 3)

        # scale image for better contrast
        temp = copy.deepcopy(NanOptimizer.nan_img)
        # temp[Optimize2D.img_indices[:, 0], Optimize2D.img_indices[:, 1]] = 2
        # not_one = np.argwhere(temp != 1)
        # temp[not_one[:, 0], not_one[:, 1]] = 0
        #temp[self.cont_ind[:, 0], self.cont_ind[:, 1]] = 1

        percentage = 0
        offset = 0
        temp_arr = []

        NanOptimizer.nan_sums = {}
        start_indices_x = [0, 1, 2]
        start_indices_y = [0, 1, 2]
        NanOptimizer.dic_grid = {}

        for x in start_indices_x:
            for y in start_indices_y:
                temp2 = copy.deepcopy(temp)
                #print "---------  x=%d y=%d" % (x, y)
                nan_sum = 0
                lst_grid = []
                #print ("Start: " + str(Optimize2D.nan_img[x, y]))
                for col in np.arange(y, len(NanOptimizer.raw_img[0]), 3):
                    for row in np.arange(x, len(NanOptimizer.raw_img[:, 0]), 3):
                        for arr in NanOptimizer.img_indices[list(range(len(NanOptimizer.img_indices)))]:
                            el = np.asarray([row, col])
                            if (el == arr).all():
                                #print Optimize2D.nan_img[row, col]
                                nan_sum += NanOptimizer.nan_img[row, col]
                                lst_grid.append([row,col])
                                temp2[row,col] = np.max(temp)+1 #just to get some contrast to the center pixels

                NanOptimizer.dic_grid[(x,y)] = copy.deepcopy(lst_grid)
                NanOptimizer.nan_sums[(x,y)] = nan_sum
                percentage += 11.11

                # only to show a 3x3 subplot with all possibilities
                temp_arr.append(copy.deepcopy(temp2))
                offset += 1
                plt.subplot(3,3,offset)
                plt.imshow(temp_arr[offset-1], cmap='gray')
                plt.title("# of NANs: " + str(nan_sum))

                print(str(percentage) + "%")
        plt.show()
        print(NanOptimizer.nan_sums)

        return_value = 0

        # Find lowest number of NANs
        min_val = min(NanOptimizer.nan_sums.values())
        for key, val in NanOptimizer.nan_sums.items():
            if val == min_val:
                return_value = key

        # Optimal start pixel returned
        x, y = return_value

        # # Show the optimal center pixel distribution
        # # Therefore the nan-image is scaled for a better contrast
        # temp = copy.deepcopy(NanOptimizer.nan_img)
        #
        # temp[NanOptimizer.img_indices[:, 0], NanOptimizer.img_indices[:, 1]] = 2
        # not_one = np.argwhere(temp != 1)
        # temp[not_one[:,0], not_one[:,1]] = 0
        # #temp[self.cont_ind[:,0], self.cont_ind[:,1]] = 1
        #
        # for col in np.arange(y, len(NanOptimizer.raw_img[0]), 3):
        #     for row in np.arange(x, len(NanOptimizer.raw_img[:, 0]), 3):
        #         for arr in NanOptimizer.img_indices[range(len(NanOptimizer.img_indices))]:
        #             el = np.asarray([row, col])
        #             if (el == arr).all():
        #                 temp[row, col] = 3
        #
        # self.plot(img = temp)

        print(NanOptimizer.dic_grid[return_value])
        return return_value

    def calculate_tumor_coverage_slc(self, grid_points):
        print("CALC TUMOR COVERAGE PER SLICE")
        tt_sum = 0
        tt_total = NanOptimizer.slc_nof_tt[grid_points[0][0]]
        print("TOTAL:")
        print(tt_total)

        for gp in grid_points:
            tt_sum += NanOptimizer.tt_img_2D[gp[0]][gp[1], gp[2]]

        coverage = (tt_sum/tt_total)*100

        print("COV = %.2f %%" % coverage)
        print("END")
        return coverage

    def calculate_tumor_coverage_total(self, grid_points):
        print("CALC TUMOR COVERAGE TOTAL")
        cov_percentage_slices = []

        slices = np.unique(grid_points[:,0])

        for slc in slices:
            slices_points = grid_points[np.where(grid_points[:,0] == slc)]

            cov_percentage = 0
            tt_sum = 0

            for gp in slices_points:
                tt_sum += NanOptimizer.tt_img_2D[slc][gp[1], gp[2]]

            tt_total_slc = NanOptimizer.slc_nof_tt[slc]
            cov_percentage += (tt_sum / tt_total_slc) * 100

            if np.isnan(cov_percentage):
                cov_percentage = 0

            cov_percentage_slices.append(cov_percentage)

        print("Coverage per slice percentages:")
        print(cov_percentage_slices)

        print(len(slices))
        cov_percentage = sum(cov_percentage_slices) / len(slices)

        print("TOTAL TUMOR COVERAGE = %.2f %%" % cov_percentage)
        print("END")
        return cov_percentage

    def get_nan_sum(self, grid_points):
        print("CALC NAN_SUM")
        nan_sum_value = 0

        for gp in grid_points:
            nan_sum_value += NanOptimizer.nan_img[gp[0]][gp[1], gp[2]]

        print(nan_sum_value)
        print("END")

        return nan_sum_value

    def create_full_grid(self, dynamic=False):
        if dynamic == True:
            temp = []
            grid_positions = NanOptimizer.dic_grid
            for i in np.arange(0, len(grid_positions)):
                if len(grid_positions[i]):
                    temp.append(list(grid_positions[i]))
                    new = copy.deepcopy(grid_positions[i])
                    new[0] = grid_positions[i][0] - 1
                    temp.append(copy.deepcopy(list(new)))
                    new = copy.deepcopy(grid_positions[i])
                    new[0] = grid_positions[i][0] + 1
                    temp.append(copy.deepcopy(list(new)))

            NanOptimizer.dic_grid_full = copy.deepcopy(temp)
            NanOptimizer.dic_tot_tumcov["1"] = self.calculate_tumor_coverage_total(np.asarray(temp))
        else:
            for key in list(NanOptimizer.dic_grid.keys()):
                temp = []
                grid_positions = NanOptimizer.dic_grid[key]
                for i in np.arange(0, len(grid_positions)):
                    if len(grid_positions[i]):
                        temp.append(list(grid_positions[i]))
                        new = copy.deepcopy(grid_positions[i])
                        new[0] = grid_positions[i][0] - 1
                        temp.append(copy.deepcopy(list(new)))
                        new = copy.deepcopy(grid_positions[i])
                        new[0] = grid_positions[i][0] + 1
                        temp.append(copy.deepcopy(list(new)))

                NanOptimizer.dic_grid_full[key] = copy.deepcopy(temp)
                NanOptimizer.dic_tot_tumcov[key] = self.calculate_tumor_coverage_total(np.asarray(temp))

        print(NanOptimizer.dic_tot_tumcov)


    def get_grid_points_recurrence(self, dynamic=False):

        if dynamic == True:
            print("----------------------------- RECURRENCE COORDS ------------------------------------------")
            grid_positions = NanOptimizer.dic_grid_full
            temp = []
            for gp in grid_positions:
                if ~np.isnan(NanOptimizer.raw_rec_data[gp[0], gp[1], gp[2]]):
                    temp.append(gp)
            NanOptimizer.rec_grid = copy.deepcopy(temp)
        else:
            print("----------------------------- RECURRENCE COORDS ------------------------------------------")
            for key in list(NanOptimizer.dic_grid_full.keys()):
                grid_positions = NanOptimizer.dic_grid_full[key]
                temp = []
                for gp in grid_positions:
                    if ~np.isnan(NanOptimizer.raw_rec_data[gp[0],gp[1],gp[2]]):
                        temp.append(gp)
                NanOptimizer.rec_grid[key] = copy.deepcopy(temp)

        print("END")
        print(NanOptimizer.rec_grid)


