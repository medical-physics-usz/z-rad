import numpy as np
from ast import literal_eval
import matplotlib
import matplotlib.pyplot as plt
import Tkinter as tk
from Tkinter import LEFT, RIGHT, CENTER, BOTTOM, TOP
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from LocRadTexture import Texture
from Optimizer import NanOptimizer
import copy

class Content(tk.Frame):

    #lst_entries = ['Default folder', 'Start']
    def __init__(self, main_window, textures):
        tk.Frame.__init__(self)

        self.textures = []

        for tex in textures[0]:
            self.textures.append(tex)

        self.root = main_window
        self.default_folder = "D:\\Uni\\ETH\\MAS-Arbeit\\LocalRadiomics\\TestData\\15"

        #self.log(self.default_folder + " used for dicom readout")

        self.initialize_variables()

        self.create_controls()

        self.optimizer = 0

    def reset(self):

        temp_cov = self.chk_opt_overage
        temp_grid = self.chk_grid_dynamic

        for we in self.master.winfo_children():
            we.pack_forget()

        self.initialize_variables()

        self.chk_opt_overage = temp_cov
        self.chk_grid_dynamic = temp_grid

        self.create_controls()

    def initialize_variables(self):
        # Control definitions

        # Frames
        self.frm1, self.frm2, self.frm3, self.frm4, self.frm5, self.frm6 = 0, 0, 0, 0, 0, 0

        # Labels
        self.lbl1, self.lbl2, self.lbl3, self.lbl4, self.lbl5, self.lbl6, self.lbl7 = 0, 0, 0, 0, 0, 0, 0

        # Buttons
        self.btn1, self.btn2 = 0, 0
        self.chk1, self.chk2 = 0, 0

        # Text fields
        self.txt1 = 0

        #Entries
        self.e1, self.e2, self.e3 = 0, 0, 0

        # Scales
        self.sld1 = 0
        self.list_box1 = 0
        self.scrollbar_listbox1 = 0

        # Canvas
        self.cnv, self.cnv2, self.cnv3 = 0, 0, 0
        self.fig, self.fig2, self.fig3 = 0, 0, 0
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5 = 0, 0, 0, 0, 0
        self.image_left = 0
        self.image_right = 0
        self.image_compare = 0
        self.image_comp_rec = 0

        # Variable definitions
        self.slider_val = 0
        self.pixel_data = []
        self.pixel_data_roi = []
        self.pixel_data_roi_rec = []
        self.z_pos = tk.StringVar()
        self.nof_roi_slices = tk.StringVar()
        self.combo_var = tk.StringVar()
        self.current_slice = tk.StringVar()
        self.chk_opt_overage = tk.IntVar()
        self.chk_grid_dynamic = tk.IntVar()
        self.slice_coverage = tk.StringVar()
        self.slice_coverage_total = tk.StringVar()
        self.nan_sum = tk.StringVar()
        self.cont_dim_x = []
        self.cont_dim_y = []
        self.roi_offset = -1
        self.roi_offset_rec = -1


    def add_center_slices(self):
        self.list_box1.delete(0, tk.END)
        for pos in self.grid_position:
            self.list_box1.insert(tk.END, pos)


    def on_combo_change(self, val):
        self.grid_key = val.split('|')[0]
        self.grid_position = np.asarray(copy.deepcopy(NanOptimizer.dic_grid[literal_eval(val.split('|')[0])]))
        self.add_center_slices()

        self.nan_sum_value = self.optimizer.get_nan_sum(self.grid_position)
        self.grid_position = np.asarray(copy.deepcopy(NanOptimizer.dic_grid_full[literal_eval(val.split('|')[0])]))
        coverage_total = self.optimizer.calculate_tumor_coverage_total(self.grid_position)
        print "-----------------------------------RATIO--------------------------------------------"
        print coverage_total / self.nan_sum_value
        print "------------------------------------------------------------------------------------"
        self.slice_coverage_total.set("Total tumor coverage: %.2f %%" % coverage_total)

        self.cnv3.draw()

    def get_tumor_coverage_slc(self, grid_points):
        coverage = self.optimizer.calculate_tumor_coverage_slc(grid_points)
        self.slice_coverage.set("Tumor coverage current slice: %.2f %%" % coverage)
        self.nan_sum.set("# of NANs: %d " % self.nan_sum_value)

    def on_value_changed(self, val):
        self.slider_val = val
        pd = self.pixel_data[int(val)]
        self.image_left.set_data(pd)
        ind = ~np.isnan(pd)
        min_limit = np.min(pd[ind])
        max_limit = np.max(pd[ind])
        self.image_left.set_clim(vmin=min_limit, vmax=max_limit)

        ind_rec = []
        img_rec_drawn = False

        if len(self.pixel_data_roi[int(self.slider_val)]):
            pd_roi = self.pixel_data_roi[int(val)][0]
            if len(self.pixel_data_roi_rec[int(self.slider_val)]):
                pd_roi_rec = self.pixel_data_roi_rec[int(val)][0]
                ind_rec = ~np.isnan(pd_roi_rec)
            ind = ~np.isnan(pd_roi)
            if True in ind:
                self.image_right.set_data(pd_roi)
                min_limit = np.min(pd_roi[ind])
                max_limit = np.max(pd_roi[ind])
                self.image_right.set_clim(vmin=min_limit, vmax=max_limit)
                if True in ind_rec:
                    print "RECURRENCE"
                    self.fig2.hold()
                    self.image_rec.set_data(pd_roi_rec)
                    min_limit = np.min(pd_roi_rec[ind_rec])
                    max_limit = np.max(pd_roi_rec[ind_rec])
                    self.image_rec.set_clim(vmin=min_limit, vmax=max_limit)
                    self.fig2.hold()
                else:
                    temp = np.zeros(shape=(self.base_shape[0], self.base_shape[1]))
                    temp.fill(np.nan)
                    self.image_rec.set_data(temp)
            else:
                temp = np.zeros(shape=(self.base_shape[0], self.base_shape[1]))
                temp.fill(np.nan)
                self.image_right.set_data(temp)


            # if self.grid_position[:,0].__contains__(int(val)-self.roi_offset):
            roi_slice = int(val)-self.roi_offset
            grid_points = self.grid_position[np.where(self.grid_position[:,0] == roi_slice)]
            self.current_slice.set("Current ROI- slice = " + str(roi_slice))
            print grid_points
            temp = copy.deepcopy(pd_roi)
            rec_temp = np.zeros(shape=(self.base_shape[0], self.base_shape[1]))
            rec_temp.fill(np.nan)
            if len(grid_points):
                self.get_tumor_coverage_slc(grid_points)
                min_limit = np.min(temp[ind]) - 30
                max_limit = np.max(temp[ind])
                for gp in grid_points:
                    temp[gp[1],gp[2]] = min_limit
                    if self.chk_grid_dynamic.get() == 0:
                        gp_rec = NanOptimizer.rec_grid[literal_eval(self.grid_key)]
                        if list(gp) in gp_rec:
                            rec_temp[gp[1], gp[2]] = max_limit
                    else:
                        gp_rec = NanOptimizer.rec_grid
                        if list(gp) in gp_rec:
                             rec_temp[gp[1], gp[2]] = max_limit

                self.image_compare.set_data(temp)
                self.image_compare.set_clim(vmin=min_limit, vmax=max_limit)
                self.fig3.hold()
                self.image_comp_rec.set_data(rec_temp)
                self.image_comp_rec.set_clim(vmin=min_limit, vmax=max_limit)
                self.fig3.hold()
            else:
                temp = np.zeros(shape=(self.base_shape[0], self.base_shape[1]))
                temp.fill(np.nan)
                self.image_compare.set_data(temp)
                self.image_comp_rec.set_data(temp)

        # z_pos = Texture.slices[int(val)].slice_location
        z_pos = self.textures[0].slice_data[int(val)].slice_location
        self.z_pos.set("Slice position: %.2f " % z_pos)

        # self.lbl3 = tk.Label(self.frm4, textvariable=self.z_pos)
        self.cnv.draw()
        self.cnv2.draw()
        self.cnv3.draw()


    def btn_read(self):
        self.reset()
        slices = 0
        shape = 0
        shape_rec = 0
        ind = 0
        ind_rec = 0

        #self.pixel_data = Texture.matrix
        for pxd in self.textures[0].slice_data:#Texture.slices:
            self.pixel_data.append(pxd.pixel_data)
            self.pixel_data_roi.append(pxd.pixel_data_roi)
            if len(pxd.pixel_data_roi):
                slices+=1
                shape = pxd.pixel_data_roi[0].shape
                if self.roi_offset == -1:
                    self.roi_offset = ind
            ind += 1

        self.base_shape = shape
        print "SHAPE PTV:"
        print shape
        print "_----------------------------------------------------------------------------------------_"
        print "ROI OFFSET: " + str(self.roi_offset)
        self.nof_roi_slices.set("Number of roi-slices: " + str(slices))

        for pxd in self.textures[1].slice_data:#Texture.slices:
            self.pixel_data_roi_rec.append(pxd.pixel_data_roi)
            if len(pxd.pixel_data_roi):
                shape_rec = pxd.pixel_data_roi[0].shape
                if self.roi_offset_rec == -1:
                    self.roi_offset_rec = ind_rec
            ind_rec += 1

        print len(self.pixel_data_roi)
        print len(self.pixel_data_roi_rec)

        self.ax1 = self.fig.add_subplot(111)
        self.image_left = self.ax1.imshow(self.pixel_data[0], cmap='gray')
        self.ax2 = self.fig2.add_subplot(111)
        self.image_right = self.ax2.imshow(self.pixel_data[0], cmap='gray', extent=(0,shape[1],shape[0],0), aspect='auto')
        self.ax2.set_xticks(np.arange(0, shape[1], 2))
        self.ax2.set_yticks(np.arange(0,shape[0], 2))
        self.ax2.grid(color='red', linestyle='-', linewidth=0.2)

        self.ax4 = self.fig2.add_subplot(111)
        temp = np.zeros(shape=(shape[0], shape[1]))
        temp.fill(np.nan)
        self.image_rec = self.ax4.imshow(temp, alpha=0.6,
                                         extent=(0, self.base_shape[1], self.base_shape[0], 0), aspect='auto')
        self.ax4.xaxis.set_visible(True)
        self.ax4.yaxis.set_visible(True)

        self.cnv.draw()
        self.cnv2.draw()

        # Create slider (scale)- widget to click through all slices
        self.sld1 = tk.Scale(self.frm3, from_=0, to=len(self.pixel_data)-1,
                             command=self.on_value_changed,
                             sliderlength=20,
                             length=200)
        self.sld1.pack(side=RIGHT, padx=10)

        slice = 0
        temp = np.zeros(shape=(slices, shape[0], shape[1]))

        for roidat in self.pixel_data_roi:
            if len(roidat):
                print roidat[0]
                temp[slice] = roidat[0]
                slice += 1

        slice = self.roi_offset_rec - self.roi_offset
        temp_rec = np.zeros(shape=(slices, shape[0], shape[1]))
        temp_rec.fill(np.nan)

        for roidat in self.pixel_data_roi_rec:
            if len(roidat):
                print roidat[0]
                temp_rec[slice] = roidat[0]
                slice += 1

        self.optimizer = NanOptimizer(VOI_matrix=temp, REC_matrix=temp_rec,
                            max_tumor_coverage=self.chk_opt_overage.get(),
                            dynamic_grid=self.chk_grid_dynamic.get())
        self.optimizer.minimize_nan_contribtion()

        if self.chk_grid_dynamic.get() == 0:
            comboChoices = []

            for i in range(len(NanOptimizer.nan_sums)):
                key = NanOptimizer.nan_sums.keys()[i]
                value = NanOptimizer.nan_sums.values()[i]
                cov = np.round(NanOptimizer.dic_tot_tumcov[key])
                comboChoices.append(str(key) + "|" + str(value) + "|" + str(cov))

            self.combo_var.set(comboChoices[0])

            self.popupMenu = tk.OptionMenu(self.frm5, self.combo_var, *comboChoices, command=self.on_combo_change)
            self.popupMenu.pack(side=TOP)

        self.fig3 = matplotlib.pyplot.Figure(figsize=(3, 3))
        self.cnv3 = FigureCanvasTkAgg(self.fig3, self.frm6)
        self.cnv3.get_tk_widget().pack(side=RIGHT, padx=20)
        self.ax3 = self.fig3.add_subplot(111)
        self.image_compare = self.ax3.imshow(self.pixel_data[0], cmap='gray', extent=(0,shape[1],shape[0],0), aspect='auto')
        self.ax3.set_xticks(np.arange(0, shape[1], 2))
        self.ax3.set_yticks(np.arange(0,shape[0], 2))
        self.ax3.grid(color='red', linestyle='-', linewidth=0.2)

        self.ax5 = self.fig3.add_subplot(111)
        temp = np.zeros(shape=(shape[0], shape[1]))
        temp.fill(np.nan)
        self.image_comp_rec = self.ax5.imshow(temp, alpha=0.6,
                                         extent=(0, self.base_shape[1], self.base_shape[0], 0), aspect='auto')
        self.ax5.xaxis.set_visible(True)
        self.ax5.yaxis.set_visible(True)

        self.list_box1 = tk.Listbox(self.frm5)
        self.scrollbar_listbox1 = tk.Scrollbar(self.frm5, command=self.list_box1.yview)
        self.list_box1.configure(yscrollcommand=self.scrollbar_listbox1.set)
        self.list_box1.pack(side=LEFT)
        self.scrollbar_listbox1.pack(side=RIGHT, fill=tk.Y)

        self.nan_sum.set("# of NANs: ")
        self.lbl6 = tk.Label(self.frm6, textvariable=self.nan_sum)
        self.lbl6.pack(padx=20, pady=10)

        self.slice_coverage.set("Tumor coverage current slice: ")
        self.lbl5 = tk.Label(self.frm6, textvariable=self.slice_coverage)
        self.lbl5.pack(padx=20, pady=10)

        self.slice_coverage_total.set("Total tumor coverage: ")
        self.lbl7 = tk.Label(self.frm6, textvariable=self.slice_coverage_total)
        self.lbl7.pack(padx=20, pady=10)

        self.cnv3.draw()

        if self.chk_grid_dynamic.get() == 1:
            self.grid_position = NanOptimizer.dic_grid
            self.add_center_slices()
            self.nan_sum_value = self.optimizer.get_nan_sum(self.grid_position)
            self.grid_position = np.asarray(NanOptimizer.dic_grid_full)
            coverage_total = self.optimizer.calculate_tumor_coverage_total(self.grid_position)
            print "-----------------------------------RATIO--------------------------------------------"
            print coverage_total / self.nan_sum_value
            print "------------------------------------------------------------------------------------"
            self.slice_coverage_total.set("Total tumor coverage: %.2f %%" % coverage_total)

    def create_controls(self):
        self.frm1 = tk.Frame(self.root)
        self.frm1.pack()
        self.frm2 = tk.Frame(self.root)
        self.frm2.pack()
        self.frm3 = tk.Frame(self.root)
        self.frm3.pack()
        self.frm4 = tk.Frame(self.root)
        self.frm4.pack()
        self.frm5 = tk.Frame(self.root)
        self.frm5.pack()
        self.frm6 = tk.Frame(self.frm5)
        self.frm6.pack(side=RIGHT)

        self.lbl1 = tk.Label(self.frm1, text="Local radiomics application\n\n")
        self.lbl1.pack(side=TOP)
        self.chk1 = tk.Checkbutton(self.frm1, text="Maximize tumor coverage", variable=self.chk_opt_overage)
        self.chk1.pack(side=LEFT)
        self.lbl2 = tk.Label(self.frm1, text="Image folder: ")
        self.lbl2.pack(side=LEFT)
        self.txt1 = tk.Label(self.frm1, text=self.default_folder)
        self.txt1.pack(side=LEFT, pady=10, padx=10)

        self.chk2 = tk.Checkbutton(self.frm2, text="Dynamic grid", variable=self.chk_grid_dynamic)
        self.chk2.pack(side=TOP, pady=10, padx=10)
        self.btn1 = tk.Button(self.frm2, text="Optimize", command=self.btn_read)
        self.btn1.pack(side=TOP, pady=10, padx=10)


        self.fig = matplotlib.pyplot.Figure(figsize=(3, 3))
        self.cnv = FigureCanvasTkAgg(self.fig, self.frm3)
        self.cnv.get_tk_widget().pack(side=LEFT)

        self.fig2 = matplotlib.pyplot.Figure(figsize=(3, 3))
        self.cnv2 = FigureCanvasTkAgg(self.fig2, self.frm3)
        self.cnv2.get_tk_widget().pack(side=RIGHT)

        self.z_pos.set("Slice position = ")
        self.lbl3 = tk.Label(self.frm4, textvariable=self.z_pos)
        self.lbl3.pack(side=LEFT, pady=10)

        self.nof_roi_slices.set("Number of roi-slices = ")
        self.lbl3 = tk.Label(self.frm4, textvariable=self.nof_roi_slices)
        self.lbl3.pack(side=RIGHT, pady=10)

        self.current_slice.set("Current ROI- slice = ")
        self.lbl4 = tk.Label(self.frm5, textvariable=self.current_slice)
        self.lbl4.pack(side=TOP, pady=10)


    def SetConfigData(self, cfg):
        print cfg
        self.default_folder = str(cfg[0][0])
        self.ROIs = str(cfg[0][1])
        self.HUmin = str(cfg[0][2])
        self.HUmax = str(cfg[0][3])


    #def log(self, message):
        #Common.log_handler.log("GUI", message)

def CreateGUI(path_image, path_save, structure, pixNr, binSize, l_ImName, HUmin, HUmax, outlier_corr,wv):
    textures = []
    for struct in structure:
        print "struct: " + struct
        temp=[]
        temp.append(struct)
        textures.append(Texture(path_image, path_save, temp, pixNr, binSize, l_ImName, HUmin, HUmax, outlier_corr, wv))
    # Texture(path_image, path_save, structure, pixNr, binSize, l_ImName, HUmin, HUmax, outlier_corr, wv)
    print len(textures)
    config_data=[]
    config_data.append(path_image)
    config_data.append(structure)
    config_data.append(HUmin)
    config_data.append(HUmax)
    main(textures)

def main(*args):
    main_window = tk.Tk()
    main_window.title("Local radiomics GUI")
    main_window.geometry("800x800")
    app = Content(main_window, args)
    # app.SetConfigData(args)
    app.mainloop()

if __name__ == "__main__":
    print "Please start main_texture.py first"
    pass
