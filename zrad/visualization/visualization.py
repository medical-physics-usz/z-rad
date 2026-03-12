import os
import sys
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore, QtGui


class SliceView(pg.GraphicsLayoutWidget):
    sigScrolled = QtCore.pyqtSignal(object, int)
    sigClicked = QtCore.pyqtSignal(object, float, float)

    def __init__(self, title="", parent=None):
        super().__init__(parent=parent)

        self.setBackground("k")
        self.plot = self.addPlot(title=title)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)

        self.vb = self.plot.getViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.enableAutoRange(x=False, y=False)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            ev.ignore()
            return
        self.sigScrolled.emit(self, 1 if delta > 0 else -1)
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos = self.vb.mapSceneToView(ev.pos())
            self.sigClicked.emit(self, pos.x(), pos.y())
            ev.accept()
            return
        super().mousePressEvent(ev)


class ColorLegendWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout_ = QtWidgets.QHBoxLayout(self)
        self.layout_.setContentsMargins(8, 4, 8, 4)
        self.layout_.setSpacing(14)
        self.layout_.addStretch(1)

        self.setStyleSheet("background-color: #222;")

    def set_items(self, items):
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for name, rgb in items:
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(6)

            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
                "border: 1px solid #888;"
            )

            label = QtWidgets.QLabel(name)
            label.setStyleSheet("color: white;")

            item_layout.addWidget(swatch)
            item_layout.addWidget(label)
            self.layout_.addWidget(item_widget)

        self.layout_.addStretch(1)


class Visualization(QtWidgets.QMainWindow):
    def __init__(self, image_sets):
        super().__init__()
        self.setWindowTitle("Z-Rad Viewer")

        if not image_sets:
            raise ValueError("image_sets must contain at least one image.")

        self.image_sets = image_sets
        self.current_image_index = 0

        self.mask_alpha = 0.35
        self.crosshair_pen = pg.mkPen((255, 0, 0, 180), width=1)

        self.mask_colors = [
            (0, 255, 0),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 255),
            (255, 165, 0),
            (0, 191, 255),
            (0, 255, 127),
            (255, 105, 180),
            (255, 215, 0),
            (30, 144, 255),
            (238, 130, 238),
            (127, 255, 0),
        ]

        self.gray_lut = np.empty((256, 3), dtype=np.uint8)
        self.gray_lut[:, 0] = np.arange(256, dtype=np.uint8)
        self.gray_lut[:, 1] = np.arange(256, dtype=np.uint8)
        self.gray_lut[:, 2] = np.arange(256, dtype=np.uint8)

        self.volume = None
        self.masks = []
        self.nifti_path = None
        self.current_case_name = ""
        self.nx = self.ny = self.nz = 0
        self.sx = self.sy = self.sz = 1.0
        self.vmin = 0.0
        self.vmax = 1.0

        self.current_sagittal = 0   # x
        self.current_coronal = 0    # y
        self.current_axial = 0      # z

        self._build_ui()
        self._connect_events()
        self._init_views()

        self._load_image_set(0)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.main_layout = QtWidgets.QVBoxLayout(central)

        views_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(views_layout)

        self.sag_view = SliceView("Sagittal")
        self.cor_view = SliceView("Coronal")
        self.axi_view = SliceView("Axial")

        views_layout.addWidget(self.sag_view)
        views_layout.addWidget(self.cor_view)
        views_layout.addWidget(self.axi_view)

        self.legend_widget = ColorLegendWidget()
        self.main_layout.addWidget(self.legend_widget)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)

        self.prev_button = QtWidgets.QPushButton("Previous Image")
        self.next_button = QtWidgets.QPushButton("Next Image")
        self.close_button = QtWidgets.QPushButton("Close")

        self.prev_button.setMinimumHeight(34)
        self.next_button.setMinimumHeight(34)
        self.close_button.setMinimumHeight(34)

        button_style = """
        QPushButton {
            background-color: orange;
            color: black;
            border-radius: 6px;
            padding: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #ffb347;
        }
        QPushButton:pressed {
            background-color: #e69500;
        }
        """

        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)
        self.close_button.setStyleSheet(button_style)

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.close_button)

        self.main_layout.addLayout(button_layout)

        self.statusBar().showMessage("Ready")

    def _connect_events(self):
        self.sag_view.sigScrolled.connect(self._on_view_scrolled)
        self.cor_view.sigScrolled.connect(self._on_view_scrolled)
        self.axi_view.sigScrolled.connect(self._on_view_scrolled)

        self.sag_view.sigClicked.connect(self._on_view_clicked)
        self.cor_view.sigClicked.connect(self._on_view_clicked)
        self.axi_view.sigClicked.connect(self._on_view_clicked)

        self.prev_button.clicked.connect(self._show_previous_image)
        self.next_button.clicked.connect(self._show_next_image)
        self.close_button.clicked.connect(self.close)

    def _make_gray_item(self):
        item = pg.ImageItem(axisOrder="row-major")
        item.setLookupTable(self.gray_lut)
        item.setLevels([0, 255])
        item.setZValue(0)
        return item

    def _make_rgba_item(self):
        item = pg.ImageItem(axisOrder="row-major")
        item.setZValue(1)
        return item

    def _set_item_transform(self, item, x_spacing, y_spacing):
        tr = QtGui.QTransform()
        tr.scale(x_spacing, y_spacing)
        item.setTransform(tr)

    def _init_views(self):
        self.sag_img = self._make_gray_item()
        self.cor_img = self._make_gray_item()
        self.axi_img = self._make_gray_item()

        self.sag_view.vb.addItem(self.sag_img)
        self.cor_view.vb.addItem(self.cor_img)
        self.axi_view.vb.addItem(self.axi_img)

        self.sag_mask_items = []
        self.cor_mask_items = []
        self.axi_mask_items = []

        self.sag_vline = pg.InfiniteLine(angle=90, movable=False, pen=self.crosshair_pen)
        self.sag_hline = pg.InfiniteLine(angle=0, movable=False, pen=self.crosshair_pen)
        self.cor_vline = pg.InfiniteLine(angle=90, movable=False, pen=self.crosshair_pen)
        self.cor_hline = pg.InfiniteLine(angle=0, movable=False, pen=self.crosshair_pen)
        self.axi_vline = pg.InfiniteLine(angle=90, movable=False, pen=self.crosshair_pen)
        self.axi_hline = pg.InfiniteLine(angle=0, movable=False, pen=self.crosshair_pen)

        self.sag_vline.setZValue(2)
        self.sag_hline.setZValue(2)
        self.cor_vline.setZValue(2)
        self.cor_hline.setZValue(2)
        self.axi_vline.setZValue(2)
        self.axi_hline.setZValue(2)

        self.sag_view.vb.addItem(self.sag_vline)
        self.sag_view.vb.addItem(self.sag_hline)
        self.cor_view.vb.addItem(self.cor_vline)
        self.cor_view.vb.addItem(self.cor_hline)
        self.axi_view.vb.addItem(self.axi_vline)
        self.axi_view.vb.addItem(self.axi_hline)

    def _clear_mask_items(self):
        for item in self.sag_mask_items:
            self.sag_view.vb.removeItem(item)
        for item in self.cor_mask_items:
            self.cor_view.vb.removeItem(item)
        for item in self.axi_mask_items:
            self.axi_view.vb.removeItem(item)

        self.sag_mask_items = []
        self.cor_mask_items = []
        self.axi_mask_items = []

    def _rebuild_mask_items(self):
        self._clear_mask_items()

        for _ in self.masks:
            sag_item = self._make_rgba_item()
            cor_item = self._make_rgba_item()
            axi_item = self._make_rgba_item()

            self.sag_view.vb.addItem(sag_item)
            self.cor_view.vb.addItem(cor_item)
            self.axi_view.vb.addItem(axi_item)

            self.sag_mask_items.append(sag_item)
            self.cor_mask_items.append(cor_item)
            self.axi_mask_items.append(axi_item)

        self._apply_item_transforms()

    def _apply_item_transforms(self):
        # sagittal: displayed x=y, y=z
        self._set_item_transform(self.sag_img, self.sy, self.sz)
        # coronal: displayed x=x, y=z
        self._set_item_transform(self.cor_img, self.sx, self.sz)
        # axial: displayed x=x, y=y
        self._set_item_transform(self.axi_img, self.sx, self.sy)

        for item in self.sag_mask_items:
            self._set_item_transform(item, self.sy, self.sz)
        for item in self.cor_mask_items:
            self._set_item_transform(item, self.sx, self.sz)
        for item in self.axi_mask_items:
            self._set_item_transform(item, self.sx, self.sy)

    def _load_image_set(self, index):
        if index < 0 or index >= len(self.image_sets):
            return

        item = self.image_sets[index]
        image = item["image"]
        masks = item.get("masks", [])

        self.volume = np.asarray(image.array)
        self.current_case_name = item["image_name"]

        # volume array order: (z, y, x)
        self.nz, self.ny, self.nx = self.volume.shape

        # spacing order: (x, y, z)
        self.sx, self.sy, self.sz = image.spacing

        finite_vals = self.volume[np.isfinite(self.volume)]

        if finite_vals.size == 0:
            # Fully invalid / corrupted volume: keep a harmless default window
            self.vmin = 0.0
            self.vmax = 1.0
        else:
            self.vmin = float(np.percentile(finite_vals, 1))
            self.vmax = float(np.percentile(finite_vals, 99))

            if not np.isfinite(self.vmin) or not np.isfinite(self.vmax) or self.vmax <= self.vmin:
                self.vmin = float(np.min(finite_vals))
                self.vmax = float(np.max(finite_vals))
                if self.vmax <= self.vmin:
                    self.vmax = self.vmin + 1.0

        self.masks = self._load_masks_for_volume(masks)

        self.current_image_index = index
        self.current_sagittal = self.nx // 2
        self.current_coronal = self.ny // 2
        self.current_axial = self.nz // 2

        self._rebuild_mask_items()

        legend_items = [(m["name"], m["color"]) for m in self.masks]
        self.legend_widget.set_items(legend_items)
        self.legend_widget.setVisible(len(legend_items) > 0)

        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_sets) - 1)

        self._update_all_views()

    def _load_masks_for_volume(self, masks):
        loaded_masks = []

        for i, mask_instance in enumerate(masks):
            mask_name, mask = next(iter(mask_instance.items()))
            mask_bool = np.asarray(mask.array > 0, dtype=np.uint8)
            color = self.mask_colors[i % len(self.mask_colors)]

            loaded_masks.append(
                {
                    "name": mask_name,
                    "data": mask_bool,
                    "color": color,
                }
            )

        return loaded_masks

    def _clip_index(self, value, upper):
        return max(0, min(int(round(value)), upper))

    def _normalize_to_u8(self, arr):
        arr = np.nan_to_num(arr, nan=self.vmin, posinf=self.vmax, neginf=self.vmin)
        arr = np.clip(arr, self.vmin, self.vmax)
        arr = ((arr - self.vmin) / (self.vmax - self.vmin) * 255.0).astype(np.uint8)
        return arr

    def _mask_to_rgba(self, mask_slice_2d, rgb):
        h, w = mask_slice_2d.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = rgb[0]
        rgba[..., 1] = rgb[1]
        rgba[..., 2] = rgb[2]
        rgba[..., 3] = mask_slice_2d.astype(np.uint8) * int(255 * self.mask_alpha)
        return rgba

    def _rot180(self, arr):
        return np.ascontiguousarray(arr[::-1, ::-1])

    def _get_sagittal_slice(self, x):
        # base slice: (z, y)
        return self._rot180(self.volume[:, :, x])

    def _get_coronal_slice(self, y):
        # base slice: (z, x)
        return self._rot180(self.volume[:, y, :])

    def _get_axial_slice(self, z):
        # base slice: (y, x)
        return self._rot180(self.volume[z, :, :])

    def _get_sagittal_mask(self, mask, x):
        return self._rot180(mask[:, :, x])

    def _get_coronal_mask(self, mask, y):
        return self._rot180(mask[:, y, :])

    def _get_axial_mask(self, mask, z):
        return self._rot180(mask[z, :, :])

    def _set_view_extent(self, view, width_phys, height_phys):
        view.vb.setRange(
            xRange=(0, width_phys),
            yRange=(0, height_phys),
            padding=0.0,
            disableAutoRange=True,
        )

    def _update_crosshairs(self):
        # because displayed images are rotated 180°, both display axes are reversed

        # sagittal view: display x <- reversed y, display y <- reversed z
        self.sag_vline.setPos((self.ny - 1 - self.current_coronal) * self.sy)
        self.sag_hline.setPos((self.nz - 1 - self.current_axial) * self.sz)

        # coronal view: display x <- reversed x, display y <- reversed z
        self.cor_vline.setPos((self.nx - 1 - self.current_sagittal) * self.sx)
        self.cor_hline.setPos((self.nz - 1 - self.current_axial) * self.sz)

        # axial view: display x <- reversed x, display y <- reversed y
        self.axi_vline.setPos((self.nx - 1 - self.current_sagittal) * self.sx)
        self.axi_hline.setPos((self.ny - 1 - self.current_coronal) * self.sy)

    def _update_titles(self):
        prefix = f"[{self.current_image_index + 1}/{len(self.image_sets)}] {self.current_case_name}"

        self.sag_view.plot.setTitle(f"{prefix} | Sagittal {self.current_sagittal}", color="w")
        self.cor_view.plot.setTitle(f"{prefix} | Coronal {self.current_coronal}", color="w")
        self.axi_view.plot.setTitle(f"{prefix} | Axial {self.current_axial}", color="w")

    def _get_current_voxel_value(self):
        value = self.volume[self.current_axial, self.current_coronal, self.current_sagittal]
        return float(value) if np.isfinite(value) else np.nan

    def _build_status_text(self):
        voxel_value = self._get_current_voxel_value()
        voxel_str = "nan" if np.isnan(voxel_value) else f"{voxel_value:.3f}"
        return (
            f"Image {self.current_image_index + 1}/{len(self.image_sets)} | "
            f"Case: {self.current_case_name} | "
            f"Shape: {self.volume.shape} | "
            f"Spacing: ({self.sx:.3f}, {self.sy:.3f}, {self.sz:.3f}) mm | "
            f"Voxel: ({self.current_sagittal}, {self.current_coronal}, {self.current_axial}) | "
            f"Intensity: {voxel_str}"
        )

    def _update_all_views(self):
        sag = self._normalize_to_u8(self._get_sagittal_slice(self.current_sagittal))
        cor = self._normalize_to_u8(self._get_coronal_slice(self.current_coronal))
        axi = self._normalize_to_u8(self._get_axial_slice(self.current_axial))

        self.sag_img.setImage(sag, autoLevels=False)
        self.cor_img.setImage(cor, autoLevels=False)
        self.axi_img.setImage(axi, autoLevels=False)

        for i, m in enumerate(self.masks):
            self.sag_mask_items[i].setImage(
                self._mask_to_rgba(
                    self._get_sagittal_mask(m["data"], self.current_sagittal),
                    m["color"],
                ),
                autoLevels=False,
            )
            self.cor_mask_items[i].setImage(
                self._mask_to_rgba(
                    self._get_coronal_mask(m["data"], self.current_coronal),
                    m["color"],
                ),
                autoLevels=False,
            )
            self.axi_mask_items[i].setImage(
                self._mask_to_rgba(
                    self._get_axial_mask(m["data"], self.current_axial),
                    m["color"],
                ),
                autoLevels=False,
            )

        self._set_view_extent(self.sag_view, self.ny * self.sy, self.nz * self.sz)
        self._set_view_extent(self.cor_view, self.nx * self.sx, self.nz * self.sz)
        self._set_view_extent(self.axi_view, self.nx * self.sx, self.ny * self.sy)

        self._update_crosshairs()
        self._update_titles()
        self.statusBar().showMessage(self._build_status_text())

    def _on_view_scrolled(self, view, step):
        if view is self.sag_view:
            self.current_sagittal = self._clip_index(self.current_sagittal + step, self.nx - 1)
        elif view is self.cor_view:
            self.current_coronal = self._clip_index(self.current_coronal + step, self.ny - 1)
        elif view is self.axi_view:
            self.current_axial = self._clip_index(self.current_axial + step, self.nz - 1)

        self._update_all_views()

    def _on_view_clicked(self, view, x_phys, y_phys):
        if view is self.sag_view:
            self.current_coronal = self._clip_index((self.ny - 1) - (x_phys / self.sy), self.ny - 1)
            self.current_axial = self._clip_index((self.nz - 1) - (y_phys / self.sz), self.nz - 1)

        elif view is self.cor_view:
            self.current_sagittal = self._clip_index((self.nx - 1) - (x_phys / self.sx), self.nx - 1)
            self.current_axial = self._clip_index((self.nz - 1) - (y_phys / self.sz), self.nz - 1)

        elif view is self.axi_view:
            self.current_sagittal = self._clip_index((self.nx - 1) - (x_phys / self.sx), self.nx - 1)
            self.current_coronal = self._clip_index((self.ny - 1) - (y_phys / self.sy), self.ny - 1)

        self._update_all_views()

    def _show_previous_image(self):
        if self.current_image_index > 0:
            self._load_image_set(self.current_image_index - 1)

    def _show_next_image(self):
        if self.current_image_index < len(self.image_sets) - 1:
            self._load_image_set(self.current_image_index + 1)