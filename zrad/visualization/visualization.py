import os
import sys
import warnings
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore, QtGui


class SliceView(pg.GraphicsLayoutWidget):
    sigScrolled = QtCore.pyqtSignal(object, int)
    sigClicked = QtCore.pyqtSignal(object, float, float)
    sigZoomed = QtCore.pyqtSignal(object, float, float, float)
    sigResetView = QtCore.pyqtSignal(object)

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

        self._pan_active = False
        self._pan_last_scene_pos = None

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            ev.ignore()
            return

        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers & QtCore.Qt.ControlModifier:
            pos = self.vb.mapSceneToView(ev.pos())
            zoom_factor = 0.8 if delta > 0 else 1.25
            self.sigZoomed.emit(self, zoom_factor, pos.x(), pos.y())
            ev.accept()
            return

        self.sigScrolled.emit(self, 1 if delta > 0 else -1)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.sigResetView.emit(self)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MiddleButton:
            self._pan_active = True
            self._pan_last_scene_pos = ev.pos()
            ev.accept()
            return

        if ev.button() == QtCore.Qt.LeftButton:
            pos = self.vb.mapSceneToView(ev.pos())
            self.sigClicked.emit(self, pos.x(), pos.y())
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_active and self._pan_last_scene_pos is not None:
            old_view = self.vb.mapSceneToView(self._pan_last_scene_pos)
            new_view = self.vb.mapSceneToView(ev.pos())
            delta = old_view - new_view
            self.vb.translateBy(x=delta.x(), y=delta.y())
            self._pan_last_scene_pos = ev.pos()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.MiddleButton:
            self._pan_active = False
            self._pan_last_scene_pos = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


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
        self.crosshair_pen = pg.mkPen((255, 0, 0, 220), width=1)

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

        self.current_sagittal = 0
        self.current_coronal = 0
        self.current_axial = 0

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

        self.sag_view.sigZoomed.connect(self._on_view_zoomed)
        self.cor_view.sigZoomed.connect(self._on_view_zoomed)
        self.axi_view.sigZoomed.connect(self._on_view_zoomed)

        self.sag_view.sigResetView.connect(self._reset_single_view)
        self.cor_view.sigResetView.connect(self._reset_single_view)
        self.axi_view.sigResetView.connect(self._reset_single_view)

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

        for line in (
            self.sag_vline, self.sag_hline,
            self.cor_vline, self.cor_hline,
            self.axi_vline, self.axi_hline,
        ):
            line.setZValue(10)

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
        self._set_item_transform(self.sag_img, self.sy, self.sz)
        self._set_item_transform(self.cor_img, self.sx, self.sz)
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

        self.nz, self.ny, self.nx = self.volume.shape
        self.sx, self.sy, self.sz = image.spacing

        finite_vals = self.volume[np.isfinite(self.volume)]

        if finite_vals.size == 0:
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

        self.masks = self._load_masks_for_volume(image, masks)

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
        QtCore.QTimer.singleShot(0, self._reset_all_view_ranges)

    def _to_tuple(self, value):
        if value is None:
            return None
        return tuple(np.asarray(value).tolist())

    def _same_exact_shape(self, image, mask):
        image_shape = tuple(np.asarray(image.array).shape)
        mask_shape = tuple(np.asarray(mask.array).shape)
        return image_shape == mask_shape

    def _same_close_tuple(self, lhs, rhs, atol=1e-6):
        lhs_tuple = self._to_tuple(lhs)
        rhs_tuple = self._to_tuple(rhs)

        if lhs_tuple is None or rhs_tuple is None:
            return False

        if len(lhs_tuple) != len(rhs_tuple):
            return False

        return bool(np.allclose(lhs_tuple, rhs_tuple, atol=atol, rtol=0.0))

    def _metadata_matches_image(self, image, mask):
        if not self._same_exact_shape(image, mask):
            return False, (
                f"shape mismatch: image={tuple(np.asarray(image.array).shape)} "
                f"mask={tuple(np.asarray(mask.array).shape)}"
            )

        image_spacing = getattr(image, "spacing", None)
        mask_spacing = getattr(mask, "spacing", None)
        if not self._same_close_tuple(image_spacing, mask_spacing):
            return False, f"spacing mismatch: image={image_spacing} mask={mask_spacing}"

        image_origin = getattr(image, "origin", None)
        mask_origin = getattr(mask, "origin", None)
        if not self._same_close_tuple(image_origin, mask_origin):
            return False, f"origin mismatch: image={image_origin} mask={mask_origin}"

        image_direction = getattr(image, "direction", None)
        mask_direction = getattr(mask, "direction", None)
        if not self._same_close_tuple(image_direction, mask_direction):
            return False, f"direction mismatch: image={image_direction} mask={mask_direction}"

        return True, ""

    def _load_masks_for_volume(self, image, masks):
        loaded_masks = []

        for i, mask_instance in enumerate(masks):
            mask_name, mask = next(iter(mask_instance.items()))

            is_compatible, reason = self._metadata_matches_image(image, mask)
            if not is_compatible:
                warnings.warn(
                    f"Skipping mask '{mask_name}' for image '{self.current_case_name}': {reason}",
                    RuntimeWarning,
                )
                continue

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
        return self._rot180(self.volume[:, :, x])

    def _get_coronal_slice(self, y):
        return self._rot180(self.volume[:, y, :])

    def _get_axial_slice(self, z):
        return self._rot180(self.volume[z, :, :])

    def _get_sagittal_mask(self, mask, x):
        return self._rot180(mask[:, :, x])

    def _get_coronal_mask(self, mask, y):
        return self._rot180(mask[:, y, :])

    def _get_axial_mask(self, mask, z):
        return self._rot180(mask[z, :, :])

    def _get_view_dimensions(self, view):
        if view is self.sag_view:
            return self.ny * self.sy, self.nz * self.sz
        if view is self.cor_view:
            return self.nx * self.sx, self.nz * self.sz
        if view is self.axi_view:
            return self.nx * self.sx, self.ny * self.sy
        raise ValueError("Unknown view")

    def _fit_full_view_with_aspect(self, view):
        width_phys, height_phys = self._get_view_dimensions(view)

        rect = view.vb.sceneBoundingRect()
        if rect.width() <= 1 or rect.height() <= 1:
            view.vb.setRange(
                xRange=(0.0, width_phys),
                yRange=(0.0, height_phys),
                padding=0.0,
                disableAutoRange=True,
            )
            return

        widget_ratio = rect.width() / rect.height()
        data_ratio = width_phys / height_phys

        if widget_ratio >= data_ratio:
            visible_h = height_phys
            visible_w = height_phys * widget_ratio
        else:
            visible_w = width_phys
            visible_h = width_phys / widget_ratio

        cx = width_phys / 2.0
        cy = height_phys / 2.0

        x0 = cx - visible_w / 2.0
        x1 = cx + visible_w / 2.0
        y0 = cy - visible_h / 2.0
        y1 = cy + visible_h / 2.0

        view.vb.setLimits(
            xMin=min(0.0, x0),
            xMax=max(width_phys, x1),
            yMin=min(0.0, y0),
            yMax=max(height_phys, y1),
            minXRange=max(width_phys * 0.01, 1e-6),
            minYRange=max(height_phys * 0.01, 1e-6),
            maxXRange=max(visible_w, width_phys),
            maxYRange=max(visible_h, height_phys),
        )

        view.vb.setRange(
            xRange=(x0, x1),
            yRange=(y0, y1),
            padding=0.0,
            disableAutoRange=True,
        )

    def _reset_all_view_ranges(self):
        self._fit_full_view_with_aspect(self.sag_view)
        self._fit_full_view_with_aspect(self.cor_view)
        self._fit_full_view_with_aspect(self.axi_view)

    def _reset_single_view(self, view):
        self._fit_full_view_with_aspect(view)

    def _clamp_range(self, start, end, lower, upper):
        size = end - start
        total = upper - lower

        if size >= total:
            return lower, upper

        if start < lower:
            end += (lower - start)
            start = lower
        if end > upper:
            start -= (end - upper)
            end = upper

        start = max(lower, start)
        end = min(upper, end)
        return start, end

    def _zoom_view(self, view, zoom_factor, center_x, center_y):
        width_phys, height_phys = self._get_view_dimensions(view)

        (x0, x1), (y0, y1) = view.vb.viewRange()
        cur_w = max(x1 - x0, 1e-12)
        cur_h = max(y1 - y0, 1e-12)

        min_w = max(width_phys * 0.01, 1e-6)
        min_h = max(height_phys * 0.01, 1e-6)

        new_w = max(min_w, cur_w * zoom_factor)
        new_h = max(min_h, cur_h * zoom_factor)

        rel_x = (center_x - x0) / cur_w
        rel_y = (center_y - y0) / cur_h
        rel_x = float(np.clip(rel_x, 0.0, 1.0))
        rel_y = float(np.clip(rel_y, 0.0, 1.0))

        new_x0 = center_x - rel_x * new_w
        new_x1 = new_x0 + new_w
        new_y0 = center_y - rel_y * new_h
        new_y1 = new_y0 + new_h

        view.vb.setRange(
            xRange=(new_x0, new_x1),
            yRange=(new_y0, new_y1),
            padding=0.0,
            disableAutoRange=True,
        )

    def _crosshair_pos(self, reversed_index, spacing):
        return (reversed_index + 0.5) * spacing

    def _update_crosshairs(self):
        self.sag_vline.setPos(self._crosshair_pos(self.ny - 1 - self.current_coronal, self.sy))
        self.sag_hline.setPos(self._crosshair_pos(self.nz - 1 - self.current_axial, self.sz))

        self.cor_vline.setPos(self._crosshair_pos(self.nx - 1 - self.current_sagittal, self.sx))
        self.cor_hline.setPos(self._crosshair_pos(self.nz - 1 - self.current_axial, self.sz))

        self.axi_vline.setPos(self._crosshair_pos(self.nx - 1 - self.current_sagittal, self.sx))
        self.axi_hline.setPos(self._crosshair_pos(self.ny - 1 - self.current_coronal, self.sy))

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
            f"Intensity: {voxel_str} | "
            f"Wheel=slices, Ctrl+Wheel=zoom, MiddleDrag=pan, DoubleClick=reset, R=reset all"
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

    def _on_view_zoomed(self, view, zoom_factor, x_phys, y_phys):
        self._zoom_view(view, zoom_factor, x_phys, y_phys)

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

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_R:
            self._reset_all_view_ranges()
            ev.accept()
            return
        super().keyPressEvent(ev)