import os
import sys
import warnings
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore, QtGui


class RangeSlider(QtWidgets.QWidget):
    valuesChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self._minimum = 0
        self._maximum = 100
        self._lower_value = 25
        self._upper_value = 75
        self._handle_radius = 8
        self._groove_thickness = 6
        self._active_handle = None
        self._margin = self._handle_radius + 2
        self.setMinimumHeight(28)
        self.setMouseTracking(True)

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def lowerValue(self):
        return self._lower_value

    def upperValue(self):
        return self._upper_value

    def setRange(self, minimum, maximum):
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum <= minimum:
            maximum = minimum + 1
        self._minimum = minimum
        self._maximum = maximum
        self._lower_value = max(self._minimum, min(self._lower_value, self._maximum))
        self._upper_value = max(self._minimum, min(self._upper_value, self._maximum))
        if self._lower_value > self._upper_value:
            self._lower_value = self._upper_value
        self.update()

    def setValues(self, lower, upper):
        lower = int(round(lower))
        upper = int(round(upper))
        lower = max(self._minimum, min(lower, self._maximum))
        upper = max(self._minimum, min(upper, self._maximum))
        if lower > upper:
            lower, upper = upper, lower

        changed = (lower != self._lower_value) or (upper != self._upper_value)
        self._lower_value = lower
        self._upper_value = upper
        self.update()
        if changed:
            self.valuesChanged.emit(self._lower_value, self._upper_value)

    def sizeHint(self):
        return QtCore.QSize(260, 28)

    def _usable_length(self):
        if self.orientation == QtCore.Qt.Horizontal:
            return max(1, self.width() - 2 * self._margin)
        return max(1, self.height() - 2 * self._margin)

    def _value_to_pos(self, value):
        ratio = (value - self._minimum) / max(1, (self._maximum - self._minimum))
        ratio = float(np.clip(ratio, 0.0, 1.0))

        if self.orientation == QtCore.Qt.Horizontal:
            return self._margin + ratio * self._usable_length()
        return self.height() - (self._margin + ratio * self._usable_length())

    def _pos_to_value(self, pos):
        if self.orientation == QtCore.Qt.Horizontal:
            ratio = (pos - self._margin) / self._usable_length()
        else:
            ratio = (self.height() - pos - self._margin) / self._usable_length()

        ratio = float(np.clip(ratio, 0.0, 1.0))
        value = self._minimum + ratio * (self._maximum - self._minimum)
        return int(round(value))

    def _lower_handle_center(self):
        if self.orientation == QtCore.Qt.Horizontal:
            return QtCore.QPointF(self._value_to_pos(self._lower_value), self.height() / 2.0)
        return QtCore.QPointF(self.width() / 2.0, self._value_to_pos(self._lower_value))

    def _upper_handle_center(self):
        if self.orientation == QtCore.Qt.Horizontal:
            return QtCore.QPointF(self._value_to_pos(self._upper_value), self.height() / 2.0)
        return QtCore.QPointF(self.width() / 2.0, self._value_to_pos(self._upper_value))

    def _handle_hit(self, pos, center):
        return (pos - center).manhattanLength() <= (self._handle_radius + 4)

    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        groove_pen = QtGui.QPen(QtGui.QColor(120, 120, 120))
        groove_pen.setWidth(1)
        painter.setPen(groove_pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(70, 70, 70)))

        if self.orientation == QtCore.Qt.Horizontal:
            groove_rect = QtCore.QRectF(
                self._margin,
                (self.height() - self._groove_thickness) / 2.0,
                self.width() - 2 * self._margin,
                self._groove_thickness,
            )
        else:
            groove_rect = QtCore.QRectF(
                (self.width() - self._groove_thickness) / 2.0,
                self._margin,
                self._groove_thickness,
                self.height() - 2 * self._margin,
            )

        painter.drawRoundedRect(groove_rect, 3, 3)

        lower_pos = self._value_to_pos(self._lower_value)
        upper_pos = self._value_to_pos(self._upper_value)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 165, 0)))

        if self.orientation == QtCore.Qt.Horizontal:
            selected_rect = QtCore.QRectF(
                lower_pos,
                groove_rect.top(),
                max(1.0, upper_pos - lower_pos),
                groove_rect.height(),
            )
        else:
            selected_rect = QtCore.QRectF(
                groove_rect.left(),
                upper_pos,
                groove_rect.width(),
                max(1.0, lower_pos - upper_pos),
            )

        painter.drawRoundedRect(selected_rect, 3, 3)

        for center, active in (
            (self._lower_handle_center(), self._active_handle == "lower"),
            (self._upper_handle_center(), self._active_handle == "upper"),
        ):
            painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 1))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 190, 80) if active else QtGui.QColor(245, 245, 245)))
            painter.drawEllipse(center, self._handle_radius, self._handle_radius)

    def mousePressEvent(self, ev):
        pos = ev.pos()
        lower_center = self._lower_handle_center()
        upper_center = self._upper_handle_center()

        lower_hit = self._handle_hit(pos, lower_center)
        upper_hit = self._handle_hit(pos, upper_center)

        if lower_hit and upper_hit:
            dist_lower = (pos - lower_center).manhattanLength()
            dist_upper = (pos - upper_center).manhattanLength()
            self._active_handle = "lower" if dist_lower <= dist_upper else "upper"
        elif lower_hit:
            self._active_handle = "lower"
        elif upper_hit:
            self._active_handle = "upper"
        else:
            click_value = self._pos_to_value(pos.x() if self.orientation == QtCore.Qt.Horizontal else pos.y())
            if abs(click_value - self._lower_value) <= abs(click_value - self._upper_value):
                self._active_handle = "lower"
            else:
                self._active_handle = "upper"
            self._move_active_handle(click_value)

        self.update()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._active_handle is None:
            ev.ignore()
            return

        pos_value = self._pos_to_value(ev.pos().x() if self.orientation == QtCore.Qt.Horizontal else ev.pos().y())
        self._move_active_handle(pos_value)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._active_handle = None
        self.update()
        ev.accept()

    def _move_active_handle(self, value):
        if self._active_handle == "lower":
            self.setValues(min(value, self._upper_value), self._upper_value)
        elif self._active_handle == "upper":
            self.setValues(self._lower_value, max(value, self._lower_value))


class SliceView(pg.GraphicsLayoutWidget):
    sigScrolled = QtCore.pyqtSignal(object, int)
    sigClicked = QtCore.pyqtSignal(object, float, float)
    sigZoomed = QtCore.pyqtSignal(object, float, float, float)
    sigToggleMaximize = QtCore.pyqtSignal(object)
    sigResized = QtCore.pyqtSignal(object)

    def __init__(self, title="", parent=None):
        super().__init__(parent=parent)

        self.setBackground("k")
        self.plot = self.addPlot(title=title)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()

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
            self.sigToggleMaximize.emit(self)
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

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.sigResized.emit(self)


class MaskLegendButton(QtWidgets.QToolButton):
    toggledWithIndex = QtCore.pyqtSignal(int, bool)

    def __init__(self, index, name, rgb, checked=True, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = name
        self.rgb = rgb
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.setAutoRaise(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(32)
        self.setToolTip(name)
        self.setStyleSheet(
            """
            QToolButton {
                color: white;
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 5px 8px;
                text-align: left;
            }
            QToolButton:hover {
                background-color: #303030;
            }
            QToolButton:checked {
                background-color: #3a3a3a;
                border: 1px solid #888;
            }
            """
        )
        self._update_icon()
        self.toggled.connect(self._emit_toggled)

    def _create_icon_pixmap(self, enabled):
        size = 16
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        fill = QtGui.QColor(self.rgb[0], self.rgb[1], self.rgb[2], 255 if enabled else 70)
        border = QtGui.QColor(220, 220, 220) if enabled else QtGui.QColor(120, 120, 120)

        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(QtGui.QBrush(fill))
        painter.drawRoundedRect(QtCore.QRectF(1, 1, size - 2, size - 2), 3, 3)

        if not enabled:
            painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180), 2))
            painter.drawLine(3, 3, size - 3, size - 3)

        painter.end()
        return pixmap

    def _elided_text(self, text):
        fm = self.fontMetrics()
        available_width = max(70, self.width() - 44)
        return fm.elidedText(text, QtCore.Qt.ElideRight, available_width)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_icon()

    def _update_icon(self):
        self.setIcon(QtGui.QIcon(self._create_icon_pixmap(self.isChecked())))
        self.setText(self._elided_text(self.name))

    def _emit_toggled(self, checked):
        self._update_icon()
        self.toggledWithIndex.emit(self.index, checked)

    def setChecked(self, checked):
        super().setChecked(checked)
        self._update_icon()


class ColorLegendWidget(QtWidgets.QFrame):
    sigMaskToggled = QtCore.pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #222;
                border: 1px solid #444;
                border-radius: 6px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                border: none;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #666;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #888;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
                border: none;
            }
            QPushButton {
                background-color: orange;
                color: black;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffb347;
            }
            QPushButton:pressed {
                background-color: #e69500;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #222;
            }
            """
        )

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QtWidgets.QLabel("Masks")
        self.hide_all_button = QtWidgets.QPushButton("Hide All Masks")
        self.hide_all_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.hide_all_button.setMinimumHeight(30)
        self.hide_all_button.clicked.connect(self.toggle_all_masks)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.hide_all_button)

        outer_layout.addLayout(header_layout)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.scroll_content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        outer_layout.addWidget(self.scroll_area, 1)

        self._buttons = []
        self.hide_all_button.setEnabled(False)
        self._all_hidden = False

    def set_items(self, items):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._buttons = []

        for i, item in enumerate(items):
            button = MaskLegendButton(
                index=i,
                name=item["name"],
                rgb=item["color"],
                checked=item.get("visible", True),
            )
            button.toggledWithIndex.connect(self._on_button_toggled)
            self.content_layout.addWidget(button)
            self._buttons.append(button)

        self.content_layout.addStretch(1)
        self.title_label.setText(f"Masks ({len(items)})")
        self.hide_all_button.setEnabled(len(items) > 0)
        self._update_toggle_button_text()

    def _on_button_toggled(self, index, checked):
        self.sigMaskToggled.emit(index, checked)
        self._update_toggle_button_text()

    def set_item_checked(self, index, checked):
        if 0 <= index < len(self._buttons):
            button = self._buttons[index]
            blocker = QtCore.QSignalBlocker(button)
            button.setChecked(checked)
            del blocker
        self._update_toggle_button_text()

    def _update_toggle_button_text(self):
        any_visible = any(button.isChecked() for button in self._buttons)
        self._all_hidden = (len(self._buttons) > 0 and not any_visible)

        if self._all_hidden:
            self.hide_all_button.setText("Show All Masks")
        else:
            self.hide_all_button.setText("Hide All Masks")

    def toggle_all_masks(self):
        if not self._buttons:
            return

        target_checked = self._all_hidden

        for i, button in enumerate(self._buttons):
            if button.isChecked() != target_checked:
                button.setChecked(target_checked)
                self.sigMaskToggled.emit(i, target_checked)

        self._update_toggle_button_text()


class Visualization(QtWidgets.QMainWindow):
    def __init__(self, image_sets, case_identifiers=None, case_loader=None):
        super().__init__()
        self.setWindowTitle("Z-Rad Viewer")
        self.resize(1500, 900)

        if not image_sets:
            raise ValueError("image_sets must contain at least one image.")

        self.image_sets = image_sets
        self.case_identifiers = case_identifiers or [item.get("image_name", str(i)) for i, item in enumerate(image_sets)]
        self.case_loader = case_loader
        self.total_cases = len(self.case_identifiers)
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
        self.current_case_name = ""
        self.current_imaging_modality = None
        self.nx = self.ny = self.nz = 0
        self.sx = self.sy = self.sz = 1.0

        self.data_vmin = 0.0
        self.data_vmax = 1.0
        self.window_min = 0.0
        self.window_max = 1.0

        self.window_slider_scale = 1000
        self._window_slider_updating = False

        self.current_sagittal = 0
        self.current_coronal = 0
        self.current_axial = 0

        self._suspend_resize_refit = False
        self._maximized_view = None
        self._normal_stretch = [1, 1, 1]

        self._build_ui()
        self._connect_events()
        self._init_views()

        self._load_image_set(0)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.main_layout = QtWidgets.QHBoxLayout(central)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(12)

        self.legend_widget = ColorLegendWidget()
        self.main_layout.addWidget(self.legend_widget, 0)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.views_layout = QtWidgets.QHBoxLayout()
        self.views_layout.setSpacing(10)

        self.sag_view = SliceView("Sagittal")
        self.cor_view = SliceView("Coronal")
        self.axi_view = SliceView("Axial")

        self.views_layout.addWidget(self.sag_view, 1)
        self.views_layout.addWidget(self.cor_view, 1)
        self.views_layout.addWidget(self.axi_view, 1)

        right_layout.addLayout(self.views_layout, 1)

        window_group = QtWidgets.QGroupBox("Windowing")
        window_group.setStyleSheet(
            """
            QGroupBox {
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                margin-top: 8px;
                background-color: #222;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
            QLabel {
                color: white;
            }
            """
        )
        window_layout = QtWidgets.QHBoxLayout(window_group)
        window_layout.setContentsMargins(12, 12, 12, 10)
        window_layout.setSpacing(10)

        self.window_min_label_title = QtWidgets.QLabel("Min:")
        self.window_min_label = QtWidgets.QLabel("0.000")
        self.window_slider = RangeSlider(QtCore.Qt.Horizontal)
        self.window_max_label_title = QtWidgets.QLabel("Max:")
        self.window_max_label = QtWidgets.QLabel("1.000")

        window_layout.addWidget(self.window_min_label_title)
        window_layout.addWidget(self.window_min_label)
        window_layout.addWidget(self.window_slider, 1)
        window_layout.addWidget(self.window_max_label_title)
        window_layout.addWidget(self.window_max_label)

        right_layout.addWidget(window_group, 0)

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

        right_layout.addLayout(button_layout, 0)

        self.main_layout.addWidget(right_panel, 1)

        self.statusBar().showMessage("Ready")
        self.setStyleSheet("QMainWindow { background-color: #111; }")

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

        self.sag_view.sigToggleMaximize.connect(self._toggle_maximized_view)
        self.cor_view.sigToggleMaximize.connect(self._toggle_maximized_view)
        self.axi_view.sigToggleMaximize.connect(self._toggle_maximized_view)

        self.sag_view.sigResized.connect(self._on_slice_view_resized)
        self.cor_view.sigResized.connect(self._on_slice_view_resized)
        self.axi_view.sigResized.connect(self._on_slice_view_resized)

        self.prev_button.clicked.connect(self._show_previous_image)
        self.next_button.clicked.connect(self._show_next_image)
        self.close_button.clicked.connect(self.close)

        self.window_slider.valuesChanged.connect(self._on_window_slider_changed)
        self.legend_widget.sigMaskToggled.connect(self._on_mask_toggled)

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

    def _set_item_rect(self, item, width_phys, height_phys):
        item.setRect(QtCore.QRectF(0.0, 0.0, float(width_phys), float(height_phys)))

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

    def _toggle_maximized_view(self, view):
        all_views = [self.sag_view, self.cor_view, self.axi_view]

        if self._maximized_view is view:
            for i, v in enumerate(all_views):
                v.setVisible(True)
                self.views_layout.setStretch(i, self._normal_stretch[i])
            self._maximized_view = None
            QtCore.QTimer.singleShot(0, self._reset_all_view_ranges)
            return

        self._maximized_view = view
        for i, v in enumerate(all_views):
            is_target = (v is view)
            v.setVisible(is_target)
            self.views_layout.setStretch(i, 1 if is_target else 0)

        QtCore.QTimer.singleShot(0, self._reset_single_visible_view)

    def _reset_single_visible_view(self):
        if self._maximized_view is not None:
            self._fit_full_view_with_aspect(self._maximized_view)

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

        self._apply_item_rects()
        self._apply_mask_visibility()

    def _apply_item_rects(self):
        sag_w, sag_h = self._get_view_dimensions(self.sag_view)
        cor_w, cor_h = self._get_view_dimensions(self.cor_view)
        axi_w, axi_h = self._get_view_dimensions(self.axi_view)

        self._set_item_rect(self.sag_img, sag_w, sag_h)
        self._set_item_rect(self.cor_img, cor_w, cor_h)
        self._set_item_rect(self.axi_img, axi_w, axi_h)

        for item in self.sag_mask_items:
            self._set_item_rect(item, sag_w, sag_h)
        for item in self.cor_mask_items:
            self._set_item_rect(item, cor_w, cor_h)
        for item in self.axi_mask_items:
            self._set_item_rect(item, axi_w, axi_h)

    def _apply_mask_visibility(self):
        for i, item in enumerate(self.sag_mask_items):
            item.setVisible(self.masks[i]["visible"])
        for i, item in enumerate(self.cor_mask_items):
            item.setVisible(self.masks[i]["visible"])
        for i, item in enumerate(self.axi_mask_items):
            item.setVisible(self.masks[i]["visible"])

        self.legend_widget.setVisible(len(self.masks) > 0)

    def _load_image_set(self, index):
        if index < 0 or index >= self.total_cases:
            return

        if self.case_loader is not None:
            if index < len(self.image_sets) and self.image_sets[index] is not None:
                item = self.image_sets[index]
            else:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    case_identifier = self.case_identifiers[index]
                    item = self.case_loader(case_identifier)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()

                if item is None:
                    self.statusBar().showMessage(
                        f"Unable to load patient '{self.case_identifiers[index]}'.",
                        5000,
                    )
                    return

                if index >= len(self.image_sets):
                    self.image_sets.extend([None] * (index - len(self.image_sets) + 1))
                self.image_sets[index] = item
        else:
            item = self.image_sets[index]

        image = item["image"]
        masks = item.get("masks", [])

        self.volume = np.asarray(image.array)
        self.current_case_name = item["image_name"]
        self.current_imaging_modality = str(item.get("imaging_modality", "")).strip().upper()

        self.nz, self.ny, self.nx = self.volume.shape
        self.sx, self.sy, self.sz = image.spacing

        finite_vals = self.volume[np.isfinite(self.volume)]

        if finite_vals.size == 0:
            base_vmin = 0.0
            finite_max = 1.0
            pet_default_high = 3.0
        else:
            base_vmin = float(np.min(finite_vals))
            finite_max = float(np.max(finite_vals))

            pet_default_high = float(np.percentile(finite_vals, 99.5))
            if not np.isfinite(pet_default_high):
                pet_default_high = finite_max

            if finite_max <= base_vmin:
                finite_max = base_vmin + 1.0

        if self.current_imaging_modality == "CT":
            self.window_min = -160.0
            self.window_max = 240.0
            self.data_vmin = min(base_vmin, self.window_min)
            self.data_vmax = max(finite_max, self.window_max)

        elif self.current_imaging_modality == "PET":
            self.window_min = 0.0
            self.window_max = max(4.0, pet_default_high)
            if self.window_max <= self.window_min:
                self.window_max = self.window_min + 1.0

            self.data_vmin = min(0.0, base_vmin)
            self.data_vmax = max(finite_max, self.window_max)
            if self.data_vmax <= self.data_vmin:
                self.data_vmax = self.data_vmin + 1.0

        else:
            p1 = float(np.percentile(finite_vals, 1)) if finite_vals.size else 0.0
            p99 = float(np.percentile(finite_vals, 99)) if finite_vals.size else 1.0

            if not np.isfinite(p1):
                p1 = base_vmin
            if not np.isfinite(p99) or p99 <= p1:
                p99 = finite_max

            self.data_vmin = base_vmin
            self.data_vmax = finite_max
            self.window_min = p1
            self.window_max = p99

            if self.data_vmax <= self.data_vmin:
                self.data_vmax = self.data_vmin + 1.0
            if self.window_max <= self.window_min:
                self.window_max = self.window_min + 1.0

        self._update_window_slider_from_values()

        self.masks = self._load_masks_for_volume(image, masks)

        self.current_image_index = index
        self.current_sagittal = self.nx // 2
        self.current_coronal = self.ny // 2
        self.current_axial = self.nz // 2

        self._rebuild_mask_items()
        self.legend_widget.set_items(self.masks)
        self._apply_mask_visibility()

        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < self.total_cases - 1)

        if self.case_loader is not None:
            for i in range(len(self.image_sets)):
                if i != self.current_image_index:
                    self.image_sets[i] = None

        self._update_all_views()
        QtCore.QTimer.singleShot(0, self._reset_all_view_ranges)
        QtCore.QTimer.singleShot(50, self._reset_all_view_ranges)

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
                    "visible": True,
                }
            )

        return loaded_masks

    def _clip_index(self, value, upper):
        return max(0, min(int(round(value)), upper))

    def _normalize_to_u8(self, arr):
        arr = np.nan_to_num(arr, nan=self.window_min, posinf=self.window_max, neginf=self.window_min)
        arr = np.clip(arr, self.window_min, self.window_max)
        denom = max(self.window_max - self.window_min, 1e-12)
        arr = ((arr - self.window_min) / denom * 255.0).astype(np.uint8)
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

    def _flipud(self, arr):
        return np.ascontiguousarray(arr[::-1, :])

    def _get_sagittal_slice(self, x):
        return self._flipud(self.volume[:, :, x])

    def _get_coronal_slice(self, y):
        return self._rot180(self.volume[:, y, :])

    def _get_axial_slice(self, z):
        return self.volume[z, :, ::-1]

    def _get_sagittal_mask(self, mask, x):
        return self._flipud(mask[:, :, x])

    def _get_coronal_mask(self, mask, y):
        return self._rot180(mask[:, y, :])

    def _get_axial_mask(self, mask, z):
        return mask[z, :, ::-1]

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

        geom = view.vb.screenGeometry()
        if geom.width() <= 1 or geom.height() <= 1:
            view.vb.setRange(
                xRange=(0.0, width_phys),
                yRange=(0.0, height_phys),
                padding=0.0,
                disableAutoRange=True,
            )
            return

        widget_ratio = float(geom.width()) / float(geom.height())
        data_ratio = float(width_phys) / max(float(height_phys), 1e-12)

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

        pad_w = max(width_phys * 0.01, 1e-6)
        pad_h = max(height_phys * 0.01, 1e-6)

        self._suspend_resize_refit = True
        try:
            view.vb.setLimits(
                xMin=x0 - pad_w,
                xMax=x1 + pad_w,
                yMin=y0 - pad_h,
                yMax=y1 + pad_h,
                minXRange=max(width_phys * 0.01, 1e-6),
                minYRange=max(height_phys * 0.01, 1e-6),
                maxXRange=max(visible_w + 2 * pad_w, width_phys + 2 * pad_w),
                maxYRange=max(visible_h + 2 * pad_h, height_phys + 2 * pad_h),
            )

            view.vb.setRange(
                xRange=(x0, x1),
                yRange=(y0, y1),
                padding=0.0,
                disableAutoRange=True,
            )
        finally:
            self._suspend_resize_refit = False

    def _reset_all_view_ranges(self):
        if self._maximized_view is not None:
            self._fit_full_view_with_aspect(self._maximized_view)
            return

        self._fit_full_view_with_aspect(self.sag_view)
        self._fit_full_view_with_aspect(self.cor_view)
        self._fit_full_view_with_aspect(self.axi_view)

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

    def _crosshair_pos(self, index_value, spacing):
        return (index_value + 0.5) * spacing

    def _update_crosshairs(self):
        self.sag_vline.setPos(self._crosshair_pos(self.current_coronal, self.sy))
        self.sag_hline.setPos(self._crosshair_pos(self.nz - 1 - self.current_axial, self.sz))

        self.cor_vline.setPos(self._crosshair_pos(self.nx - 1 - self.current_sagittal, self.sx))
        self.cor_hline.setPos(self._crosshair_pos(self.nz - 1 - self.current_axial, self.sz))

        self.axi_vline.setPos(self._crosshair_pos(self.nx - 1 - self.current_sagittal, self.sx))
        self.axi_hline.setPos(self._crosshair_pos(self.current_coronal, self.sy))

    def _update_titles(self):
        prefix = f"{self.current_case_name if len(self.current_case_name) <= 25 else self.current_case_name[:25] + '...'}"
        self.sag_view.plot.setTitle(f"{prefix} | {self.current_sagittal}", color="w")
        self.cor_view.plot.setTitle(f"{prefix} | {self.current_coronal}", color="w")
        self.axi_view.plot.setTitle(f"{prefix} | {self.current_axial}", color="w")

    def _get_current_voxel_value(self):
        value = self.volume[self.current_axial, self.current_coronal, self.current_sagittal]
        return float(value) if np.isfinite(value) else np.nan

    def _build_status_text(self):
        voxel_value = self._get_current_voxel_value()
        voxel_str = "nan" if np.isnan(voxel_value) else f"{voxel_value:.3f}"
        return (
            f"Image {self.current_image_index + 1}/{self.total_cases} | "
            f"Case: {self.current_case_name if len(self.current_case_name) <= 20 else self.current_case_name[:20] + '...'} | "
            f"Shape: {self.volume.shape} | "
            f"Spacing: ({self.sx:.3f}, {self.sy:.3f}, {self.sz:.3f}) mm | "
            f"Voxel: ({self.current_sagittal}, {self.current_coronal}, {self.current_axial}) | "
            f"Intensity: {voxel_str}"
        )

    def _window_value_to_slider(self, value):
        denom = max(self.data_vmax - self.data_vmin, 1e-12)
        ratio = (value - self.data_vmin) / denom
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(ratio * self.window_slider_scale))

    def _slider_to_window_value(self, slider_value):
        ratio = slider_value / float(self.window_slider_scale)
        return self.data_vmin + ratio * (self.data_vmax - self.data_vmin)

    def _update_window_slider_from_values(self):
        self._window_slider_updating = True
        self.window_slider.setRange(0, self.window_slider_scale)
        self.window_slider.setValues(
            self._window_value_to_slider(self.window_min),
            self._window_value_to_slider(self.window_max),
        )
        self._window_slider_updating = False
        self._update_window_labels()

    def _update_window_labels(self):
        self.window_min_label.setText(f"{self.window_min:.3f}")
        self.window_max_label.setText(f"{self.window_max:.3f}")

    def _on_window_slider_changed(self, lower, upper):
        if self._window_slider_updating:
            return

        new_min = self._slider_to_window_value(lower)
        new_max = self._slider_to_window_value(upper)

        if new_max <= new_min:
            new_max = new_min + 1e-6

        self.window_min = float(new_min)
        self.window_max = float(new_max)
        self._update_window_labels()
        self._update_all_views()

    def _on_mask_toggled(self, index, checked):
        if 0 <= index < len(self.masks):
            self.masks[index]["visible"] = bool(checked)
            self._apply_mask_visibility()
            self.statusBar().showMessage(self._build_status_text())

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

        self._apply_item_rects()
        self._apply_mask_visibility()
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
            self.current_coronal = self._clip_index(x_phys / self.sy, self.ny - 1)
            self.current_axial = self._clip_index((self.nz - 1) - (y_phys / self.sz), self.nz - 1)

        elif view is self.cor_view:
            self.current_sagittal = self._clip_index((self.nx - 1) - (x_phys / self.sx), self.nx - 1)
            self.current_axial = self._clip_index((self.nz - 1) - (y_phys / self.sz), self.nz - 1)

        elif view is self.axi_view:
            self.current_sagittal = self._clip_index((self.nx - 1) - (x_phys / self.sx), self.nx - 1)
            self.current_coronal = self._clip_index(y_phys / self.sy, self.ny - 1)

        self._update_all_views()

    def _show_previous_image(self):
        if self.current_image_index > 0:
            self._load_image_set(self.current_image_index - 1)

    def _show_next_image(self):
        if self.current_image_index < self.total_cases - 1:
            self._load_image_set(self.current_image_index + 1)

    def _on_slice_view_resized(self, view):
        if self._suspend_resize_refit or self.volume is None:
            return
        if self._maximized_view is not None and view is not self._maximized_view:
            return
        QtCore.QTimer.singleShot(0, lambda v=view: self._fit_full_view_with_aspect(v))

    def showEvent(self, ev):
        super().showEvent(ev)
        QtCore.QTimer.singleShot(0, self._reset_all_view_ranges)
        QtCore.QTimer.singleShot(50, self._reset_all_view_ranges)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_R:
            self._reset_all_view_ranges()
            ev.accept()
            return
        super().keyPressEvent(ev)
