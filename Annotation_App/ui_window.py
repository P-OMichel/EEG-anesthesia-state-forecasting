import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QCheckBox,
    QMessageBox, QSplitter, QListWidget, QGraphicsRectItem
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPen, QBrush, QColor
import pyqtgraph as pg

from Annotation_App.processing import EEGProcessor
from Annotation_App.annotations import AnnotationStore, CONDITIONS

pg.setConfigOption("background", "#1e1e1e")
pg.setConfigOption("foreground", "#d4d4d4")
pg.setConfigOptions(antialias=True)


# =====================================================================
# CUSTOM INTERACTIVE PLOT (SHIFT-BOX SELECT + CTRL+SHIFT DESELECT + DRAG)
# =====================================================================
class InteractiveAnnotationPlot(pg.PlotWidget):
    """
    Supports:
    1. Shift + Left-Drag: Rubber-band box selection (selects all points inside box).
    2. Ctrl + Shift + Left-Drag: Rubber-band box deselection (deselects all points inside box).
    3. Left-Drag on a point: Vertical drag-and-drop modification (snapped to integers).
       All selected points snap to the exact same target level as the dragged point.
    4. Normal Left-Drag on background: Standard PyQtGraph pan/zoom.
    """

    valuesChanged = pyqtSignal(list, list)

    def __init__(self, y_min: int, y_max: int, line_color: str, default_brush: str, title: str):
        super().__init__(title=title)
        self.y_min = y_min
        self.y_max = y_max
        self.default_brush_color = default_brush
        self.selected_brush_color = "#ffeb3b"

        self.x_data = np.array([], dtype=np.float32)
        self.y_data = np.array([], dtype=np.int32)
        self.selected_indices = set()

        self.line_item = self.plot(pen=pg.mkPen(line_color, width=2))
        self.scatter_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen("#ffffff", width=1))
        self.addItem(self.scatter_item)

        self._mode = None  # None | 'drag_points' | 'box_select' | 'box_deselect'
        self._drag_anchor_idx = None
        self._drag_anchor_init_y = 0
        self._drag_start_y = 0.0

        self._box_start_view = None
        self._selection_rect = QGraphicsRectItem()
        self._selection_rect.setZValue(100)
        self._selection_rect.hide()
        self.addItem(self._selection_rect, ignoreBounds=True)

    def _style_box_rect(self, deselect: bool):
        color = QColor("#ff6b6b") if deselect else QColor("#ffeb3b")
        fill = QColor(255, 107, 107, 35) if deselect else QColor(255, 235, 59, 35)
        box_pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
        box_pen.setCosmetic(True)
        self._selection_rect.setPen(box_pen)
        self._selection_rect.setBrush(QBrush(fill))

    def set_series(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data.copy()
        self.y_data = y_data.copy()
        self.selected_indices.clear()
        self._redraw()

    def clear_selection(self):
        self.selected_indices.clear()
        self._redraw()

    def _redraw(self):
        self.line_item.setData(self.x_data, self.y_data)
        brushes = [
            pg.mkBrush(self.selected_brush_color if i in self.selected_indices else self.default_brush_color)
            for i in range(len(self.x_data))
        ]
        sizes = [13 if i in self.selected_indices else 9 for i in range(len(self.x_data))]
        self.scatter_item.setData(self.x_data, self.y_data, brush=brushes, size=sizes)

    def _find_point_near_pixel(self, pos_widget, radius_px: float = 14.0):
        if len(self.x_data) == 0:
            return None

        vb = self.getViewBox()
        mouse_scene = self.mapToScene(pos_widget)
        if not vb.sceneBoundingRect().contains(mouse_scene):
            return None

        best_idx = None
        best_dist = radius_px

        for i in range(len(self.x_data)):
            pt_scene = vb.mapViewToScene(pg.Point(self.x_data[i], self.y_data[i]))
            dist = (pt_scene - mouse_scene).manhattanLength()
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and len(self.x_data) > 0:
            vb = self.getViewBox()
            view_pos = vb.mapSceneToView(self.mapToScene(ev.pos()))
            mods = ev.modifiers()

            has_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            has_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

            if has_shift and has_ctrl:
                self._mode = "box_deselect"
                self._style_box_rect(deselect=True)
                self._box_start_view = view_pos
                self._selection_rect.setRect(QRectF(view_pos.x(), view_pos.y(), 0, 0))
                self._selection_rect.show()
                ev.accept()
                return

            if has_shift:
                self._mode = "box_select"
                self._style_box_rect(deselect=False)
                self._box_start_view = view_pos
                self._selection_rect.setRect(QRectF(view_pos.x(), view_pos.y(), 0, 0))
                self._selection_rect.show()
                ev.accept()
                return

            hit_idx = self._find_point_near_pixel(ev.pos())
            if hit_idx is not None:
                self._mode = "drag_points"
                self._drag_anchor_idx = hit_idx
                self._drag_anchor_init_y = int(self.y_data[hit_idx])
                self._drag_start_y = view_pos.y()

                if hit_idx not in self.selected_indices:
                    self.selected_indices = {hit_idx}

                self._redraw()
                ev.accept()
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        vb = self.getViewBox()
        view_pos = vb.mapSceneToView(self.mapToScene(ev.pos()))

        if self._mode in ("box_select", "box_deselect") and self._box_start_view is not None:
            x0, y0 = self._box_start_view.x(), self._box_start_view.y()
            x1, y1 = view_pos.x(), view_pos.y()
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self._selection_rect.setRect(rect)
            ev.accept()
            return

        if self._mode == "drag_points" and self._drag_anchor_idx is not None:
            delta_y = view_pos.y() - self._drag_start_y
            target_val = int(
                np.clip(round(self._drag_anchor_init_y + delta_y), self.y_min, self.y_max)
            )
            for idx in self.selected_indices:
                self.y_data[idx] = target_val
            self._redraw()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._mode in ("box_select", "box_deselect"):
                rect = self._selection_rect.rect()
                self._selection_rect.hide()

                in_box = {
                    int(idx)
                    for idx in np.where(
                        (self.x_data >= rect.left())
                        & (self.x_data <= rect.right())
                        & (self.y_data >= rect.top())
                        & (self.y_data <= rect.bottom())
                    )[0]
                }

                if self._mode == "box_deselect":
                    self.selected_indices -= in_box
                else:
                    self.selected_indices |= in_box

                self._redraw()
                self._mode = None
                ev.accept()
                return

            if self._mode == "drag_points":
                indices = sorted(self.selected_indices)
                new_vals = [int(self.y_data[i]) for i in indices]
                self.valuesChanged.emit(indices, new_vals)
                self._mode = None
                self._drag_anchor_idx = None
                ev.accept()
                return

        super().mouseReleaseEvent(ev)


# =====================================================================
# MAIN APPLICATION WINDOW
# =====================================================================
class EEGAnnotatorWindow(QMainWindow):
    def __init__(self, processor: EEGProcessor, store: AnnotationStore):
        super().__init__()
        self.processor = processor
        self.store = store

        self.setWindowTitle("EEG Depth & Event Annotator (128 Hz) — VitalDB Dataset Tool")
        self.resize(1480, 990)

        self.time_eeg: np.ndarray = np.array([])
        self.event_line_pairs = []

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # -------------------------------------------------------------
        # LEFT: 6 Synchronized Plots Container (Compact Hypnotic Plot on Top)
        # -------------------------------------------------------------
        plots_container = QWidget()
        plots_layout = QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(3)
        splitter.addWidget(plots_container)

        # Plot 0 (Compact Top Strip): Hypnotic Concentration (0 to 8)
        self.p_hyp = pg.PlotWidget(title="Hypnotic Concentration (0–8: Propofol µg/mL or Sevoflurane %)")
        self.p_hyp.setLabel("left", "Conc.")
        self.p_hyp.setYRange(0, 8, padding=0.05)
        self.p_hyp.getAxis("left").setTicks([[(0, "0"), (2, "2"), (4, "4"), (6, "6"), (8, "8")]])
        self.p_hyp.showGrid(x=True, y=True, alpha=0.3)
        self.p_hyp.addLegend(offset=(10, 5))
        plots_layout.addWidget(self.p_hyp, stretch=1)

        # Plot 1: Raw EEG
        self.p_eeg = pg.PlotWidget(
            title="1. Raw EEG Signal (128 Hz) — Green Lines = Crop Boundaries | Dashed Lines = Entry Events"
        )
        self.p_eeg.setLabel("left", "Amplitude", units="µV")
        self.p_eeg.showGrid(x=True, y=True, alpha=0.3)
        self.curve_eeg = self.p_eeg.plot(pen=pg.mkPen("#4ec9b0", width=1))
        self.curve_eeg.setDownsampling(auto=True, method="peak")
        self.curve_eeg.setClipToView(True)

        crop_pen = pg.mkPen("#ff4d4d", width=2.5)
        crop_hover_pen = pg.mkPen("#ff8080", width=3.5)

        self.crop_start_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=crop_pen,
            hoverPen=crop_hover_pen,
            label="Keep Start",
            labelOpts={"position": 0.12, "color": "#ff4d4d", "movable": True},
        )
        self.crop_end_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=crop_pen,
            hoverPen=crop_hover_pen,
            label="Keep End",
            labelOpts={"position": 0.12, "color": "#ff4d4d", "movable": True},
        )
        self.crop_start_line.setZValue(15)
        self.crop_end_line.setZValue(15)
        self.crop_start_line.sigPositionChanged.connect(self._on_crop_lines_moved)
        self.crop_end_line.sigPositionChanged.connect(self._on_crop_lines_moved)

        self.p_eeg.addItem(self.crop_start_line)
        self.p_eeg.addItem(self.crop_end_line)
        plots_layout.addWidget(self.p_eeg, stretch=2)

        # Plot 2: Spectrogram
        self.p_spec = pg.PlotWidget(
            title="2. Spectrogram (0.5 - 45 Hz) — Drag Vertical Markers for Shallow / BS Entry"
        )
        self.p_spec.setLabel("left", "Frequency", units="Hz")
        self.img_spec = pg.ImageItem()
        self.p_spec.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get("viridis"))
        plots_layout.addWidget(self.p_spec, stretch=2)

        # Plot 3: Sliding Window Power
        self.p_power = pg.PlotWidget(title="3. Sliding Window Power (30s window, 10s step)")
        self.p_power.setLabel("left", "Power", units="µV²")
        self.p_power.showGrid(x=True, y=True, alpha=0.3)
        self.curve_power = self.p_power.plot(
            pen=pg.mkPen("#ce9178", width=2), symbol="o", symbolSize=5, symbolBrush="#ce9178"
        )
        plots_layout.addWidget(self.p_power, stretch=2)

        # Plot 4: Interactive Drag-and-Drop State (0..20)
        self.p_state = InteractiveAnnotationPlot(
            y_min=0,
            y_max=20,
            line_color="#569cd6",
            default_brush="#569cd6",
            title="4. Depth State (0–20) — Drag Points | Shift+Box Select | Ctrl+Shift+Box Deselect",
        )
        self.p_state.setLabel("left", "State", units="0-20")
        self.p_state.setYRange(-1, 21)
        self.p_state.showGrid(x=True, y=True, alpha=0.3)
        self.p_state.valuesChanged.connect(self._on_state_points_dragged)
        plots_layout.addWidget(self.p_state, stretch=2)

        # Plot 5: Interactive Drag-and-Drop Clinical Condition (0..5)
        self.p_cond = InteractiveAnnotationPlot(
            y_min=0,
            y_max=len(CONDITIONS) - 1,
            line_color="#c586c0",
            default_brush="#c586c0",
            title="5. Clinical Condition — Drag Points | Shift+Box Select | Ctrl+Shift+Box Deselect",
        )
        self.p_cond.setLabel("left", "Condition")
        self.p_cond.setLabel("bottom", "Time", units="s")
        self.p_cond.setYRange(-0.5, len(CONDITIONS) - 0.5)
        self.p_cond.showGrid(x=True, y=True, alpha=0.3)
        y_ticks = [(i, name) for i, name in enumerate(CONDITIONS)]
        self.p_cond.getAxis("left").setTicks([y_ticks])
        self.p_cond.valuesChanged.connect(self._on_condition_points_dragged)
        plots_layout.addWidget(self.p_cond, stretch=2)

        # Synchronize X-axes across all 6 plots
        for plot in [self.p_hyp, self.p_spec, self.p_power, self.p_state, self.p_cond]:
            plot.setXLink(self.p_eeg)

        # -------------------------------------------------------------
        # RIGHT: Control Panel
        # -------------------------------------------------------------
        control_panel = QWidget()
        control_panel.setFixedWidth(340)
        ctrl_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)

        # Group 1: File, Sample ID & Auto-Detected Hypnotic Agent
        grp_file = QGroupBox("1. Dataset, Sample & Hypnotic Agent")
        lay_file = QVBoxLayout(grp_file)

        self.lbl_sample_id = QLabel("Sample ID: No file loaded")
        self.lbl_sample_id.setWordWrap(True)
        self.lbl_sample_id.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        lay_file.addWidget(self.lbl_sample_id)

        self.lbl_file_status = QLabel("Status: Waiting for file...")
        self.lbl_file_status.setWordWrap(True)
        self.lbl_file_status.setStyleSheet("color: #aaaaaa;")
        lay_file.addWidget(self.lbl_file_status)

        self.lbl_hypnotic_agent = QLabel("Hypnotic Agent (auto): —")
        self.lbl_hypnotic_agent.setStyleSheet("font-weight: bold; color: #4fc1ff;")
        lay_file.addWidget(self.lbl_hypnotic_agent)

        btn_load = QPushButton("📂 Open Recording (.npz / .npy)")
        btn_load.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_load.clicked.connect(self._open_recording_file)
        lay_file.addWidget(btn_load)
        ctrl_layout.addWidget(grp_file)

        # Group 2: Valid Signal Crop Boundaries (EEG Plot)
        grp_crop = QGroupBox("2. Valid Recording Crop (EEG Plot)")
        lay_crop = QVBoxLayout(grp_crop)
        self.lbl_crop = QLabel("Keep Signal: —")
        self.lbl_crop.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        lay_crop.addWidget(self.lbl_crop)
        lay_crop.addWidget(
            QLabel(
                "Drag the green 'Keep Start' and 'Keep End'\n"
                "vertical lines on the EEG plot to trim\n"
                "unplugged electrodes at start/end."
            )
        )
        ctrl_layout.addWidget(grp_crop)

        # Group 3: Interactive Editing Guide
        grp_edit = QGroupBox("3. Direct Point Editing (Plots 4 & 5)")
        lay_edit = QVBoxLayout(grp_edit)
        lay_edit.addWidget(
            QLabel(
                "• Drag Point Vertically: Modify single point\n"
                "• Shift + Left-Drag Box: Select points\n"
                "• Ctrl + Shift + Drag Box: Deselect points\n"
                "• Drag Selected Point: Align group to level\n"
            )
        )
        btn_clear_sel = QPushButton("✖ Clear All Point Selections")
        btn_clear_sel.clicked.connect(self._clear_all_selections)
        lay_edit.addWidget(btn_clear_sel)
        ctrl_layout.addWidget(grp_edit)

        # Group 4: Point Event Markers
        grp_events = QGroupBox("4. Entry Markers (EEG & Spectrogram)")
        lay_events = QVBoxLayout(grp_events)
        lay_events.addWidget(QLabel("Adds synced draggable bars on EEG & Spec:"))

        btn_row_ev = QHBoxLayout()
        btn_shallow = QPushButton("+ Shallow Entry")
        btn_shallow.setStyleSheet("background-color: #8a6d1b; color: white;")
        btn_shallow.clicked.connect(lambda: self._add_event_marker("shallow_entry"))
        btn_row_ev.addWidget(btn_shallow)

        btn_bs = QPushButton("+ BS Entry")
        btn_bs.setStyleSheet("background-color: #8a2be2; color: white;")
        btn_bs.clicked.connect(lambda: self._add_event_marker("bs_entry"))
        btn_row_ev.addWidget(btn_bs)
        lay_events.addLayout(btn_row_ev)

        self.list_events = QListWidget()
        self.list_events.setMaximumHeight(95)
        lay_events.addWidget(self.list_events)

        btn_del_event = QPushButton("🗑 Remove Selected Event")
        btn_del_event.clicked.connect(self._remove_selected_event)
        lay_events.addWidget(btn_del_event)
        ctrl_layout.addWidget(grp_events)

        # Group 5: Selective Save Options & Validation
        grp_save = QGroupBox("5. Selective Save & Validation")
        lay_save = QVBoxLayout(grp_save)

        self.chk_save_states = QCheckBox("Save Depth State (0–20)")
        self.chk_save_states.setChecked(True)
        lay_save.addWidget(self.chk_save_states)

        self.chk_save_conditions = QCheckBox("Save Clinical Classes (ok, shallow, ...)")
        self.chk_save_conditions.setChecked(True)
        lay_save.addWidget(self.chk_save_conditions)

        self.chk_save_events = QCheckBox("Save Entry Events (Shallow / BS)")
        self.chk_save_events.setChecked(True)
        lay_save.addWidget(self.chk_save_events)

        self.chk_save_crop = QCheckBox("Save Crop Boundaries (Keep Start / End)")
        self.chk_save_crop.setChecked(True)
        lay_save.addWidget(self.chk_save_crop)

        self.btn_save = QPushButton("💾 Validate & Save Annotated Sample (.npz)")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_save.clicked.connect(self._save_annotations)
        lay_save.addWidget(self.btn_save)
        ctrl_layout.addWidget(grp_save)
        ctrl_layout.addStretch()

    # =================================================================
    # REAL FILE LOADING & EXISTING ANNOTATION DETECTION
    # =================================================================
    def _open_recording_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VitalDB Case (.npz) or EEG (.npy)",
            "",
            "EEG / VitalDB Files (*.npz *.npy);;NumPy Archives (*.npz);;NumPy Arrays (*.npy)",
        )
        if not file_path:
            return

        try:
            # Case 1: User selected an already-annotated .npz file directly
            if AnnotationStore.is_annotated_npz(file_path):
                self._load_existing_annotated_file(file_path)
                return

            # Case 2: Check if sidecar *_annotated.npz exists next to raw file
            sidecar_annotated = AnnotationStore.get_default_annotated_path(file_path)
            if sidecar_annotated != file_path and AnnotationStore.is_annotated_npz(sidecar_annotated):
                reply = QMessageBox.question(
                    self,
                    "Existing Annotations Detected",
                    f"An annotated file already exists for this recording:\n"
                    f"{os.path.basename(sidecar_annotated)}\n\n"
                    f"Would you like to load the existing annotations to continue editing?\n"
                    f"(Click 'No' to start fresh from the raw recording.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._load_existing_annotated_file(sidecar_annotated, raw_fallback_path=file_path)
                    return

            # Case 3: Load fresh raw recording
            eeg, track_name = self.processor.extract_eeg_from_file(file_path)
            hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(file_path)

            sample_id = os.path.splitext(os.path.basename(file_path))[0]
            if sample_id.endswith("_annotated"):
                sample_id = sample_id[: -len("_annotated")]

            self._ingest_raw_eeg(
                sample_id=sample_id,
                eeg=eeg,
                source_path=file_path,
                track_name=track_name,
                hypnotic_type=detected_hyp,
                hypnotic_tracks=hyp_tracks,
            )

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load recording:\n{e}")

    def _clear_visual_events(self):
        for eeg_line, spec_line in self.event_line_pairs:
            self.p_eeg.removeItem(eeg_line)
            self.p_spec.removeItem(spec_line)
        self.event_line_pairs.clear()
        self.list_events.clear()

    def _reset_save_checkboxes(self, states=True, conditions=True, events=True, crop=True):
        self.chk_save_states.setChecked(states)
        self.chk_save_conditions.setChecked(conditions)
        self.chk_save_events.setChecked(events)
        self.chk_save_crop.setChecked(crop)

    def _ingest_raw_eeg(
        self,
        sample_id: str,
        eeg: np.ndarray,
        source_path: str,
        track_name: str,
        hypnotic_type: str,
        hypnotic_tracks: list,
    ):
        self._clear_visual_events()

        freqs, t_spec, Sxx_db = self.processor.compute_spectrogram(eeg)
        t_windows, powers = self.processor.compute_sliding_power(
            eeg, win_sec=self.store.win_sec, step_sec=self.store.step_sec
        )
        initial_states = self.processor.estimate_initial_state(powers)

        self.store.initialize_record(
            sample_id=sample_id,
            eeg_data=eeg,
            t_windows=t_windows,
            powers=powers,
            initial_states=initial_states,
            source_path=source_path,
            eeg_track_name=track_name,
            hypnotic_type=hypnotic_type,
            hypnotic_tracks=hypnotic_tracks,
        )

        self._reset_save_checkboxes(True, True, True, True)
        self._render_store_to_plots(freqs, t_spec, Sxx_db)
        self.lbl_file_status.setText(f"Status: New Raw Recording ({track_name})")
        self.lbl_file_status.setStyleSheet("color: #ce9178; font-weight: bold;")
        self.btn_save.setEnabled(True)

    def _load_existing_annotated_file(self, annotated_path: str, raw_fallback_path: str = ""):
        self._clear_visual_events()

        hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(annotated_path)
        if not hyp_tracks and raw_fallback_path:
            hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(raw_fallback_path)

        self.store.load_from_annotated_npz(
            annotated_path,
            fallback_initial_states_fn=self.processor.estimate_initial_state,
            hypnotic_tracks=hyp_tracks,
        )
        if self.store.hypnotic_type == "unknown" and detected_hyp != "unknown":
            self.store.hypnotic_type = detected_hyp

        # Restore checkboxes to match what was previously saved
        sec = self.store.loaded_sections
        self._reset_save_checkboxes(
            states=sec.get("states", True),
            conditions=sec.get("conditions", True),
            events=sec.get("events", True),
            crop=sec.get("crop", True),
        )

        freqs, t_spec, Sxx_db = self.processor.compute_spectrogram(self.store.eeg_data)
        self._render_store_to_plots(freqs, t_spec, Sxx_db)

        for ev in self.store.events:
            self._spawn_event_line_pair(ev.time, ev.event_type)
        self._refresh_event_list()

        saved_names = [k for k, v in sec.items() if v]
        self.lbl_file_status.setText(
            f"Status: Resumed ({os.path.basename(annotated_path)})\n"
            f"Previously saved: {', '.join(saved_names) if saved_names else 'none'}"
        )
        self.lbl_file_status.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        self.btn_save.setEnabled(True)

    def _render_store_to_plots(self, freqs: np.ndarray, t_spec: np.ndarray, Sxx_db: np.ndarray):
        self.time_eeg = np.arange(len(self.store.eeg_data)) / float(self.processor.fs)
        self.lbl_sample_id.setText(f"Sample ID: {self.store.sample_id}")
        self.lbl_hypnotic_agent.setText(f"Hypnotic Agent (auto): {self.store.hypnotic_type}")

        # Plot 0: Hypnotic Concentration (0 to 8)
        self.p_hyp.clear()
        self.p_hyp.setYRange(0, 8, padding=0.05)
        if self.store.hypnotic_tracks:
            palette = ["#4fc1ff", "#ff9d00", "#89d185", "#f48771"]
            for i, trk in enumerate(self.store.hypnotic_tracks):
                pen = pg.mkPen(palette[i % len(palette)], width=2)
                self.p_hyp.plot(trk["time"], trk["val"], pen=pen, name=trk["name"])
            self.p_hyp.setTitle(f"Hypnotic Concentration (0–8) — Agent: {self.store.hypnotic_type}")
        else:
            self.p_hyp.setTitle("Hypnotic Concentration (0–8) — No concentration track found (unknown)")

        # Plot 1: Raw EEG
        self.curve_eeg.setData(self.time_eeg, self.store.eeg_data)

        # Plot 2: Spectrogram
        self.img_spec.setImage(Sxx_db.T, autoLevels=True)
        dt = t_spec[1] - t_spec[0] if len(t_spec) > 1 else 1.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        self.img_spec.setRect(t_spec[0], freqs[0], len(t_spec) * dt, len(freqs) * df)

        # Plots 3, 4, 5
        self.curve_power.setData(self.store.t_windows, self.store.powers)
        self.p_state.set_series(self.store.t_windows, self.store.states)
        self.p_cond.set_series(self.store.t_windows, self.store.conditions)

        max_t = float(self.time_eeg[-1])
        self.p_eeg.setXRange(0, max_t, padding=0.01)
        self._set_crop_bounds_ui(self.store.keep_start_sec, self.store.keep_end_sec)

    # =================================================================
    # CROP BOUNDARY LINES & POINT DRAG HANDLERS
    # =================================================================
    def _on_crop_lines_moved(self):
        start_val = float(self.crop_start_line.value())
        end_val = float(self.crop_end_line.value())

        if start_val > end_val:
            if self.sender() is self.crop_start_line:
                start_val = end_val
                self.crop_start_line.blockSignals(True)
                self.crop_start_line.setValue(start_val)
                self.crop_start_line.blockSignals(False)
            else:
                end_val = start_val
                self.crop_end_line.blockSignals(True)
                self.crop_end_line.setValue(end_val)
                self.crop_end_line.blockSignals(False)

        self.store.set_crop_bounds(start_val, end_val)
        self.lbl_crop.setText(
            f"Keep Signal: {self.store.keep_start_sec:.1f}s — {self.store.keep_end_sec:.1f}s"
        )

    def _set_crop_bounds_ui(self, start_t: float, end_t: float):
        self.store.set_crop_bounds(start_t, end_t)

        max_t = float(self.time_eeg[-1]) if len(self.time_eeg) > 0 else end_t
        for line, pos in (
            (self.crop_start_line, self.store.keep_start_sec),
            (self.crop_end_line, self.store.keep_end_sec),
        ):
            line.blockSignals(True)
            line.setBounds([0.0, max_t])
            line.setValue(pos)
            line.blockSignals(False)

        self.lbl_crop.setText(
            f"Keep Signal: {self.store.keep_start_sec:.1f}s — {self.store.keep_end_sec:.1f}s"
        )

    def _on_state_points_dragged(self, indices: list, new_values: list):
        self.store.set_states_at_indices(indices, new_values)

    def _on_condition_points_dragged(self, indices: list, new_values: list):
        for idx, val in zip(indices, new_values):
            self.store.set_conditions_at_indices([idx], val)

    def _clear_all_selections(self):
        self.p_state.clear_selection()
        self.p_cond.clear_selection()

    # =================================================================
    # POINT EVENT MARKERS (SYNCHRONIZED ON EEG & SPECTROGRAM)
    # =================================================================
    def _add_event_marker(self, event_type: str):
        if len(self.store.eeg_data) == 0:
            return
        x_min, x_max = self.p_spec.getViewBox().viewRange()[0]
        event_time = 0.5 * (x_min + x_max)
        self.store.add_event(event_time, event_type)
        self._spawn_event_line_pair(event_time, event_type)
        self._refresh_event_list()

    def _spawn_event_line_pair(self, event_time: float, event_type: str):
        color = "#ffd700" if event_type == "shallow_entry" else "#da70d6"
        hover_color = "#fff176" if event_type == "shallow_entry" else "#f3a6ff"
        label_txt = "Shallow Entry" if event_type == "shallow_entry" else "BS Entry"

        eeg_line = pg.InfiniteLine(
            pos=event_time,
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(hover_color, width=3.5, style=Qt.PenStyle.DashLine),
            label=label_txt,
            labelOpts={"position": 0.88, "color": color, "movable": True},
        )
        spec_line = pg.InfiniteLine(
            pos=event_time,
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(hover_color, width=3.5, style=Qt.PenStyle.DashLine),
            label=label_txt,
            labelOpts={"position": 0.88, "color": color, "movable": True},
        )
        eeg_line.setZValue(25)
        spec_line.setZValue(25)

        self.p_eeg.addItem(eeg_line)
        self.p_spec.addItem(spec_line)

        pair = (eeg_line, spec_line)
        self.event_line_pairs.append(pair)

        eeg_line.sigPositionChanged.connect(
            lambda moved=eeg_line, p=pair: self._on_event_line_moved(moved, p)
        )
        spec_line.sigPositionChanged.connect(
            lambda moved=spec_line, p=pair: self._on_event_line_moved(moved, p)
        )

    def _on_event_line_moved(self, moved_line: pg.InfiniteLine, pair: tuple):
        new_pos = float(moved_line.value())
        idx = self.event_line_pairs.index(pair)
        self.store.update_event_time(idx, new_pos)

        for line in pair:
            if line is not moved_line:
                line.blockSignals(True)
                line.setValue(new_pos)
                line.blockSignals(False)

        self._refresh_event_list()

    def _refresh_event_list(self):
        self.list_events.clear()
        for i, ev in enumerate(self.store.events):
            self.list_events.addItem(f"[{i + 1}] {ev.event_type} @ {ev.time:.2f} s")

    def _remove_selected_event(self):
        row = self.list_events.currentRow()
        if row < 0 or row >= len(self.store.events):
            return

        self.store.remove_event(row)
        eeg_line, spec_line = self.event_line_pairs.pop(row)
        self.p_eeg.removeItem(eeg_line)
        self.p_spec.removeItem(spec_line)
        self._refresh_event_list()

    # =================================================================
    # DIRECT SIDE-BY-SIDE SELECTIVE MANUAL SAVING
    # =================================================================
    def _save_annotations(self):
        if not self.store.source_path:
            QMessageBox.warning(
                self, "No Source File", "No recording path is associated with the current session."
            )
            return

        save_states = self.chk_save_states.isChecked()
        save_conditions = self.chk_save_conditions.isChecked()
        save_events = self.chk_save_events.isChecked()
        save_crop = self.chk_save_crop.isChecked()

        if not any([save_states, save_conditions, save_events, save_crop]):
            QMessageBox.warning(
                self,
                "Nothing Selected to Save",
                "Please check at least one annotation section (States, Classes, Events, or Crop) before saving.",
            )
            return

        save_path = AnnotationStore.get_default_annotated_path(self.store.source_path)

        try:
            summary = self.store.save_annotated_npz(
                save_path,
                save_states=save_states,
                save_conditions=save_conditions,
                save_events=save_events,
                save_crop=save_crop,
            )
            self.store.source_path = save_path
            saved_list = [k for k, v in summary["saved_sections"].items() if v]

            self.lbl_file_status.setText(
                f"Status: Saved ({os.path.basename(save_path)})\n"
                f"Saved sections: {', '.join(saved_list)}"
            )
            self.lbl_file_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")

            QMessageBox.information(
                self,
                "Saved Successfully",
                f"Saved annotated dataset alongside original recording:\n{save_path}\n\n"
                f"• Sample ID: {self.store.sample_id}\n"
                f"• Hypnotic Agent: {self.store.hypnotic_type}\n"
                f"• Saved Sections: {', '.join(saved_list)}\n"
                f"• Kept Duration: {summary['kept_duration_sec']:.1f}s ({summary['kept_samples']} samples)\n"
                f"• Valid Labeled Windows: {summary['kept_windows']}\n"
                f"• Events Inside Crop: {summary['kept_events']}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")