"""Standard display controls shared between the IcoScope tabs.

``_DisplayBlock`` builds the Coloring + Overlays + Animation + Export
sections that every tab (Ico, LonLat, File) exposes. The ``with_time``
flag controls whether Animation includes the time slider, play button,
and speed spinbox — only the File tab passes ``with_time=True``.

Kept private (leading underscore) because tabs are responsible for
forwarding its signals and exposing its setters; consumers outside
:mod:`icoscope.tabs` should not touch a display block directly.
"""
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .widgets import ColorButton, _expand

# Color-by options shown for a synthetic mesh (no NetCDF loaded). The File
# tab overrides this with the loaded file's actual field names.
SYNTHETIC_COLOR_BY = [
    "None", "Latitude", "Cell kind", "Mock temperature", "Realistic temperature",
]

# Units displayed next to a picked cell's value when a synthetic colour scheme
# is active. File fields read units from FieldMeta; synthetic schemes have no
# metadata source, so the mapping is hardcoded here.
SYNTHETIC_UNITS = {
    "None": "",
    "Latitude": "°",
    "Cell kind": "",                    # categorical: 5 = pent, 6 = hex
    "Mock temperature": "K",
    "Realistic temperature": "K",
}


class _LevelSpinBox(QDoubleSpinBox):
    """Discrete-axis QDoubleSpinBox: arrow keys / wheel move by level index.

    Atmospheric level axes are typically non-uniform — LMDZ ``presnivs``
    spans ~1000 hPa near the surface down to <1 hPa in the stratosphere
    with gaps that grow from ~10 hPa to >100 hPa. A plain QDoubleSpinBox
    with ``singleStep`` in display units (hPa here) overshoots the gap
    in some regions: stepping down by 10 hPa from a level whose
    next-lower neighbour is 30 hPa away lands on a value that's still
    closer to the current level than to the next one, so the parent's
    snap-to-nearest sends it right back. Manifests as a "wall" the user
    can't arrow-key past — confusing.

    This subclass overrides ``stepBy`` to emit a separate signal the
    parent uses to advance the slider by one level index (in whichever
    display direction the user asked for), bypassing the singleStep
    arithmetic entirely. Typed input still goes through the normal
    ``valueChanged`` → snap-to-nearest path so the user can jump to any
    pressure.
    """

    levelStepRequested = Signal(int)

    def stepBy(self, steps: int) -> None:  # type: ignore[override]
        # Don't call super(); the default would mutate value by
        # singleStep * steps and then trigger valueChanged, which would
        # interfere with the index-based stepping below.
        self.levelStepRequested.emit(steps)


class _DisplayBlock(QWidget):
    """Standard display controls: Coloring + Overlays + Animation + Export.

    Three modes select which subset of groups is built:

    - ``combined`` (default; Ico / LonLat tabs): every group — Coloring,
      Overlays, Animation, Export. ``with_time=True`` additionally shows
      the time / speed / level rows inside Animation.
    - ``global`` (multi-pane File tab, Global side panel): only the
      *globally-shared* groups — Overlays, Animation (autorotate +
      playback speed only), Export.
    - ``pane`` (multi-pane File tab, per-pane side panel): only the
      *per-pane* groups — Coloring, plus the time / play / level rows
      that the selected pane carries.

    The split mirrors how multi-pane mode divides the side panel: global
    settings (theme, overlays, autorotate, layout selector) live in one
    block; per-pane settings (Color by, vertical level, etc.) live in
    another. The File tab uses ``global`` + ``pane`` together to compose
    its multi-pane side panel; the Ico and LonLat tabs use ``combined``.
    """

    _VALID_MODES = ("combined", "global", "pane")

    # Coloring
    cmap_changed        = Signal(str)
    color_by_changed    = Signal(str)
    colorbar_toggled    = Signal(bool)
    center_zero_toggled = Signal(bool)
    edge_color_changed  = Signal(str)
    coast_color_changed = Signal(str)
    grat_color_changed  = Signal(str)
    cbar_color_changed  = Signal(str)

    # Overlays
    coastlines_toggled  = Signal(bool)
    graticule_toggled   = Signal(bool)
    edges_toggled       = Signal(bool)
    coast_width_changed = Signal(float)
    grat_width_changed  = Signal(float)
    edge_width_changed  = Signal(float)

    # Animation
    autorotate_toggled  = Signal(bool)
    sync_cameras_toggled = Signal(bool)

    # Time (only emitted if with_time=True)
    time_changed        = Signal(int)
    play_toggled        = Signal(bool)
    play_speed_changed  = Signal(int)

    # Vertical level (only emitted if with_time=True; gated by current field)
    level_changed       = Signal(int)

    # Export
    export_clicked         = Signal()

    def __init__(self, cmaps: list[str], with_time: bool,
                 parent: QWidget | None = None,
                 mode: str = "combined"):
        super().__init__(parent)
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}; got {mode!r}"
            )
        self.mode = mode
        # `with_time` only meaningfully applies in "combined" and "pane"
        # modes; "global" ignores it because the global side panel never
        # shows time / level rows.
        self.with_time = with_time and mode in ("combined", "pane")
        # Vertical-level state populated by set_levels:
        # - _levels_raw: the level array in the file's native units
        # - _level_units: the units attribute as a plain string (may be "")
        # - _level_display: the parsed display config from _classify_level_unit
        #   (factor, suffix, decimals, single-step) — all formatting +
        #   spinbox-vs-slider conversion derives from this struct
        self._levels_raw = None
        self._level_units = ""
        self._level_display: dict | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        # Group selection per mode. The animation group is split — global
        # gets autorotate + playback speed, pane gets the time / level
        # rows, combined gets everything.
        if mode in ("combined", "pane"):
            v.addWidget(self._build_coloring_group(cmaps))
        if mode in ("combined", "global"):
            v.addWidget(self._build_overlays_group())
        v.addWidget(self._build_animation_group())
        if mode in ("combined", "global"):
            v.addWidget(self._build_export_group())

    # ── group builders ────────────────────────────────────────────────────

    def _build_coloring_group(self, cmaps: list[str]) -> QGroupBox:
        col = QGroupBox("Coloring")
        cf = QFormLayout(col)
        cf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.color_by_box = _expand(QComboBox())
        self.color_by_box.addItems(SYNTHETIC_COLOR_BY)
        self.color_by_box.currentTextChanged.connect(self.color_by_changed)
        cf.addRow("Color by", self.color_by_box)

        self.cmap_box = _expand(QComboBox())
        self.cmap_box.addItems(cmaps)
        self.cmap_box.currentTextChanged.connect(self.cmap_changed)
        cf.addRow("Colormap", self.cmap_box)

        self.center_cb = QCheckBox("Symmetric scale around 0")
        self.center_cb.setToolTip(
            "Center the colormap at 0 — useful for anomalies and signed fields."
        )
        self.center_cb.toggled.connect(self.center_zero_toggled)
        cf.addRow(self.center_cb)

        # Colorbar toggle + text-colour picker share one row so the colour
        # control sits visually next to what it affects (per-pane).
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.setSpacing(6)
        self.bar_cb = QCheckBox("Colorbar")
        self.bar_cb.setChecked(True)
        self.bar_cb.toggled.connect(self.colorbar_toggled)
        bar_row.addWidget(self.bar_cb, stretch=1)
        self.cbar_btn = ColorButton("#ffffff")
        self.cbar_btn.setToolTip("Colorbar text colour")
        self.cbar_btn.color_changed.connect(self.cbar_color_changed)
        bar_row.addWidget(self.cbar_btn)
        bar_wrap = QWidget()
        bar_wrap.setLayout(bar_row)
        cf.addRow(bar_wrap)

        # color-by defaults to "None" → cmap-related widgets start disabled
        self.center_cb.setEnabled(False)
        self.bar_cb.setEnabled(False)
        self.cbar_btn.setEnabled(False)
        self.cmap_box.setEnabled(False)

        return col

    def _build_overlays_group(self) -> QGroupBox:
        ov = QGroupBox("Overlays")
        ol = QVBoxLayout(ov)
        ol.setSpacing(2)

        def toggle_color_row(label, signal_toggle, signal_color, signal_width,
                             default_width, checked=False):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.toggled.connect(signal_toggle)
            width = QDoubleSpinBox()
            width.setRange(0.2, 10.0)
            width.setSingleStep(0.2)
            width.setDecimals(1)
            width.setValue(default_width)
            width.setKeyboardTracking(False)
            width.setFixedWidth(54)
            width.valueChanged.connect(signal_width)
            btn = ColorButton("#ffffff")
            btn.color_changed.connect(signal_color)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(cb, stretch=1)
            row.addWidget(width)
            row.addWidget(btn)
            wrap = QWidget()
            wrap.setLayout(row)
            ol.addWidget(wrap)
            return cb, btn, width

        self.coast_cb, self.coast_btn, self.coast_width = toggle_color_row(
            "Coastlines", self.coastlines_toggled, self.coast_color_changed,
            self.coast_width_changed, default_width=1.2)
        self.grat_cb, self.grat_btn, self.grat_width = toggle_color_row(
            "Graticule (30°)", self.graticule_toggled, self.grat_color_changed,
            self.grat_width_changed, default_width=0.6)
        self.edges_cb, self.edge_btn, self.edge_width = toggle_color_row(
            "Cell edges", self.edges_toggled, self.edge_color_changed,
            self.edge_width_changed, default_width=0.6, checked=True)

        return ov

    def _build_animation_group(self) -> QGroupBox:
        anim = QGroupBox("Animation")
        # Saved so set_levels / set_time_axis can hide the whole group
        # when every child row is empty (otherwise the user sees a
        # labelled-but-empty box — e.g. File-tab pane mode with no
        # vertical-level dim has time + speed hidden by the strip and
        # level hidden by lack of presnivs, leaving the group blank).
        self._anim_group = anim
        al = QVBoxLayout(anim)
        # Auto-rotate lives in global / combined. Pane-mode side panels don't
        # show it because rotation is a tab-shared interaction.
        if self.mode in ("combined", "global"):
            self.spin_cb = QCheckBox("Auto-rotate")
            self.spin_cb.toggled.connect(self.autorotate_toggled)
            al.addWidget(self.spin_cb)
        # Camera sync only makes sense when there's >1 pane available —
        # ship the toggle only on the global block (File-tab multi-pane).
        # Default ON: rotating one pane mirrors across the others.
        if self.mode == "global":
            self.sync_cb = QCheckBox("Sync cameras")
            self.sync_cb.setChecked(True)
            self.sync_cb.setToolTip(
                "Mirror camera moves across all visible panes "
                "(rotation, pan, zoom)."
            )
            self.sync_cb.toggled.connect(self.sync_cameras_toggled)
            al.addWidget(self.sync_cb)

        if not self.with_time:
            return anim

        ROW_H = 22
        SLIDER_STYLE = (
            "QSlider::groove:horizontal {"
            "  height: 4px; background: #888; border-radius: 2px; }"
            "QSlider::handle:horizontal {"
            "  width: 12px; margin: -6px 0; background: #ccc;"
            "  border: 1px solid #555; border-radius: 6px; }"
        )

        # Time row spans two visual rows: [play] [slider] on top, and the
        # full datetime label underneath. The bottom row gets unbounded
        # horizontal space so the datetime can be shown in full
        # (`YYYY-MM-DD HH:MM:SS · i/N`) without truncation.
        self.time_row = QWidget()
        trow_outer = QVBoxLayout(self.time_row)
        trow_outer.setContentsMargins(0, 0, 0, 0)
        trow_outer.setSpacing(2)
        trow_top = QHBoxLayout()
        trow_top.setContentsMargins(0, 0, 0, 0)
        trow_top.setSpacing(6)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(28, ROW_H)
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        trow_top.addWidget(self.play_btn)
        # The File-tab pane block delegates the entire time row (play
        # button + slider + datetime label) to the bottom timeline strip
        # — its per-track scrubbing + PlaybackBar replace the redundant
        # per-pane slider. Hide the row wholesale in pane mode; the
        # Ico/LonLat combined blocks keep it (they have no strip).
        if self.mode == "pane":
            self.play_btn.setVisible(False)
            self._pane_mode_hides_time_row = True
        else:
            self._pane_mode_hides_time_row = False
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setFixedHeight(ROW_H)
        self.time_slider.setStyleSheet(SLIDER_STYLE)
        self.time_slider.valueChanged.connect(self.time_changed)
        trow_top.addWidget(self.time_slider, stretch=1)
        trow_outer.addLayout(trow_top)
        # Second row: full-width datetime label, right-aligned so it sits
        # under the slider's right edge — visually anchored to the cursor.
        self.time_label = QLabel("—")
        self.time_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")
        trow_outer.addWidget(self.time_label)
        al.addWidget(self.time_row)
        self.time_row.setVisible(False)
        # Datetime axis values for the currently-active field, if available.
        # Set via set_time_axis(); None means "labels fall back to i/N".
        self._times = None
        self._n_time = 0

        self.speed_row = QWidget()
        srow = QHBoxLayout(self.speed_row)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(6)
        speed_lbl = QLabel("Step")
        speed_lbl.setFixedHeight(ROW_H)
        srow.addWidget(speed_lbl)
        self.speed_box = QSpinBox()
        self.speed_box.setRange(50, 5000)
        self.speed_box.setSingleStep(50)
        self.speed_box.setValue(500)
        self.speed_box.setSuffix(" ms")
        self.speed_box.setFixedHeight(ROW_H)
        self.speed_box.setKeyboardTracking(False)
        self.speed_box.valueChanged.connect(self.play_speed_changed)
        srow.addWidget(self.speed_box, stretch=1)
        al.addWidget(self.speed_row)
        self.speed_row.setVisible(False)
        # Speed is on the strip's PlaybackBar in pane mode — keep this
        # row in the widget tree (for back-compat references) but never
        # show it.
        self._pane_mode_hides_speed_row = (self.mode == "pane")

        # Vertical-level slider: shown only when the active field has a
        # presnivs dim. Label shows the pressure value (hPa) at the current
        # index so users can navigate by altitude, not just slider position.
        self.level_row = QWidget()
        lrow = QHBoxLayout(self.level_row)
        lrow.setContentsMargins(0, 0, 0, 0)
        lrow.setSpacing(6)
        level_lbl = QLabel("Level")
        level_lbl.setFixedHeight(ROW_H)
        lrow.addWidget(level_lbl)
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(0, 0)
        self.level_slider.setFixedHeight(ROW_H)
        self.level_slider.setStyleSheet(SLIDER_STYLE)
        self.level_slider.valueChanged.connect(self._on_level_slider_changed)
        lrow.addWidget(self.level_slider, stretch=1)
        # Editable spinbox doubles as the value display and the type-in
        # entry point: the user can drag the slider, or type the desired
        # level value (e.g. "850" for the 850 hPa surface) and the slider
        # snaps to the nearest available level. set_levels configures
        # the range / suffix / decimals from the file's units attribute.
        self.level_value_box = _LevelSpinBox()
        # Fixed width sized for 4-digit values + suffix + spin arrows;
        # 60 px clipped "1024.0 hPa" with the old QLabel.
        self.level_value_box.setFixedSize(105, ROW_H)
        self.level_value_box.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.level_value_box.setKeyboardTracking(False)
        self.level_value_box.setEnabled(False)
        self.level_value_box.valueChanged.connect(
            self._on_level_box_value_changed)
        # Arrow keys / wheel go through a separate index-based path so
        # they always move by exactly one level (see _LevelSpinBox).
        self.level_value_box.levelStepRequested.connect(
            self._on_level_box_step_requested)
        lrow.addWidget(self.level_value_box)
        al.addWidget(self.level_row)
        self.level_row.setVisible(False)

        return anim

    def _build_export_group(self) -> QGroupBox:
        exp = QGroupBox("Export")
        elay = QHBoxLayout(exp)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.export_clicked)
        elay.addWidget(self.export_btn)
        return exp

    # ── public setters ────────────────────────────────────────────────────

    def set_cmap(self, name: str) -> None:
        """Select *name* in the colormap combo box (no-op if not listed)."""
        i = self.cmap_box.findText(name)
        if i >= 0:
            self.cmap_box.setCurrentIndex(i)

    def set_color_by(self, name: str) -> None:
        """Select *name* in the color-by combo box (no-op if not listed)."""
        i = self.color_by_box.findText(name)
        if i >= 0:
            self.color_by_box.blockSignals(True)
            self.color_by_box.setCurrentIndex(i)
            self.color_by_box.blockSignals(False)

    def set_color_by_items(self, items: list[str]) -> None:
        """Repopulate the color-by combo while preserving the current selection."""
        self.color_by_box.blockSignals(True)
        cur = self.color_by_box.currentText()
        self.color_by_box.clear()
        self.color_by_box.addItems(items)
        i = self.color_by_box.findText(cur)
        self.color_by_box.setCurrentIndex(i if i >= 0 else 0)
        self.color_by_box.blockSignals(False)

    def set_edge_color(self, hex_str: str) -> None:
        """Set the edge-color swatch."""
        self.edge_btn.set_color(hex_str)

    def set_coast_color(self, hex_str: str) -> None:
        """Set the coastline-color swatch."""
        self.coast_btn.set_color(hex_str)

    def set_grat_color(self, hex_str: str) -> None:
        """Set the graticule-color swatch."""
        self.grat_btn.set_color(hex_str)

    def set_cbar_color(self, hex_str: str) -> None:
        """Set the colorbar-text-color swatch."""
        self.cbar_btn.set_color(hex_str)

    def set_time_axis(self, n_steps: int, times=None) -> None:
        """Configure the time slider for a time-varying field, or hide it.

        ``times`` is the parsed datetime array (cftime objects) for the
        active field's time axis, or ``None`` if unavailable (subsetted
        file, unparseable units, etc.). When ``None``, labels fall back
        to ``i / N``.
        """
        if not self.with_time:
            return
        if n_steps and n_steps > 1:
            self._n_time = n_steps
            self._times = times
            # Time row (play + slider + datetime label) is on the strip
            # in pane mode — keep it hidden in the side panel there.
            self.time_row.setVisible(not self._pane_mode_hides_time_row)
            # speed_row hosts Step + Loop which moved to the timeline
            # strip's PlaybackBar in pane mode — keep it hidden there.
            self.speed_row.setVisible(not self._pane_mode_hides_speed_row)
            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, n_steps - 1)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
            self.set_time_label(0)
            self.play_btn.setChecked(False)
        else:
            self._n_time = 0
            self._times = None
            self.time_row.setVisible(False)
            self.speed_row.setVisible(False)
            self.play_btn.setChecked(False)
        self._refresh_anim_group_visibility()

    def _refresh_anim_group_visibility(self) -> None:
        """Hide the Animation group when every child row is hidden.

        Pane mode hides time + speed rows wholesale (the strip owns
        those); when the active field also has no vertical levels the
        Animation group ends up empty and labelled, which looks broken.
        Combined / global blocks always have something in the group
        (auto-rotate, sync_cameras, etc.) so the hide only triggers
        for the File-tab pane block.

        Uses ``isHidden()`` rather than ``isVisible()`` for the row
        checks — the latter returns False whenever any ancestor is
        hidden, which creates a chicken-and-egg with the group itself
        (the group is hidden → child row's ``isVisible()`` is False →
        ``has_content`` is False → group stays hidden, even right after
        a ``setVisible(True)`` on the row).
        """
        if not hasattr(self, "_anim_group"):
            return
        # Anything else visible? autorotate (combined/global) or
        # sync_cameras (global) lives in the group too; if either is
        # there the group has content. In pane mode neither is built,
        # so visibility reduces to "any of the time-aware rows visible".
        has_content = (
            getattr(self, "spin_cb", None) is not None
            or getattr(self, "sync_cb", None) is not None
            or (hasattr(self, "time_row") and not self.time_row.isHidden())
            or (hasattr(self, "speed_row") and not self.speed_row.isHidden())
            or (hasattr(self, "level_row") and not self.level_row.isHidden())
        )
        self._anim_group.setVisible(has_content)

    # Back-compat alias: the per-tab caller used to be set_time_steps(n).
    # Keep it routed at the new datetime-aware setter so existing callers
    # that don't have datetimes still work.
    def set_time_steps(self, n_steps: int) -> None:
        """Configure the slider with no datetime info (index-only labels)."""
        self.set_time_axis(n_steps, times=None)

    def set_time_label(self, idx: int) -> None:
        """Update the line under the slider for sample index ``idx``."""
        if not self.with_time or self._n_time <= 0:
            return
        suffix = f" · {idx + 1}/{self._n_time}"
        if self._times is not None and 0 <= idx < len(self._times):
            try:
                stamp = self._times[idx].isoformat(sep=" ")
            except (AttributeError, TypeError):
                stamp = str(self._times[idx])
            self.time_label.setText(stamp + suffix)
            self.time_label.setToolTip("")
        else:
            self.time_label.setText(f"{idx + 1}/{self._n_time}")
            self.time_label.setToolTip(
                "time-axis values unavailable; showing sample indices"
            )

    def _on_play_toggled(self, on):
        self.play_btn.setText("⏸" if on else "▶")
        self.play_toggled.emit(on)

    def set_levels(self, levels, units: str = "") -> None:
        """Configure the level slider + spinbox from the file's level axis.

        ``levels`` is the raw values array in the file's native units;
        ``units`` is the coord variable's ``units`` attribute (``"Pa"``
        for LMDZ, ``"hPa"`` for some processed files, ``"m"`` / ``"km"``
        for height coordinates, ``""`` when missing). The display
        layer classifies these into a (factor, suffix, decimals)
        config — see :meth:`_classify_level_unit` — so the spinbox
        shows the value in the most natural unit (hPa for pressure,
        km for tall height ranges, etc.) and the typed-value path can
        snap to the nearest available level.

        Pass ``None`` or a length-≤1 array to hide the slider (used
        when the active field has no vertical dim, or when no file is
        loaded). When the values look like a contiguous integer index
        sequence (subsetted file where the coord var was dropped), the
        spinbox shows raw indices.
        """
        if not self.with_time:
            return
        if levels is None or len(levels) <= 1:
            self.level_row.setVisible(False)
            self._levels_raw = None
            self._level_units = ""
            self._level_display = None
            self.level_value_box.setEnabled(False)
            self._refresh_anim_group_visibility()
            return
        import numpy as np
        self._levels_raw = np.asarray(levels, dtype=float)
        self._level_units = units or ""
        self._level_display = self._classify_level_unit(
            self._levels_raw, self._level_units)
        self.level_row.setVisible(True)
        self._refresh_anim_group_visibility()
        self.level_slider.blockSignals(True)
        self.level_slider.setRange(0, len(self._levels_raw) - 1)
        self.level_slider.setValue(0)
        self.level_slider.blockSignals(False)
        # Configure the spinbox from the display config.
        d = self._level_display
        display_values = self._levels_raw * d["factor"]
        self.level_value_box.blockSignals(True)
        self.level_value_box.setDecimals(d["decimals"])
        self.level_value_box.setSingleStep(d["step"])
        self.level_value_box.setSuffix(d["suffix"])
        lo = float(min(display_values))
        hi = float(max(display_values))
        # Range must be lo..hi but spinbox uses min..max; allow some
        # slack so the user can type slightly outside (snap finds
        # nearest anyway) without the spinbox clamping the typed text.
        slack = max(abs(hi - lo) * 0.001, 1e-3)
        self.level_value_box.setRange(lo - slack, hi + slack)
        self.level_value_box.setEnabled(True)
        self.level_value_box.blockSignals(False)
        self.set_level_label(0)

    @staticmethod
    def _classify_level_unit(values, units_attr: str) -> dict:
        """Pick a display unit + format from a level array + units attribute.

        Returns a dict with:
            factor   — multiply a raw value to get the displayed value
            suffix   — spinbox suffix (e.g. " hPa", " m", " (idx)", "")
            decimals — spinbox decimal places
            step     — spinbox single-step (in displayed units)

        Rules (precedence top-down):
            "Pa" / "pascal(s)"            → display in hPa
            "hPa" / "mbar" / "millibar"   → display as-is in hPa
            "km" / "kilometer(s)"         → display as-is in km
            "m" / "meter(s)"              → display in km when max > 5000 m
                                            (stratospheric heights), else m
            "K" / "kelvin"                → display in K (isentropic θ)
            "" / "1" / "level" / "index"  → integer index ("level k")
            anything else                 → display raw with the original
                                            ``units`` string as the suffix
        """
        import numpy as np
        u = (units_attr or "").strip().lower()
        max_abs = float(np.max(np.abs(np.asarray(values, dtype=float))))
        if u in {"pa", "pascal", "pascals"}:
            return {"factor": 1 / 100.0, "suffix": " hPa",
                    "decimals": 1, "step": 10.0}
        if u in {"hpa", "mbar", "millibar", "millibars"}:
            return {"factor": 1.0, "suffix": " hPa",
                    "decimals": 1, "step": 10.0}
        if u in {"km", "kilometer", "kilometers"}:
            return {"factor": 1.0, "suffix": " km",
                    "decimals": 2, "step": 0.5}
        if u in {"m", "meter", "metre", "meters", "metres"}:
            if max_abs > 5000:
                return {"factor": 1 / 1000.0, "suffix": " km",
                        "decimals": 2, "step": 0.5}
            return {"factor": 1.0, "suffix": " m",
                    "decimals": 0, "step": 50.0}
        if u in {"k", "kelvin"}:
            return {"factor": 1.0, "suffix": " K",
                    "decimals": 1, "step": 1.0}
        # Dimensionless / unknown — show integer index unless the values
        # span a wide non-integer range (rare unknown-unit case).
        looks_like_index = u in {"", "1", "level", "index"} or bool(
            np.array_equal(values, np.arange(len(values), dtype=float))
        )
        if looks_like_index:
            return {"factor": 1.0, "suffix": " (idx)",
                    "decimals": 0, "step": 1.0}
        # Truly unknown units: pass them through verbatim with a sane
        # default precision.
        return {"factor": 1.0, "suffix": f" {units_attr}",
                "decimals": 3, "step": max(max_abs / 100.0, 1e-3)}

    def set_level_label(self, idx: int) -> None:
        """Push level index ``idx``'s display value into the spinbox."""
        if (not self.with_time or self._levels_raw is None
                or self._level_display is None):
            return
        if not (0 <= idx < len(self._levels_raw)):
            return
        display_val = float(self._levels_raw[idx]) * self._level_display["factor"]
        self.level_value_box.blockSignals(True)
        self.level_value_box.setValue(display_val)
        self.level_value_box.blockSignals(False)

    def _on_level_slider_changed(self, idx: int) -> None:
        self.set_level_label(idx)
        self.level_changed.emit(idx)

    def _on_level_box_value_changed(self, display_val: float) -> None:
        """User typed a level value — snap the slider to the nearest match.

        Updates the spinbox back to the snapped value (so the user sees
        what they got) and emits ``level_changed`` so downstream code
        re-renders. blockSignals around both child writes to avoid
        re-entry into this handler.
        """
        if self._levels_raw is None or self._level_display is None:
            return
        import numpy as np
        # Convert spinbox display value back to raw units to find the
        # nearest level by absolute distance in the file's own scale.
        target_raw = display_val / self._level_display["factor"]
        idx = int(np.argmin(np.abs(self._levels_raw - target_raw)))
        self.level_slider.blockSignals(True)
        self.level_slider.setValue(idx)
        self.level_slider.blockSignals(False)
        # Snap the spinbox display back to the actual level value (so the
        # user sees exactly which level they landed on, not their guess).
        snapped = float(self._levels_raw[idx]) * self._level_display["factor"]
        self.level_value_box.blockSignals(True)
        self.level_value_box.setValue(snapped)
        self.level_value_box.blockSignals(False)
        self.level_changed.emit(idx)

    def _on_level_box_step_requested(self, steps: int) -> None:
        """Arrow-key / wheel step on the spinbox — move by one level index.

        ``steps`` is +1 for one step UP (user pressed Up or scrolled
        forward) and -1 for one step DOWN. We translate to a level-
        index step in whichever direction increases / decreases the
        *display value*, because that's what the user sees.

        Crucially this bypasses ``singleStep`` and never goes through
        the snap-to-nearest path, so non-uniform level spacing can't
        leave the user stuck at a "wall" (the bug the dedicated path
        was added to fix).
        """
        if self._levels_raw is None or self._level_display is None:
            return
        if len(self._levels_raw) == 0:
            return
        import numpy as np
        cur_idx = self.level_slider.value()
        all_display = (
            np.asarray(self._levels_raw, dtype=float)
            * self._level_display["factor"]
        )
        cur_display = float(all_display[cur_idx])
        if steps > 0:
            # User wants a HIGHER display value — find the smallest
            # display value that's still strictly greater than current.
            mask = all_display > cur_display
            if not mask.any():
                return
            higher = all_display[mask]
            new_display = float(np.min(higher))
        else:
            mask = all_display < cur_display
            if not mask.any():
                return
            lower = all_display[mask]
            new_display = float(np.max(lower))
        new_idx = int(np.argmin(np.abs(all_display - new_display)))
        # Route through the slider so the existing slider-driven path
        # (set_level_label + level_changed emit) runs once.
        self.level_slider.setValue(new_idx)
