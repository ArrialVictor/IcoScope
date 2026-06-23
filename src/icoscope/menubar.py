"""Menu-bar construction for the main window.

Builds the ``View`` and ``Help`` menus and exposes the theme-action dict
so the main window can keep the theme submenu's checkmarks in sync.

Kept as plain module-level functions (no class) because the menubar is
built exactly once and has no per-instance state beyond the returned
theme-action mapping.
"""
from collections.abc import Callable

from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAction, QMainWindow, QMessageBox

from .themes import THEMES

# Pane layout submenu entries: label, n_panes, keyboard shortcut.
PANE_LAYOUTS = (
    ("Single", 1, "Ctrl+1"),
    ("1 x 2", 2, "Ctrl+2"),
    ("2 x 2", 4, "Ctrl+4"),
)


def build_menubar(
    window: QMainWindow,
    current_theme: str,
    on_theme: Callable[[str], None],
    on_pane_layout: Callable[[int], None] | None = None,
) -> tuple[dict[str, QAction], dict[int, QAction]]:
    """Build the ``View`` and ``Help`` menus on ``window``.

    Parameters
    ----------
    window
        The main window to attach the menubar to.
    current_theme
        Name of the theme currently active — the matching submenu entry
        is shown checked.
    on_theme
        Callback invoked when the user picks a theme; receives the theme
        name as its sole argument.
    on_pane_layout
        Callback invoked when the user picks a pane layout; receives the
        target number of panes (1, 2, or 4). ``None`` hides the submenu
        (useful for tabs that always render single-pane).

    Returns
    -------
    theme_actions, layout_actions
        Theme-name → ``QAction`` and n_panes → ``QAction`` mappings so
        the caller can update checkmarks when the active theme or layout
        changes from outside the menubar.
    """
    mb = window.menuBar()

    # View → Theme → Dark / Light / CB-safe. Theme is window-level
    # (affects plotter background and default overlay colours), not a
    # per-tab setting, so it lives in the menu bar.
    view_menu = mb.addMenu("&View")
    theme_menu = view_menu.addMenu("Theme")
    theme_actions: dict[str, QAction] = {}
    for name in THEMES:
        act = QAction(name, window, checkable=True)
        act.setChecked(name == current_theme)
        act.triggered.connect(lambda _checked, n=name: on_theme(n))
        theme_menu.addAction(act)
        theme_actions[name] = act

    # View → Pane layout → Single / 1×2 / 2×2. Only meaningful on the File
    # tab; the main window enables/disables the submenu based on the active
    # tab. Building it unconditionally keeps the menu structure stable —
    # the on_pane_layout=None code path just leaves it disabled.
    layout_actions: dict[int, QAction] = {}
    layout_menu = view_menu.addMenu("Pane layout")
    for label, n_panes, shortcut in PANE_LAYOUTS:
        act = QAction(label, window, checkable=True)
        act.setChecked(n_panes == 1)
        act.setShortcut(QKeySequence(shortcut))
        if on_pane_layout is not None:
            act.triggered.connect(
                lambda _checked, n=n_panes: on_pane_layout(n)
            )
        layout_menu.addAction(act)
        layout_actions[n_panes] = act
    if on_pane_layout is None:
        layout_menu.setEnabled(False)

    help_menu = mb.addMenu("&Help")

    keys_act = QAction("Keyboard && mouse", window)
    keys_act.triggered.connect(lambda: _show_shortcuts(window))
    help_menu.addAction(keys_act)

    netcdf_act = QAction("NetCDF help", window)
    netcdf_act.triggered.connect(lambda: _show_netcdf_help(window))
    help_menu.addAction(netcdf_act)

    help_menu.addSeparator()
    about_act = QAction("About IcoScope", window)
    about_act.triggered.connect(lambda: _show_about(window))
    help_menu.addAction(about_act)

    return theme_actions, layout_actions


def sync_theme_checkmarks(theme_actions: dict[str, QAction], name: str) -> None:
    """Tick the entry matching ``name`` in the Theme submenu; untick others."""
    for n, act in theme_actions.items():
        act.setChecked(n == name)


def sync_layout_checkmarks(layout_actions: dict[int, QAction], n_panes: int) -> None:
    """Tick the Pane-layout submenu entry matching ``n_panes``; untick others."""
    for n, act in layout_actions.items():
        act.setChecked(n == n_panes)


def _show_shortcuts(parent: QMainWindow) -> None:
    """Show the Keyboard & mouse help dialog."""
    QMessageBox.information(
        parent, "Keyboard & mouse",
        "<b>Mouse</b><br>"
        "Left drag — rotate<br>"
        "Right drag / scroll — zoom<br>"
        "Shift + drag — pan<br>"
        "Left click — pick a cell<br>"
        "<br><b>Keys</b><br>"
        "Esc — clear selection, stop auto-rotate<br>"
        "r — reset camera<br>"
        "f — focus on cursor<br>"
        "w / s — wireframe / surface<br>"
        "q — quit<br>"
        "<br><b>lon / lat fields</b><br>"
        "Type values + Enter to fly the camera there."
    )


def _show_netcdf_help(parent: QMainWindow) -> None:
    """Show the NetCDF schema help dialog."""
    QMessageBox.information(
        parent, "NetCDF help",
        "<b>Expected schema (CF convention):</b><br>"
        "<code>lon(cell)</code>, <code>lat(cell)</code><br>"
        "<code>bounds_lon(cell, nvertex)</code>, "
        "<code>bounds_lat(cell, nvertex)</code><br>"
        "<code>&lt;field&gt;(cell)</code> or "
        "<code>&lt;field&gt;(time, cell)</code><br><br>"
        "Common variable-name variants are auto-detected "
        "(<code>lon</code>/<code>longitude</code>/<code>clon</code>, etc.). "
        "Pentagons should pad the last <code>nvertex</code> slot with a "
        "repeated vertex.<br><br>"
        "If your file fails to load, run "
        "<code>icoscope --file &lt;path&gt; --describe</code> to inspect "
        "the schema."
    )


def _show_about(parent: QMainWindow) -> None:
    """Show the About dialog."""
    QMessageBox.about(
        parent, "IcoScope",
        "<b>IcoScope</b><br>"
        "Interactive 3D viewer for icosahedral hex/pent grids on a sphere.<br><br>"
        "Renders DYNAMICO/ICOLMDZ NetCDF output or synthetic Goldberg grids.<br>"
        "Built with PyVista + Qt."
    )
