"""Menu-bar construction for the main window.

Builds the ``View`` and ``Help`` menus and exposes the theme-action dict
so the main window can keep the theme submenu's checkmarks in sync.

Kept as plain module-level functions (no class) because the menubar is
built exactly once and has no per-instance state beyond the returned
theme-action mapping.
"""
from collections.abc import Callable

from qtpy.QtWidgets import QAction, QMainWindow, QMessageBox

from .themes import THEMES


def build_menubar(
    window: QMainWindow,
    current_theme: str,
    on_theme: Callable[[str], None],
) -> dict[str, QAction]:
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

    Returns
    -------
    dict[str, QAction]
        Theme-name → ``QAction`` mapping so the caller can update
        checkmarks when the theme changes from outside the menubar.
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

    return theme_actions


def sync_theme_checkmarks(theme_actions: dict[str, QAction], name: str) -> None:
    """Tick the entry matching ``name`` in the Theme submenu; untick others."""
    for n, act in theme_actions.items():
        act.setChecked(n == name)


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
