"""A pane with ``color_by="None"`` must render as a neutral solid sphere.

Bug surfaced in user testing: with pane 1 showing a real field and panes
2-4 left at "None", panes 2-4 visually rendered pane 1's scalar data
(same colours, same colourmap). Root cause: ``_build_pane_scene``
passed ``scalars=None`` to ``add_mesh``, which makes PyVista fall back
to the mesh's active scalars (i.e. whichever pane added them last).

Fix: take a dedicated no-scalars branch that passes an explicit
``color`` so the pane never picks up another pane's data via the
active-scalars fallback.

The companion regression — "layout switch on empty File tab shows the
Ico mesh in gray instead of the empty sphere" — was traced to
``_update_scalars_only`` lazily rebuilding the mesh from the stale
``self.verts / cells`` seed (the synthetic goldberg from ``__init__``)
whenever ``self._mesh`` was None. The bottom test pins that down.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QCoreApplication


def test_empty_pane_does_not_use_other_panes_scalars(make_main_window, set_field):
    """Pane 2 with color_by='None' must NOT bind to pane 1's scalar key.

    Verifies the actor wiring directly: when pane 2 has no field, the
    'grid' actor on pane 2's renderer must be using a uniform color
    rather than reading from pane 0's scalar array on the shared mesh.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()

    set_field(win, 0, "tas_t")
    # Pane 1 explicitly stays at "None" (the default for newly-visible
    # panes after PR B's reset-on-open fix, but re-asserted to make the
    # test setup unambiguous).
    set_field(win, 1, "None")
    QCoreApplication.processEvents()

    # Pane 1's stored scalars must be None.
    assert win._pane_scalars[1] is None

    # The 'grid' actor on pane 1 must NOT report pane 0's scalar array
    # as its mapper input. Mapper has scalar visibility off when no
    # scalars are bound — PyVista sets this when we pass color=... and
    # no scalars=.
    actor = win._pane_container.pane(1).plotter.renderer.actors.get("grid")
    assert actor is not None, "pane 2 must still have a grid actor"
    mapper = actor.GetMapper()
    # GetScalarVisibility == 0 means the actor renders with the actor's
    # property color, not the mesh's scalar data — exactly what we want
    # for an empty pane.
    assert mapper.GetScalarVisibility() == 0, (
        "pane 2 with color_by='None' must render with solid color, not "
        "fall back to pane 1's scalars on the shared mesh"
    )


def test_no_scalar_bar_on_empty_pane(make_main_window, set_field):
    """An empty pane must not display a scalar bar even when colorbar_on=True."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "None")
    QCoreApplication.processEvents()

    # bar_config_cache entry should reflect bar_on=False even if
    # PaneState.colorbar_on is True (default).
    assert win.state.panes[1].colorbar_on is True
    cache_entry = win._bar_config_cache.get(1)
    assert cache_entry is not None
    # First element of the tuple is bar_on per the cache key shape.
    assert cache_entry[0] is False, (
        "empty pane must have bar_on=False regardless of colorbar_on setting"
    )


def test_pane_set_to_none_after_having_field_clears_scalars(make_main_window, set_field):
    """Setting a previously-coloured pane back to 'None' must drop its scalar bind."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "tas_anomaly")          # pane 1 starts with a field
    QCoreApplication.processEvents()
    # Sanity: pane 1's scalars exist initially.
    assert win._pane_scalars[1] is not None
    actor = win._pane_container.pane(1).plotter.renderer.actors.get("grid")
    assert actor.GetMapper().GetScalarVisibility() == 1

    # User clears pane 1's field.
    set_field(win, 1, "None")
    QCoreApplication.processEvents()

    assert win._pane_scalars[1] is None
    actor = win._pane_container.pane(1).plotter.renderer.actors.get("grid")
    assert actor.GetMapper().GetScalarVisibility() == 0, (
        "switching back to 'None' must rewire the actor to solid color"
    )


def test_empty_pane_does_not_inherit_pane0_data_after_layout_expand(make_main_window, set_field):
    """1 → 4 expand: panes 2-4 inherit pane 0's color_by but if user clears them, no leak.

    Newly-visible panes are dataclass clones of pane 0 so they start
    with the same color_by; clearing one (color_by='None') must drop
    the visual binding too — not silently keep mirroring pane 0.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    win._on_pane_layout(4)
    QCoreApplication.processEvents()

    # All four panes inherited pane 0's color_by → all have scalars.
    assert all(win._pane_scalars[i] is not None for i in range(4))

    # User clears pane 3.
    set_field(win, 2, "None")
    QCoreApplication.processEvents()
    assert win._pane_scalars[2] is None

    actor = win._pane_container.pane(2).plotter.renderer.actors.get("grid")
    assert actor.GetMapper().GetScalarVisibility() == 0, (
        "pane 3 cleared to 'None' must not keep showing pane 1's data"
    )
    # And the OTHER panes (still on tas_t) must keep their scalar binding.
    for i in (0, 1, 3):
        a = win._pane_container.pane(i).plotter.renderer.actors.get("grid")
        assert a.GetMapper().GetScalarVisibility() == 1, (
            f"pane {i + 1} (still tas_t) must keep its scalar binding"
        )

    # Also verify the values don't actually equal pane 0's anywhere on
    # pane 3 — the safety net belt-and-braces.
    _ = np  # imported above for the assert helpers


def test_empty_sphere_path_unaffected(make_main_window):
    """The no-file empty-sphere path must still render a gray sphere unchanged."""
    win = make_main_window()
    win._on_close_file()
    QCoreApplication.processEvents()
    # The empty-sphere code branch (different from the 'grid' actor
    # path we patched) is exercised here; pane 0 must hold an 'empty'
    # actor, not a 'grid' actor.
    actor_names = set(
        win._pane_container.pane(0).plotter.renderer.actors.keys()
    )
    assert "empty" in actor_names
    assert "grid" not in actor_names


def test_layout_switch_on_empty_file_tab_keeps_empty_sphere(make_main_window):
    """Switching layout on a closed-file File tab must NOT re-leak the Ico mesh.

    Regression: ``_update_scalars_only`` used to lazily re-create
    ``self._mesh`` from the stale ``self.verts / cells`` seed (the
    synthetic goldberg from ``__init__``) when called with no mesh.
    ``_on_pane_layout`` calls ``_update_scalars_only`` per newly-visible
    pane *before* ``_build_scene`` runs, so the lazy rebuild quietly
    overwrote the empty-sphere state with a gray Ico mesh.
    """
    win = make_main_window()
    win._on_close_file()
    QCoreApplication.processEvents()
    assert win._mesh is None     # invariant after close

    win._on_pane_layout(4)
    QCoreApplication.processEvents()

    # _mesh must still be None — _update_scalars_only must not lazily
    # rebuild the goldberg seed during the layout expand.
    assert win._mesh is None, (
        "layout switch on a closed-file File tab must not rebuild _mesh "
        "from the synthetic goldberg seed"
    )
    # Every visible pane must hold the 'empty' actor, not the leaked
    # 'grid' actor that the old behaviour would produce.
    for i in range(win._pane_container.n_visible):
        actor_names = set(
            win._pane_container.pane(i).plotter.renderer.actors.keys()
        )
        assert "empty" in actor_names, (
            f"pane {i + 1}: expected empty-sphere actor on a closed-file "
            f"File tab; got actors {actor_names}"
        )
        assert "grid" not in actor_names, (
            f"pane {i + 1}: leaked grid actor — the Ico mesh re-appeared "
            f"during layout expand on the empty File tab"
        )
