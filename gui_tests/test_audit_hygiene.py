"""Regression tests for the audit-cleanup PR D hygiene fixes.

- _render_empty_sphere clears every pane's plotter (including hidden
  panes), so the layout shrink → close sequence leaves no actors from
  the previous file on hidden panes.
- bar_config cache key includes pane.color_by, so two fields with
  coincidentally identical cmap/clim/cbar_color/title can't share a
  cached scalar-bar actor.
- _bar_config_cache is reset on file open/close alongside the clim
  caches.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    win._select_pane(pane_idx)        # synchronous; no events to drain
    win._on_color_by(field)
    QCoreApplication.processEvents()


def _pane_actor_names(win, pane_idx: int) -> set:
    """Return the set of named actors on pane ``pane_idx``'s renderer."""
    plotter = win._pane_container.pane(pane_idx).plotter
    renderer = plotter.renderer
    return set(renderer.actors.keys())


def test_render_empty_sphere_clears_hidden_panes(make_main_window):
    """Close after shrink: hidden panes' grid actors must be gone too.

    Sequence: load file → 4-pane layout → field on every pane → shrink
    to 1 (panes 2–4 still hold a 'grid' actor on their plotters) →
    close file. Before the fix, ``_render_empty_sphere`` only iterated
    visible panes, leaving the prior file's mesh sitting on the hidden
    panes' plotters until garbage collection.
    """
    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for i in range(4):
        _set_field(win, i, "tas_t")
    # Sanity: every pane has the grid actor before shrink.
    for i in range(4):
        assert "grid" in _pane_actor_names(win, i)

    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    # Hidden panes 2–4 still hold the previous file's 'grid' actor.
    for i in range(1, 4):
        assert "grid" in _pane_actor_names(win, i), (
            f"sanity: pane {i + 1} should still hold the prior grid "
            f"actor immediately after the shrink"
        )

    win._on_close_file()
    QCoreApplication.processEvents()
    for i in range(4):
        assert "grid" not in _pane_actor_names(win, i), (
            f"pane {i + 1}: close-file must clear stale 'grid' actor on "
            f"every plotter (visible AND hidden)"
        )


def test_bar_config_cache_key_includes_color_by(make_main_window):
    """Switching color_by changes the bar_config cache entry, not just clim/cmap.

    Without color_by in the key, two fields that happen to share
    (cmap, clim, title, cbar_color) would silently reuse a cached
    scalar-bar actor whose tick labels are stale.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    QCoreApplication.processEvents()
    first_key = win._bar_config_cache.get(0)
    assert first_key is not None

    _set_field(win, 0, "vort_t")
    QCoreApplication.processEvents()
    second_key = win._bar_config_cache.get(0)
    assert second_key is not None
    assert second_key != first_key, (
        "switching color_by must produce a different bar_config tuple "
        "(field name participates in the cache key)"
    )
    # Belt-and-braces: the field name must appear literally somewhere
    # in the tuple — protects against a future refactor that "stringifies"
    # color_by and accidentally collapses two fields onto the same key.
    assert any("tas_t" == part for part in first_key)
    assert any("vort_t" == part for part in second_key)


def test_bar_config_cache_resets_on_file_close(make_main_window):
    """File close must drop _bar_config_cache alongside the clim caches."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_t")
    assert win._bar_config_cache, "sanity: cache populated during render"

    win._on_close_file()
    QCoreApplication.processEvents()
    assert win._bar_config_cache == {}, (
        "close-file must reset _bar_config_cache so the next file "
        "rebuilds actors from scratch"
    )
