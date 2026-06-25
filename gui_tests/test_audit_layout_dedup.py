"""Layout expansion must not re-read the same slab N times.

The runtime perf audit flagged ``_on_pane_layout`` for fetching pane 0's
scalar slab once per newly-visible pane during a 1 → N expansion. The
fix shares pane 0's scalar array reference with fresh dataclass clones
rather than re-reading from disk. These tests pin down:

- Behaviour: clones receive scalars equal to pane 0's.
- Non-regression: after a clone customises its own state, refreshing it
  produces an independent array (the shared reference doesn't cause
  write-aliasing).
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QCoreApplication


def test_layout_expand_shares_pane0_scalars_with_clones(make_main_window, set_field):
    """Expanding 1 → 4 leaves pane 1-3 with identical scalars to pane 0."""
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")

    pane0_scalars = win._pane_scalars[0]
    assert pane0_scalars is not None

    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for i in range(1, 4):
        # Same color_by / time / level → identical scalar values.
        np.testing.assert_array_equal(win._pane_scalars[i], pane0_scalars)


def test_clone_pane_refresh_writes_independent_array(make_main_window, set_field):
    """After expand, customising a clone's field doesn't mutate pane 0's array.

    The shared-reference optimisation is only safe because every state
    change routes through ``_refresh_scalars``, which assigns a fresh
    numpy array rather than mutating in place. This test pins that
    contract down — if a future refactor switches to in-place updates,
    the dedup optimisation would silently corrupt pane 0.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")           # populate pane 0 first
    win._on_pane_layout(4)               # expand → panes 1..3 share pane 0's array
    QCoreApplication.processEvents()
    # Right after expand, panes 0 and 1 share the same array object.
    assert win._pane_scalars[1] is win._pane_scalars[0]
    pane0_id = id(win._pane_scalars[0])

    # Switch pane 1 to a different field — its scalars must diverge.
    set_field(win, 1, "vort_t")
    assert win._pane_scalars[1] is not win._pane_scalars[0], \
        "_refresh_scalars must assign a fresh array, not mutate the shared one"
    # Pane 0's array must not have been replaced (same object id).
    assert id(win._pane_scalars[0]) == pane0_id, \
        "customising a clone must not perturb pane 0's array reference"
    # And the data values must differ.
    assert not np.array_equal(win._pane_scalars[0], win._pane_scalars[1])


def test_layout_reexpand_preserves_per_pane_state(make_main_window, set_field):
    """1 → 4 → 1 → 4 keeps the per-pane customisations a re-expansion would inherit.

    The optimisation only shares scalars with fresh dataclone panes.
    Pre-existing PaneStates (panes that became visible again after a
    shrink) must still have their own slabs re-read so customisations
    survive.
    """
    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "vort_t")        # custom field on pane 1

    win._on_pane_layout(1)             # shrink — panes 1-3 hidden
    QCoreApplication.processEvents()
    # State for panes 1-3 must persist behind the scenes.
    assert win.state.panes[1].color_by == "vort_t"

    win._on_pane_layout(4)             # re-expand
    QCoreApplication.processEvents()
    assert win.state.panes[1].color_by == "vort_t", \
        "re-expanded pane 1 must keep its customised color_by"
    # And pane 1's scalars must reflect vort_t, not tas_t — i.e. it
    # took the re-read path, not the share-with-pane-0 path.
    assert not np.array_equal(win._pane_scalars[1], win._pane_scalars[0])
