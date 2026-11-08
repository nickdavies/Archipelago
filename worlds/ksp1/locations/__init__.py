"""KSP1 location model.

Three lifetimes, each a subset of the one before:

* ``catalog.LocationCatalog`` — static: every location that can ever exist, in
  id order.  The AP data package, the Universal Tracker map and the CLI.
* ``configured.ConfiguredLocations`` — what a world emits as a pure function of
  its resolved settings (home, slot counts, the feasibility table).
* ``seed.SeedLocations`` — the final emitted set: ``ConfiguredLocations`` plus
  what generation produced (contract rewards, goal-mode thresholds).

Locations are typed defs (``defs.py``); their display name is ``str(defn)`` and
only matters at the AP registry and client boundaries.  ``events.py`` holds the
per-body mission-event vocabulary, ``descriptor.py`` the physics descriptor the
sphere ladder reads.

Everything except ``ap.py`` is Archipelago-free (enforced by
``test_location_model``).  This package deliberately re-exports nothing:
re-exporting the catalog would import ``contracts`` here, recreating the import
cycle for modules (like ``contracts``) that only need ``events`` or ``defs``.
"""
