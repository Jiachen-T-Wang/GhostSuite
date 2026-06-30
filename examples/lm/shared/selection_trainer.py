"""Backward-compatible re-export of the online-selection driver.

The driver moved into the engine package (``ghostEngines/selection_driver.py``) so the
ghost choreography lives next to the engine instead of in the examples tree. Existing
imports — ``from shared.selection_trainer import online_selection_step`` — keep working
through this shim. New code should import from ``ghostEngines`` directly.
"""

from ghostEngines.selection_driver import (
    online_selection_step,
    _plain_update,
    _clip_step,
    _set_ddp_sync,
)

__all__ = ["online_selection_step", "_plain_update", "_clip_step", "_set_ddp_sync"]
