# tests/test_tuner_progress.py
# ---------------------------------------------------------------------
#  Unit tests for fusionlab.tools.app.utils.TunerProgressCallback
# ---------------------------------------------------------------------
from __future__ import annotations 
import copy
import pytest

from fusionlab.tools.app.utils import TunerProgressCallback


# Stubs
class DummyProgressManager:
    """Minimal stand-in that just records what was asked of it."""

    def __init__(self):
        self.updates = []        # (current, total)
        self.trial_ctx = []      # (trial, total)
        self.epoch_ctx = []      # (epoch, total)

    # called by callback
    def update(self, current, total):
        self.updates.append((current, total))

    def set_trial_context(self, *, trial, total):
        self.trial_ctx.append((trial, total))

    def set_epoch_context(self, *, epoch, total):
        self.epoch_ctx.append((epoch, total))


class DummyTrial:
    """Mimics Keras-Tuner's Trial object (just the trial_id)."""

    def __init__(self, trial_id: str | int):
        self.trial_id = str(trial_id)


# Tests
def test_epoch_level_progress():
    """Progress should reach 100 % after all epochs of all trials."""
    pm = DummyProgressManager()
    cb = TunerProgressCallback(
        total_trials=2,
        total_epochs=3,
        batches_per_epoch=2,
        progress_manager=pm,
        epoch_level=True,
        trial_batch_level=False,
        log=lambda *_: None,           # silence output
    )

    # ── simulate tuner run (epoch-level updates) -
    for t in range(2):
        trial = DummyTrial(t)
        cb.on_trial_begin(trial)
        for e in range(3):
            cb.on_epoch_end(e)
        cb.on_trial_end(trial)

    # last call must report full completion (current == total == 12)
    assert pm.updates[-1] == (12, 12)
    # should have recorded one trial context per trial
    assert pm.trial_ctx == [(1, 2), (2, 2)]
    # 3 epochs × 2 trials recorded (order not important for this assert)
    assert len(pm.epoch_ctx) == 6


def test_batch_level_progress():
    """With trial_batch_level=True the bar advances every batch."""
    pm = DummyProgressManager()
    cb = TunerProgressCallback(
        total_trials=1,
        total_epochs=2,
        batches_per_epoch=3,
        progress_manager=pm,
        epoch_level=False,
        trial_batch_level=True,
        log=lambda *_: None,
    )

    trial = DummyTrial(0)
    cb.on_trial_begin(trial)
    for b in range(6):          # 2 epochs × 3 batches
        cb.on_batch_end(b)
    cb.on_trial_end(trial)

    # Progress should reach 6/6
    assert pm.updates[-1] == (6, 6)
    # Six batch-level updates expected
    expected_updates = 2 + 6
    assert len(pm.updates) == expected_updates          # +1 from on_trial_end


def test_deepcopy_returns_self():
    """Deep-copying must return the same instance (Qt signals can’t copy)."""
    pm = DummyProgressManager()
    cb = TunerProgressCallback(1, 1, 1, pm, log=lambda *_: None)
    assert copy.deepcopy(cb) is cb


def test_invalid_args_raise():
    """Non-positive totals should raise ValueError."""
    with pytest.raises(ValueError):
        TunerProgressCallback(0, 1, 1, DummyProgressManager())

if __name__ =='__main__': 
    pytest.main ([__file__])