
from typing import Callable 
import math 
from fusionlab.nn import KERAS_DEPS 

Callback =KERAS_DEPS.Callback 

class GuiProgress(Callback):
    """
    Emit percentage updates while `model.fit` runs.

    Parameters
    ----------
    total_epochs : int
        Epochs you pass to `model.fit`.
    batches_per_epoch : int
        Length of the training dataset (`len(ds)`).
        Needed only for *batch-level* granularity.
    update_fn : Callable[[int], None]
        Function that receives an **int 0-100**.
        Examples: `my_qprogressbar.setValue`, `signal.emit`.
    epoch_level : bool, default=True
        If True, update once per epoch; otherwise per batch.
    """
    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_batches = 0

    # -------- epoch-level --------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_level:
            pct = int((epoch + 1) / self.total_epochs * 100)
            self.update_fn(pct)

    # -------- batch-level --------------------------------------------------
    def on_train_batch_end(self, batch, logs=None):
        if not self.epoch_level:
            self._seen_batches += 1
            total_batches = self.total_epochs * self.batches_per_epoch
            pct = math.floor(self._seen_batches / total_batches * 100)
            self.update_fn(pct)