import numpy as np
import tensorflow as tf
import pytest

from fusionlab.nn.pinn._geoprior_maths import compute_mv_prior_loss


class DummyLearnableMV:
    def __init__(self, mv_prior_units="strict", gamma_w=None):
        self.scaling_kwargs = {"mv_prior_units": mv_prior_units}
        self.log_mv = tf.Variable(np.nan, dtype=tf.float32)  # intentionally NaN
        self._mv_fixed = 1e-4
        self.gamma_w = gamma_w


class DummyFixedMV:
    def __init__(self, mv_prior_units="strict", mv_fixed=1e-4, gamma_w=None):
        self.scaling_kwargs = {"mv_prior_units": mv_prior_units}
        self._mv_fixed = mv_fixed
        self.gamma_w = gamma_w
        # no log_mv attribute


def make_bad_Ss(B=32, H=3, seed=123):
    tf.random.set_seed(seed)
    x = tf.random.uniform([B, H, 1], minval=1e-10, maxval=1e-2, dtype=tf.float32)
    idx = tf.constant([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], tf.int32)
    val = tf.constant([0.0, -1.0, np.nan, np.inf, 1e40], tf.float32)
    return tf.tensor_scatter_nd_update(x, idx, val)


def _assert_finite_loss_and_grads(model, Ss):
    Ss = tf.convert_to_tensor(Ss, tf.float32)

    has_log_mv = hasattr(model, "log_mv") and (getattr(model, "log_mv") is not None)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(Ss)
        if has_log_mv:
            tape.watch(model.log_mv)

        loss = compute_mv_prior_loss(
            model, Ss, alpha_disp=0.1, delta=1.0, eps=1e-12, verbose=0
        )

    gSs = tape.gradient(loss, Ss)
    gmv = tape.gradient(loss, model.log_mv) if has_log_mv else None

    tf.debugging.assert_equal(tf.rank(loss), 0, "loss must be scalar")
    tf.debugging.assert_all_finite(loss, "loss is NaN/Inf")

    assert gSs is not None, "grad(Ss) is None"
    tf.debugging.assert_all_finite(gSs, "grad(Ss) is NaN/Inf")

    if has_log_mv:
        assert gmv is not None, "grad(log_mv) is None"
        tf.debugging.assert_all_finite(gmv, "grad(log_mv) is NaN/Inf")


@pytest.mark.parametrize("mv_units", ["strict", "auto"])
def test_mv_prior_nan_safe_learnable(mv_units):
    model = DummyLearnableMV(mv_prior_units=mv_units, gamma_w=None)
    Ss = make_bad_Ss()
    _assert_finite_loss_and_grads(model, Ss)


@pytest.mark.parametrize("mv_units", ["strict", "auto"])
def test_mv_prior_nan_safe_fixed(mv_units):
    model = DummyFixedMV(mv_prior_units=mv_units, mv_fixed=1e-4, gamma_w=None)
    Ss = make_bad_Ss()
    _assert_finite_loss_and_grads(model, Ss)


@tf.function
def _run_graph(model, Ss):
    # graph-mode check
    return compute_mv_prior_loss(model, Ss, alpha_disp=0.1, delta=1.0, eps=1e-12, verbose=0)


@pytest.mark.parametrize("ModelCls", [DummyLearnableMV, DummyFixedMV])
def test_mv_prior_graph_mode(ModelCls):
    model = ModelCls(mv_prior_units="strict", gamma_w=None)
    Ss = make_bad_Ss()
    loss = _run_graph(model, Ss)
    tf.debugging.assert_all_finite(loss, "graph loss is NaN/Inf")
