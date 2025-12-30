import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_maths import (
    compute_consolidation_step_residual,
    dt_to_seconds,  
)

# ----------------------------
# Small helpers
# ----------------------------
def _assert_finite(x: tf.Tensor, *, name: str):
    ok = tf.reduce_all(tf.math.is_finite(x))
    if bool(ok.numpy()):
        return
    mask = ~tf.math.is_finite(x)
    n_bad = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
    bad_idx = tf.where(mask)[:10].numpy().tolist()
    x_np = x.numpy()
    vals = [float(x_np[tuple(i)]) for i in bad_idx]
    raise AssertionError(
        f"{name} has {n_bad} non-finite values. "
        f"sample idx={bad_idx}, sample vals={vals}"
    )


def _mk_base(B: int, T: int):
    """
    Constant case:
      h_ref - h_mean = 2 m
      Ss = 2e-4 1/m
      H  = 50 m
      => s_eq = Ss * H * 2 = 0.02 m
    """
    h_mean = tf.ones((B, T, 1), tf.float32) * 10.0
    h_ref  = tf.ones((B, T, 1), tf.float32) * 12.0
    Ss     = tf.ones((B, T, 1), tf.float32) * 2e-4
    Hf     = tf.ones((B, T, 1), tf.float32) * 50.0

    # tau = 2 * 365 days in seconds (matches your earlier tests)
    tau_sec = tf.ones((B, T - 1, 1), tf.float32) * (2.0 * 365.0 * 24.0 * 3600.0)

    # dt = 1 year per step (T-1 steps)
    dt_yr = tf.ones((B, T - 1, 1), tf.float32) * 1.0

    return dict(h_mean=h_mean, h_ref=h_ref, Ss=Ss, Hf=Hf, tau=tau_sec, dt=dt_yr)


def _build_consistent_state_exact(
    *,
    B: int,
    T: int,
    s0: float,
    s_eq: float,
    tau_sec: float,
    dt_yr: np.ndarray,
    time_units: str = "yr",
):
    """
    Build s_state that exactly follows the SAME "exact" step used in code:
      s_{n+1} = s_n * a + s_eq * (1-a)
      a = exp(-dt_sec/tau_sec)
    Returns: (B,T,1) float32 tensor.
    """
    dt_tf = tf.constant(dt_yr.reshape(1, T - 1, 1), tf.float32)
    dt_sec = dt_to_seconds(dt_tf, time_units=time_units).numpy().reshape(T - 1)

    s = np.zeros((T,), dtype=np.float64)
    s[0] = s0
    for n in range(T - 1):
        a = np.exp(-dt_sec[n] / tau_sec)
        s[n + 1] = s[n] * a + s_eq * (1.0 - a)

    s_state = np.repeat(s[None, :, None], B, axis=0).astype(np.float32)
    return tf.constant(s_state)


# ----------------------------
# Core correctness tests
# ----------------------------
def test_step_residual_shape_and_finite():
    B, T = 3, 6
    d = _mk_base(B, T)

    # Make a consistent s_state (constant s_eq)
    s_eq = 0.02
    s_state = _build_consistent_state_exact(
        B=B, T=T, s0=0.0, s_eq=s_eq,
        tau_sec=float(d["tau"][0, 0, 0].numpy()),
        dt_yr=np.ones((T - 1,), dtype=np.float32),
        time_units="yr",
    )

    out = compute_consolidation_step_residual(
        s_state_si=s_state,
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        dt=d["dt"],
        time_units="yr",
        method="exact",
        verbose=1,
    )
    assert out.shape == (B, T - 1, 1)
    _assert_finite(out, name="residual")
    np.testing.assert_allclose(out.numpy(), 0.0, atol=2e-6, rtol=0.0)


def test_step_residual_zero_for_variable_dt_exact():
    B, T = 2, 8
    d = _mk_base(B, T)

    # variable dt per step
    dt_yr = np.array([0.25, 0.5, 1.0, 0.1, 2.0, 0.75, 0.5], dtype=np.float32)
    assert dt_yr.shape[0] == T - 1

    # rebuild dt tensor with variable dt
    dt_tf = tf.constant(np.repeat(dt_yr[None, :, None], B, axis=0), tf.float32)

    s_eq = 0.02
    s_state = _build_consistent_state_exact(
        B=B, T=T, s0=0.0, s_eq=s_eq,
        tau_sec=float(d["tau"][0, 0, 0].numpy()),
        dt_yr=dt_yr,
        time_units="yr",
    )

    out = compute_consolidation_step_residual(
        s_state_si=s_state,
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        dt=dt_tf,
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(out, name="residual_var_dt")
    np.testing.assert_allclose(out.numpy(), 0.0, atol=3e-6, rtol=0.0)


def test_step_residual_localizes_perturbation():
    B, T = 2, 7
    d = _mk_base(B, T)

    dt_yr = np.ones((T - 1,), dtype=np.float32)
    s_eq = 0.02
    s_state = _build_consistent_state_exact(
        B=B, T=T, s0=0.0, s_eq=s_eq,
        tau_sec=float(d["tau"][0, 0, 0].numpy()),
        dt_yr=dt_yr,
        time_units="yr",
    ).numpy()

    # inject a perturbation at time index k (affects residual at step k-1 -> k)
    k = 4
    s_state[:, k, 0] += 1e-3  # 1 mm "jump"
    s_state_tf = tf.constant(s_state, tf.float32)

    res = compute_consolidation_step_residual(
        s_state_si=s_state_tf,
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        dt=d["dt"],
        time_units="yr",
        method="exact",
        verbose=0,
    ).numpy()[:, :, 0]

    # Only the residual at step (k-1) should show the injected jump strongly.
    # Later steps will also differ slightly because the state is changed, but the
    # biggest spike should be at k-1.
    spike = np.abs(res[:, k - 1])
    assert np.all(spike > 5e-4)

    # earlier residuals should remain ~0
    if k - 2 >= 0:
        np.testing.assert_allclose(res[:, :k - 1], 0.0, atol=3e-6, rtol=0.0)


# ----------------------------
# Broadcast / shape robustness
# ----------------------------
@pytest.mark.parametrize("dt_shape_kind", ["full", "BH1", "1H1", "scalar"])
@pytest.mark.parametrize("tau_shape_kind", ["full", "BH1", "1H1", "scalar"])
def test_step_residual_broadcasting_dt_tau(dt_shape_kind, tau_shape_kind):
    B, T = 3, 6
    d = _mk_base(B, T)

    # build consistent state
    s_eq = 0.02
    s_state = _build_consistent_state_exact(
        B=B, T=T, s0=0.0, s_eq=s_eq,
        tau_sec=float(d["tau"][0, 0, 0].numpy()),
        dt_yr=np.ones((T - 1,), dtype=np.float32),
        time_units="yr",
    )

    H = T - 1

    def make_dt(kind):
        base = 1.0
        if kind == "full":
            return tf.ones((B, H, 1), tf.float32) * base
        if kind == "BH1":
            return tf.ones((B, 1, 1), tf.float32) * base
        if kind == "1H1":
            return tf.ones((1, H, 1), tf.float32) * base
        return tf.constant(base, tf.float32)

    def make_tau(kind):
        base = float(d["tau"][0, 0, 0].numpy())
        if kind == "full":
            return tf.ones((B, H, 1), tf.float32) * base
        if kind == "BH1":
            return tf.ones((B, 1, 1), tf.float32) * base
        if kind == "1H1":
            return tf.ones((1, H, 1), tf.float32) * base
        return tf.constant(base, tf.float32)

    dt = make_dt(dt_shape_kind)
    tau = make_tau(tau_shape_kind)

    out = compute_consolidation_step_residual(
        s_state_si=s_state,
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau,
        h_ref_si=d["h_ref"],
        dt=dt,
        time_units="yr",
        method="exact",
        verbose=0,
    )
    assert out.shape == (B, H, 1)
    _assert_finite(out, name=f"res(dt={dt_shape_kind},tau={tau_shape_kind})")


# ----------------------------
# Method validation
# ----------------------------
def test_step_residual_invalid_method_raises():
    B, T = 2, 5
    d = _mk_base(B, T)
    s_state = tf.zeros((B, T, 1), tf.float32)

    with pytest.raises(ValueError):
        compute_consolidation_step_residual(
            s_state_si=s_state,
            h_mean_si=d["h_mean"],
            Ss_field=d["Ss"],
            H_field_si=d["Hf"],
            tau_field=d["tau"],
            h_ref_si=d["h_ref"],
            dt=d["dt"],
            time_units="yr",
            method="nope",
            verbose=0,
        )


# ----------------------------
# Gradient behavior wrt h_ref
# ----------------------------
def test_step_residual_stop_grad_ref_behavior():
    B, T = 2, 6
    d = _mk_base(B, T)

    # consistent-ish s_state (doesn't need to be perfect)
    s_state = tf.zeros((B, T, 1), tf.float32)

    # stop_grad_ref=True -> gradient should be None OR all zeros
    h_ref = tf.Variable(d["h_ref"])
    with tf.GradientTape() as tape:
        res = compute_consolidation_step_residual(
            s_state_si=s_state,
            h_mean_si=d["h_mean"],
            Ss_field=d["Ss"],
            H_field_si=d["Hf"],
            tau_field=d["tau"],
            h_ref_si=h_ref,
            dt=d["dt"],
            time_units="yr",
            method="exact",
            stop_grad_ref=True,
            verbose=0,
        )
        loss = tf.reduce_sum(tf.square(res))

    g = tape.gradient(loss, h_ref)
    if g is not None:
        np.testing.assert_allclose(g.numpy(), 0.0, atol=0.0, rtol=0.0)

    # stop_grad_ref=False -> gradient should exist and be nonzero (in this setup)
    h_ref2 = tf.Variable(d["h_ref"])
    with tf.GradientTape() as tape2:
        res2 = compute_consolidation_step_residual(
            s_state_si=s_state,
            h_mean_si=d["h_mean"],
            Ss_field=d["Ss"],
            H_field_si=d["Hf"],
            tau_field=d["tau"],
            h_ref_si=h_ref2,
            dt=d["dt"],
            time_units="yr",
            method="exact",
            stop_grad_ref=False,
            verbose=0,
        )
        loss2 = tf.reduce_sum(tf.square(res2))

    g2 = tape2.gradient(loss2, h_ref2)
    assert g2 is not None
    assert np.any(np.abs(g2.numpy()) > 1e-10)
