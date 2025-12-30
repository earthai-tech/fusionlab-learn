import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_maths import ( 
    dt_to_seconds, 
    integrate_consolidation_mean
)

def _mk_base(B, H):
    # Simple constant setup: drawdown = 2m, Ss=2e-4 1/m, H=50m -> s_eq=0.02m
    h_mean = tf.ones((B, H, 1), tf.float32) * 10.0
    h_ref  = tf.ones((B, H, 1), tf.float32) * 12.0
    Ss     = tf.ones((B, H, 1), tf.float32) * 2e-4
    Hf     = tf.ones((B, H, 1), tf.float32) * 50.0

    # tau: choose 2 years in seconds (approx; good enough for unit tests)
    tau_sec = tf.ones((B, H, 1), tf.float32) * (2.0 * 365.0 * 24.0 * 3600.0)

    # dt: 1 year per step
    dt_yr = tf.ones((B, H, 1), tf.float32) * 1.0

    # initial settlement
    s0 = tf.zeros((B, 1, 1), tf.float32)
    return dict(h_mean=h_mean, h_ref=h_ref, Ss=Ss, Hf=Hf, tau=tau_sec, dt=dt_yr, s0=s0)

def _mk_bad(shape, kind, *, base_value, seed):
    rng = np.random.default_rng(seed)

    if shape == ():
        if kind == "nan":
            return tf.constant(np.nan, tf.float32)
        if kind == "inf":
            return tf.constant(np.inf, tf.float32)
        if kind == "mix":
            return tf.constant(np.nan, tf.float32)  # scalar "mix" -> nan is enough

    x = np.ones(shape, dtype=np.float32) * np.float32(base_value)
    flat = x.reshape(-1)

    n = flat.size
    k = min(6, n)
    idx = rng.choice(n, size=k, replace=False)

    if kind == "nan":
        flat[idx] = np.nan
    elif kind == "inf":
        flat[idx] = np.inf
    elif kind == "mix":
        half = k // 2
        flat[idx[:half]] = np.nan
        flat[idx[half:]] = np.inf
    else:
        raise ValueError(kind)

    return tf.constant(x)


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

def test_integrate_exact_matches_closed_form_constant_case():
    B, H = 3, 5
    d = _mk_base(B, H)

    out = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        s_init_si=d["s0"],
        dt=d["dt"],
        time_units="yr",
        method="exact",
        verbose=1,  # keep verbose path exercised (pytest captures output by default)
    )
    assert out.shape == (B, H, 1)
    _assert_finite(out, name="s_bar_exact")

    # Expected: s_eq = 0.02; tau=2yr; dt=1yr
    s_eq = 0.02
    
    dt_sec = float(dt_to_seconds(
        tf.constant(1.0, tf.float32), time_units="yr").numpy()
    )
    tau_sec = float(d["tau"][0, 0, 0].numpy())
    a = np.exp(-dt_sec / tau_sec) # exp(-dt/tau) with dt=1yr, tau=2yr
    expected = np.array([s_eq * (1.0 - a ** (k + 1)) for k in range(H)], dtype=np.float32)
    got = out.numpy()[:, :, 0]
    
    expected_BH = np.repeat(expected[None, :], B, axis=0)
    
    # same expected for each batch row
    np.testing.assert_allclose(got, expected_BH, rtol=1e-5, atol=1e-6)


def test_integrate_euler_close_to_exact_when_dt_small():
    B, H = 2, 8
    d = _mk_base(B, H)

    # Make tau huge so dt/tau is tiny -> Euler ~ exact
    tau_big = tf.ones((B, H, 1), tf.float32) * (200.0 * 365.0 * 24.0 * 3600.0)
    dt_yr = tf.ones((B, H, 1), tf.float32) * 0.25  # quarter-year steps
    out_exact = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau_big,
        h_ref_si=d["h_ref"],
        s_init_si=d["s0"],
        dt=dt_yr,
        time_units="yr",
        method="exact",
        verbose=0,
    )
    out_euler = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau_big,
        h_ref_si=d["h_ref"],
        s_init_si=d["s0"],
        dt=dt_yr,
        time_units="yr",
        method="euler",
        verbose=0,
    )
    _assert_finite(out_exact, name="exact_small_dt")
    _assert_finite(out_euler, name="euler_small_dt")

    np.testing.assert_allclose(out_euler.numpy(), out_exact.numpy(), rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
@pytest.mark.parametrize("shape_kind", ["full", "BH1", "1H1", "scalar"])
@pytest.mark.parametrize("field", ["tau", "dt", "s0"])
def test_integrate_bad_inputs_are_finite(field, bad_kind, shape_kind):
    B, H = 4, 6
    d = _mk_base(B, H)

    if field == "tau":
        base = float(2.0 * 365.0 * 24.0 * 3600.0)
        seed = 11
    elif field == "dt":
        base = 1.0
        seed = 12
    else:  # s0
        base = 0.0
        seed = 13

    if shape_kind == "full":
        shape = (B, H, 1) if field != "s0" else (B, 1, 1)
    elif shape_kind == "BH1":
        shape = (B, 1, 1) if field != "s0" else (B, 1, 1)
    elif shape_kind == "1H1":
        shape = (1, H, 1) if field != "s0" else (1, 1, 1)
    else:
        shape = ()

    bad = _mk_bad(shape, bad_kind, base_value=base, seed=seed)

    tau = bad if field == "tau" else d["tau"]
    dt  = bad if field == "dt" else d["dt"]
    s0  = bad if field == "s0" else d["s0"]

    out = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau,
        h_ref_si=d["h_ref"],
        s_init_si=s0,
        dt=dt,
        time_units="yr",
        method="exact",
        verbose=1,
    )
    assert out.shape == (B, H, 1)
    _assert_finite(out, name=f"s_bar({field},{bad_kind},{shape_kind})")


def test_integrate_stop_grad_ref_zero_grad():
    # Checks that stop_grad_ref=True actually kills gradients wrt h_ref
    B, H = 2, 4
    d = _mk_base(B, H)

    h_ref = tf.Variable(d["h_ref"])
    with tf.GradientTape() as tape:
        out = integrate_consolidation_mean(
            h_mean_si=d["h_mean"],
            Ss_field=d["Ss"],
            H_field_si=d["Hf"],
            tau_field=d["tau"],
            h_ref_si=h_ref,
            s_init_si=d["s0"],
            dt=d["dt"],
            time_units="yr",
            method="exact",
            stop_grad_ref=True,
            verbose=0,
        )
        loss = tf.reduce_sum(out)

    g = tape.gradient(loss, h_ref)
    if g is None:
        return
    np.testing.assert_allclose(g.numpy(), 0.0, atol=0.0, rtol=0.0)

    # gradients should be (close to) zero everywhere
    np.testing.assert_allclose(g.numpy(), 0.0, atol=0.0, rtol=0.0)
