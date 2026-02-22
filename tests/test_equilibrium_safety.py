# tests/test_equilibrium_safety.py
import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_maths import equilibrium_compaction_si


def _mk_base(B, H):
    h_mean = tf.ones((B, H, 1), tf.float32) * 10.0
    h_ref  = tf.ones((B, H, 1), tf.float32) * 12.0
    Ss     = tf.ones((B, H, 1), tf.float32) * 2e-4
    Hf     = tf.ones((B, H, 1), tf.float32) * 50.0
    return dict(h_mean=h_mean, h_ref=h_ref, Ss=Ss, Hf=Hf)


def _inject_bad(arr: np.ndarray, kind: str, n_bad: int = 6, seed: int = 0) -> np.ndarray:
    """Inject NaN/Inf into random locations of arr (in-place)."""
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1)
    idx = rng.choice(flat.size, size=min(n_bad, flat.size), replace=False)

    kind = kind.lower()
    if kind == "nan":
        flat[idx] = np.nan
    elif kind == "inf":
        flat[idx] = np.inf
    elif kind == "mix":
        # half nan, half inf
        half = len(idx) // 2
        flat[idx[:half]] = np.nan
        flat[idx[half:]] = np.inf
    else:
        raise ValueError("kind must be {'nan','inf','mix'}")
    return arr


def _mk_bad(shape, kind: str, base_value: float = 1.0, seed: int = 0):
    x = np.full(shape, base_value, dtype=np.float32)
    x = _inject_bad(x, kind=kind, n_bad=6, seed=seed)
    return tf.constant(x)


def _assert_finite(x: tf.Tensor, *, name: str = "tensor") -> None:
    ok = tf.reduce_all(tf.math.is_finite(x))
    if bool(ok.numpy()):
        return

    mask = ~tf.math.is_finite(x)
    n_bad = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
    bad_idx = tf.where(mask)
    # show up to 10 bad entries
    sample = bad_idx[:10].numpy().tolist()

    # grab some bad values (safe gather)
    x_np = x.numpy()
    vals = []
    for ijk in sample:
        vals.append(float(x_np[tuple(ijk)]))

    raise AssertionError(
        f"{name} has {n_bad} non-finite values. "
        f"Sample indices={sample}, sample values={vals}"
    )


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_eq_bad_h_mean_is_finite(bad_kind):
    B, H = 4, 6
    d = _mk_base(B, H)

    # Try 2D and 3D forms (broadcast/shape robustness)
    h_bad_3d = _mk_bad((B, H, 1), bad_kind, base_value=10.0, seed=0)
    h_bad_2d = tf.squeeze(h_bad_3d, axis=-1)  # (B,H)

    out3 = equilibrium_compaction_si(
        h_mean_si=h_bad_3d, h_ref_si=d["h_ref"], Ss_field=d["Ss"], H_field_si=d["Hf"]
    )
    out2 = equilibrium_compaction_si(
        h_mean_si=h_bad_2d, h_ref_si=d["h_ref"], Ss_field=d["Ss"], H_field_si=d["Hf"]
    )

    assert out3.shape == (B, H, 1)
    assert out2.shape == (B, H, 1)
    _assert_finite(out3, name=f"s_eq(h_mean,{bad_kind}) 3d")
    _assert_finite(out2, name=f"s_eq(h_mean,{bad_kind}) 2d")


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
@pytest.mark.parametrize("shape_kind", ["full", "BH1", "1H1", "scalar"])
def test_eq_bad_Ss_is_finite(bad_kind, shape_kind):
    B, H = 4, 6
    d = _mk_base(B, H)

    if shape_kind == "full":
        Ss_bad = _mk_bad((B, H, 1), bad_kind, base_value=2e-4, seed=1)
    elif shape_kind == "BH1":
        Ss_bad = _mk_bad((B, 1, 1), bad_kind, base_value=2e-4, seed=2)
    elif shape_kind == "1H1":
        Ss_bad = _mk_bad((1, H, 1), bad_kind, base_value=2e-4, seed=3)
    else:  # scalar
        Ss_bad = _mk_bad((), bad_kind, base_value=2e-4, seed=4)

    out = equilibrium_compaction_si(
        h_mean_si=d["h_mean"], h_ref_si=d["h_ref"], Ss_field=Ss_bad, H_field_si=d["Hf"]
    )
    assert out.shape == (B, H, 1)
    _assert_finite(out, name=f"s_eq(Ss,{bad_kind},{shape_kind})")


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
@pytest.mark.parametrize("shape_kind", ["full", "BH1", "1H1", "scalar"])
def test_eq_bad_H_is_finite(bad_kind, shape_kind):
    B, H = 4, 6
    d = _mk_base(B, H)

    if shape_kind == "full":
        H_bad = _mk_bad((B, H, 1), bad_kind, base_value=50.0, seed=5)
    elif shape_kind == "BH1":
        H_bad = _mk_bad((B, 1, 1), bad_kind, base_value=50.0, seed=6)
    elif shape_kind == "1H1":
        H_bad = _mk_bad((1, H, 1), bad_kind, base_value=50.0, seed=7)
    else:  # scalar
        H_bad = _mk_bad((), bad_kind, base_value=50.0, seed=8)

    out = equilibrium_compaction_si(
        h_mean_si=d["h_mean"], h_ref_si=d["h_ref"], Ss_field=d["Ss"], H_field_si=H_bad
    )
    assert out.shape == (B, H, 1)
    _assert_finite(out, name=f"s_eq(H,{bad_kind},{shape_kind})")
