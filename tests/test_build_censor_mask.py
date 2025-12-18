# -*- coding: utf-8 -*-
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


from fusionlab.utils.nat_utils import build_censor_mask


def _xb_with_dynamic(
    B=2, T=3, D=4, *,
    idx=1,
    flags=None,
    with_coords=True,
):
    """
    Build xb with dynamic_features where column idx is filled by flags.

    flags: array-like shape (B, T) giving the censor flag value per time.
    """
    dyn = np.zeros((B, T, D), dtype=np.float32)
    if flags is None:
        flags = np.zeros((B, T), dtype=np.float32)
    flags = np.asarray(flags, dtype=np.float32)
    assert flags.shape == (B, T)

    dyn[..., idx] = flags
    xb = {"dynamic_features": tf.constant(dyn, dtype=tf.float32)}
    if with_coords:
        xb["coords"] = tf.zeros((B, 3), dtype=tf.float32)
    return xb


def _xb_with_future(
    B=2, T=3, D=4, *,
    idx=1,
    flags=None,
    with_coords=True,
):
    fut = np.zeros((B, T, D), dtype=np.float32)
    if flags is None:
        flags = np.zeros((B, T), dtype=np.float32)
    flags = np.asarray(flags, dtype=np.float32)
    assert flags.shape == (B, T)

    fut[..., idx] = flags
    xb = {"future_features": tf.constant(fut, dtype=tf.float32)}
    if with_coords:
        xb["coords"] = tf.zeros((B, 3), dtype=tf.float32)
    return xb


def test_idx_none_returns_all_false():
    B, H = 3, 5
    xb = {"coords": tf.zeros((B, 3), dtype=tf.float32)}
    m = build_censor_mask(xb, H, idx=None)
    assert tuple(m.shape) == (B, H, 1)
    assert m.dtype == tf.bool
    assert not bool(tf.reduce_any(m).numpy())


def test_missing_source_key_returns_all_false():
    B, H = 2, 4
    xb = {"coords": tf.zeros((B, 3), dtype=tf.float32)}
    m = build_censor_mask(xb, H, idx=0, source="dynamic")
    assert tuple(m.shape) == (B, H, 1)
    assert not bool(tf.reduce_any(m).numpy())


def test_idx_out_of_range_returns_all_false():
    B, T, D, H = 2, 3, 2, 5
    xb = _xb_with_dynamic(B=B, T=T, D=D, idx=0, flags=np.zeros((B, T)))
    m = build_censor_mask(xb, H, idx=99, source="dynamic")
    assert tuple(m.shape) == (B, H, 1)
    assert not bool(tf.reduce_any(m).numpy())


def test_dynamic_reduce_any_broadcasts_to_horizon():
    # T=3, H=5: must still output (B,H,1) by broadcasting sample-level label
    B, T, H, idx = 2, 3, 5, 1
    flags = np.array([
        [0.1, 0.6, 0.2],  # any > 0.5 => True
        [0.1, 0.2, 0.3],  # all <= 0.5 => False
    ], dtype=np.float32)
    xb = _xb_with_dynamic(B=B, T=T, D=4, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="dynamic", reduce_time="any",
    )
    assert tuple(m.shape) == (B, H, 1)

    m_np = m.numpy().astype(bool)
    assert not m_np[0, :, 0].any()
    assert m_np[1, :, 0].all()


def test_dynamic_reduce_last_broadcasts_last_step():
    B, T, H, idx = 2, 3, 5, 1
    flags = np.array([
        [0.9, 0.9, 0.2],  # last=0.2 -> False
        [0.1, 0.2, 0.8],  # last=0.8 -> True
    ], dtype=np.float32)
    xb = _xb_with_dynamic(B=B, T=T, D=4, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="dynamic", reduce_time="last",
    )
    assert tuple(m.shape) == (B, H, 1)
    m_np = m.numpy().astype(bool)
    assert not m_np[0, :, 0].any()
    assert m_np[1, :, 0].all()


def test_dynamic_reduce_all_broadcasts_all_step_conjunction():
    B, T, H, idx = 2, 3, 4, 1
    flags = np.array([
        [0.9, 0.9, 0.2],  # not all > 0.5 -> False
        [0.6, 0.7, 0.8],  # all > 0.5 -> True
    ], dtype=np.float32)
    xb = _xb_with_dynamic(B=B, T=T, D=4, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="dynamic", reduce_time="all",
    )
    assert tuple(m.shape) == (B, H, 1)
    m_np = m.numpy().astype(bool)
    assert not m_np[0, :, 0].any()
    assert m_np[1, :, 0].all()


def test_future_align_crop_takes_last_h_steps():
    B, T, H, idx = 1, 7, 5, 1
    # flags per step: [0,0,0,1,0,1,1] with thresh=0.5 -> [F,F,F,T,F,T,T]
    flags = np.array([[0, 0, 0, 1, 0, 1, 1]], dtype=np.float32)
    xb = _xb_with_future(B=B, T=T, D=3, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="future", align="crop",
    )
    assert tuple(m.shape) == (B, H, 1)

    # last 5 are steps 2..6: [0,1,0,1,1] -> [F,T,F,T,T]
    expected = np.array([[[False], [True], [False], [True], [True]]])
    assert (m.numpy() == expected).all()


def test_future_align_pad_false_pads_front_with_false():
    B, T, H, idx = 1, 3, 5, 1
    flags = np.array([[1, 0, 1]], dtype=np.float32)  # -> [T,F,T]
    xb = _xb_with_future(B=B, T=T, D=3, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="future", align="pad_false",
    )
    assert tuple(m.shape) == (B, H, 1)

    # pad 2 falses at front then [T,F,T] -> [F,F,T,F,T]
    expected = np.array([[[False], [False], [True], [False], [True]]])
    assert (m.numpy() == expected).all()


def test_future_align_pad_edge_pads_front_with_last_value():
    B, T, H, idx = 1, 3, 5, 1
    flags = np.array([[1, 0, 1]], dtype=np.float32)  # last is True
    xb = _xb_with_future(B=B, T=T, D=3, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="future", align="pad_edge",
    )
    assert tuple(m.shape) == (B, H, 1)

    # pad 2 with last=True then [T,F,T] -> [T,T,T,F,T]
    expected = np.array([[[True], [True], [True], [False], [True]]])
    assert (m.numpy() == expected).all()


def test_future_align_broadcast_repeats_last_step_when_mismatch():
    B, T, H, idx = 2, 3, 5, 1
    flags = np.array([
        [0.0, 1.0, 0.0],  # last=0.0 -> False
        [0.0, 0.0, 1.0],  # last=1.0 -> True
    ], dtype=np.float32)
    xb = _xb_with_future(B=B, T=T, D=4, idx=idx, flags=flags)

    m = build_censor_mask(
        xb, H, idx=idx, thresh=0.5,
        source="future", align="broadcast",
    )
    assert tuple(m.shape) == (B, H, 1)
    m_np = m.numpy().astype(bool)
    assert m_np[0, :, 0].all()
    assert not m_np[1, :, 0].any()


def test_future_align_error_raises_on_mismatch():
    B, T, H, idx = 1, 3, 5, 1
    xb = _xb_with_future(B=B, T=T, D=3, idx=idx, flags=np.zeros((B, T)))

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = build_censor_mask(
            xb, H, idx=idx, thresh=0.5,
            source="future", align="error",
        ).numpy()


def test_batch_size_resolves_without_coords():
    # coords missing -> B resolved from dynamic_features
    B, T, H, idx = 4, 3, 5, 1
    xb = _xb_with_dynamic(B=B, T=T, D=4, idx=idx, flags=np.zeros((B, T)), with_coords=False)
    m = build_censor_mask(xb, H, idx=idx, source="dynamic", reduce_time="any")
    assert tuple(m.shape) == (B, H, 1)
