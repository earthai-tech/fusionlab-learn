# test_infer_quantile_axis_ambiguity.py
import numpy as np
import tensorflow as tf


def _infer_quantile_axis(t, n_q: int = 3):
    """Infer Q axis (static, conservative)."""
    shape = getattr(t, "shape", None)
    rank = getattr(shape, "rank", None)

    if rank is None:
        return None

    # Canonical layouts used in FusionLab:
    #   (B,H,Q,O) -> axis=2
    #   (B,H,O,Q) -> axis=3
    if rank == 4:
        if shape[2] == n_q:
            return 2
        elif shape[3] == n_q:
            return 3
        return None

    # Rank-3: only accept (B,H,Q) on last axis.
    if rank == 3:
        return 2 if shape[2] == n_q else None

    return None


def coverage_sharpness_using_infer(y_true_bh1, pred, n_q=3):
    """
    Compute coverage/sharpness using _infer_quantile_axis() logic.
    This mimics what many metric fns effectively do when they don't
    canonicalize first.
    """
    q_axis = _infer_quantile_axis(pred, n_q=n_q)
    if q_axis is None:
        raise RuntimeError("Could not infer quantile axis.")

    # Extract q10 and q90 along inferred axis:
    if pred.shape.rank == 4:
        # Gather along q_axis
        q10 = tf.gather(pred, 0, axis=q_axis)
        q90 = tf.gather(pred, n_q - 1, axis=q_axis)
        # q10/q90 now rank-3, ensure last dim is O=1
        # They will broadcast with y_true if H==Q (this is the trap).
    else:
        raise RuntimeError("Expected rank-4 pred for this demo.")

    # coverage: mean over all elements
    hit = tf.cast((y_true_bh1 >= q10) & (y_true_bh1 <= q90), tf.float32)
    cov = float(tf.reduce_mean(hit).numpy())

    # sharpness: mean interval width
    sharp = float(tf.reduce_mean(q90 - q10).numpy())
    return q_axis, cov, sharp


def make_synthetic(B=2000, H=3, Q=3, O=1, seed=0):
    rng = np.random.default_rng(seed)

    # Make y_true vary by horizon strongly (to expose wrong-axis bugs)
    base = rng.normal(size=(B, 1, 1)).astype(np.float32)
    trend = np.array([0.0, 10.0, 20.0], dtype=np.float32).reshape(1, H, 1)
    y_true = base + trend  # (B,H,1)

    # Build "true" quantiles around y_true (guaranteed high coverage)
    q10 = y_true - 2.0
    q50 = y_true + 0.0
    q90 = y_true + 2.0
    s_q_bhqo = np.stack([q10, q50, q90], axis=2)  # (B,H,Q,1)

    # Same values but stored as (B,Q,H,1)
    s_q_bqho = np.transpose(s_q_bhqo, (0, 2, 1, 3))

    return (
        tf.convert_to_tensor(y_true, tf.float32),
        tf.convert_to_tensor(s_q_bhqo, tf.float32),
        tf.convert_to_tensor(s_q_bqho, tf.float32),
    )


if __name__ == "__main__":
    y_true, pred_bhqo, pred_bqho = make_synthetic(B=4000, H=3, Q=3, O=1)

    print("y_true:", y_true.shape)
    print("pred_bhqo:", pred_bhqo.shape, "(B,H,Q,O)")
    print("pred_bqho:", pred_bqho.shape, "(B,Q,H,O)")

    ax, cov, sharp = coverage_sharpness_using_infer(y_true, pred_bhqo, n_q=3)
    print("\n[BHQO input]")
    print("  inferred q_axis:", ax)
    print("  coverage:", cov)
    print("  sharpness:", sharp)

    ax, cov, sharp = coverage_sharpness_using_infer(y_true, pred_bqho, n_q=3)
    print("\n[BQHO input]")
    print("  inferred q_axis:", ax, "(WRONG but will still return 2)")
    print("  coverage:", cov, "(should collapse vs BHQO)")
    print("  sharpness:", sharp)

    # Extra: show that ambiguity disappears when Q != H
    print("\nNow show Q!=H avoids the trap (H=3, Q=5):")
    B, H, Q, O = 2000, 3, 5, 1
    rng = np.random.default_rng(0)
    y = (rng.normal(size=(B, H, 1)) + np.array([0, 10, 20]).reshape(1, H, 1)).astype(np.float32)
    # q10/q50/q90 not enough anymore; build 5 quantiles
    deltas = np.array([-2, -1, 0, 1, 2], dtype=np.float32).reshape(1, 1, Q, 1)
    pred = y[:, :, None, :] + deltas
    y = tf.convert_to_tensor(y, tf.float32)
    pred = tf.convert_to_tensor(pred, tf.float32)

    ax = _infer_quantile_axis(pred, n_q=5)
    print("  inferred axis for (B,H,Q,O) with Q=5:", ax)
