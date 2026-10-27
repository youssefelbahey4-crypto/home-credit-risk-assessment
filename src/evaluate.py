"""Cost-matrix threshold selection matching main01's financial scenario."""
import numpy as np


def find_optimal_business_threshold(
    y_val, probs, gain_tn=1500, loss_fp=-300, loss_fn=-4000,
    gain_tp=0, thresholds=None,
):
    """Return (best threshold, validation value) under main01's cost matrix.

    Threshold convention matches the notebook: probability >= threshold means
    predicted default / reject; true negatives are good approvals.
    """
    y = np.asarray(y_val).reshape(-1)
    p = np.asarray(probs).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("y_val and probs must have the same number of observations")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_val must contain only binary labels 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probs must be finite probabilities in [0, 1]")

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 100)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a non-empty finite sequence")

    profits = []
    for threshold in thresholds:
        predicted_default = p >= threshold
        tn = np.sum((y == 0) & ~predicted_default)
        fp = np.sum((y == 0) & predicted_default)
        fn = np.sum((y == 1) & ~predicted_default)
        tp = np.sum((y == 1) & predicted_default)
        profits.append(tn*gain_tn + fp*loss_fp + fn*loss_fn + tp*gain_tp)
    best = int(np.argmax(profits))
    return float(thresholds[best]), float(profits[best])
