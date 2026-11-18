"""Bias-corrected failure prevalence for a monitoring period."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def corrected_mode_prevalence(
    sample_preds: Sequence[int],
    test_labels: Sequence[int],
    test_preds: Sequence[int],
    confidence: float = 0.95,
    bootstrap_iterations: int = 20000,
    seed: int | None = 7,
) -> dict[str, Any]:
    """Bias-corrected live prevalence for one mode from sampled verdicts.

    The contract, precisely:

      1. ``raw`` is the uncorrected flag rate: ``mean(sample_preds)``.
      2. Compute the frozen judge's failure sensitivity and pass specificity
         from ``test_labels`` and ``test_preds``. Both use the monitoring
         convention that 1 means a failure is present. Failure sensitivity is
         the flagged fraction of human-labeled failures. Pass specificity is
         the unflagged fraction of human-labeled passes.
      3. Compute the Rogan-Gladen point estimate, then resample the held-out
         records and sampled predictions to obtain a percentile-bootstrap
         interval. Use a seeded NumPy generator so the committed result is
         reproducible.
      4. Resample the monitoring predictions and the paired held-out records
         independently with replacement. Keep their original sample sizes.
         Discard a draw if the correction cannot be computed. Clamp each
         retained estimate to [0, 1], then take the percentile interval.
         Raise ``ValueError`` if no replicate is valid.

    Args:
        sample_preds: the judge's 0/1 verdicts over the UNIFORM BASE sample
            only (never the risk strata; they are biased toward failure by
            design).
        test_labels: human labels for the frozen Homework 5 judge's test
            split.
        test_preds: the frozen judge's predictions on that test split.
        confidence: interval confidence level.
        bootstrap_iterations: number of percentile-bootstrap replicates.
        seed: numpy seed for a reproducible interval; None leaves the RNG
            untouched.

    Returns:
        {"raw", "corrected", "ci_low", "ci_high", "confidence",
         "failure_sensitivity", "pass_specificity", "n_sample"}
        with "corrected" clamped to [0, 1] and rates rounded to 4 places.

    Raises:
        ValueError: if an input is empty, the held-out inputs have different
            lengths, a value is not 0 or 1, a class is absent, the judge is
            missing a usable correction, or no bootstrap replicate is valid.
    """
    sample = np.asarray(sample_preds, dtype=int)
    labels = np.asarray(test_labels, dtype=int)
    preds = np.asarray(test_preds, dtype=int)
    if sample.size == 0 or labels.size == 0:
        raise ValueError("sample and held-out inputs must be non-empty")
    if labels.size != preds.size:
        raise ValueError("test_labels and test_preds have different lengths")
    if any(not np.isin(a, (0, 1)).all() for a in (sample, labels, preds)):
        raise ValueError("every verdict and label must be 0 or 1")
    if not (labels == 1).any() or not (labels == 0).any():
        raise ValueError("the held-out labels need both failures and passes")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    se = float(preds[labels == 1].mean())
    sp = float(1 - preds[labels == 0].mean())
    j = se + sp - 1
    if j <= 0:
        raise ValueError(f"no usable correction: sensitivity + specificity - 1 = {j:.4f} <= 0")
    raw = float(sample.mean())
    corrected = min(1.0, max(0.0, (raw + sp - 1) / j))

    # Percentile bootstrap: both sources of uncertainty, resampled independently.
    rng = np.random.default_rng(seed)
    b = bootstrap_iterations
    raw_b = sample[rng.integers(0, sample.size, size=(b, sample.size))].mean(axis=1)
    idx = rng.integers(0, labels.size, size=(b, labels.size))
    yb, pb = labels[idx], preds[idx]
    n_fail, n_pass = (yb == 1).sum(axis=1), (yb == 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_b = ((yb == 1) & (pb == 1)).sum(axis=1) / n_fail
        sp_b = ((yb == 0) & (pb == 0)).sum(axis=1) / n_pass
        j_b = se_b + sp_b - 1
        valid = (n_fail > 0) & (n_pass > 0) & (j_b > 0)
        if not valid.any():
            raise ValueError("no bootstrap replicate has a usable correction")
        est = np.clip((raw_b[valid] + sp_b[valid] - 1) / j_b[valid], 0.0, 1.0)
    alpha = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(est, [alpha, 1 - alpha])

    return {
        "raw": round(raw, 4),
        "corrected": round(corrected, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "confidence": confidence,
        "failure_sensitivity": round(se, 4),
        "pass_specificity": round(sp, 4),
        "n_sample": int(sample.size),
    }
