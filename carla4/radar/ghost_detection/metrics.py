"""Dependency-light streaming binary metrics and threshold selection."""

import math

import numpy as np


class BinaryHistogramMetrics:
    def __init__(self, bins=2000):
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, probabilities, targets):
        probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        target = np.asarray(targets, dtype=np.int64).reshape(-1)
        valid = np.isfinite(probability) & np.isin(target, (0, 1))
        if not np.any(valid):
            return
        probability = np.clip(probability[valid], 0.0, 1.0)
        target = target[valid]
        indices = np.minimum(
            (probability * self.bins).astype(np.int64),
            self.bins - 1,
        )
        self.positive += np.bincount(
            indices[target == 1],
            minlength=self.bins,
        )
        self.negative += np.bincount(
            indices[target == 0],
            minlength=self.bins,
        )

    @staticmethod
    def _safe_divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    def compute(self, fixed_threshold=0.5, max_false_positive_rate=0.01):
        true_positive = np.cumsum(self.positive[::-1])
        false_positive = np.cumsum(self.negative[::-1])
        total_positive = int(self.positive.sum())
        total_negative = int(self.negative.sum())
        false_negative = total_positive - true_positive
        precision = self._safe_divide(
            true_positive,
            true_positive + false_positive,
        )
        recall = self._safe_divide(true_positive, total_positive)
        false_positive_rate = self._safe_divide(false_positive, total_negative)
        f1 = self._safe_divide(2.0 * precision * recall, precision + recall)
        best_index = int(np.argmax(f1)) if len(f1) else 0
        thresholds = np.arange(self.bins - 1, -1, -1) / self.bins
        eligible = np.flatnonzero(
            false_positive_rate <= float(max_false_positive_rate)
        )
        operating_index = int(eligible[-1]) if len(eligible) else 0

        recall_previous = np.concatenate((np.array((0.0,)), recall[:-1]))
        auprc = float(np.sum((recall - recall_previous) * precision))
        fpr_previous = np.concatenate(
            (np.array((0.0,)), false_positive_rate[:-1])
        )
        auroc = float(
            np.sum(
                (false_positive_rate - fpr_previous)
                * (recall + np.concatenate((np.array((0.0,)), recall[:-1])))
                * 0.5
            )
        )

        fixed_index = int(
            np.clip(
                round((1.0 - float(fixed_threshold)) * self.bins) - 1,
                0,
                self.bins - 1,
            )
        )
        tp = int(true_positive[fixed_index])
        fp = int(false_positive[fixed_index])
        fn = int(total_positive - tp)
        tn = int(total_negative - fp)

        def confusion_at(index):
            """Full 2x2 confusion plus imbalance-robust derived scores."""

            c_tp = int(true_positive[index])
            c_fp = int(false_positive[index])
            c_fn = int(total_positive - c_tp)
            c_tn = int(total_negative - c_fp)
            recall = c_tp / max(c_tp + c_fn, 1)
            specificity = c_tn / max(c_tn + c_fp, 1)
            denominator = float(
                (c_tp + c_fp) * (c_tp + c_fn) * (c_tn + c_fp) * (c_tn + c_fn)
            )
            mcc = (
                (c_tp * c_tn - c_fp * c_fn) / math.sqrt(denominator)
                if denominator > 0.0
                else 0.0
            )
            return {
                "threshold": float(thresholds[index]),
                "true_positive": c_tp,
                "false_positive": c_fp,
                "true_negative": c_tn,
                "false_negative": c_fn,
                "precision": c_tp / max(c_tp + c_fp, 1),
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": c_fp / max(c_fp + c_tn, 1),
                "f1": 2.0 * c_tp / max(2 * c_tp + c_fp + c_fn, 1),
                "accuracy": (c_tp + c_tn) / max(c_tp + c_tn + c_fp + c_fn, 1),
                "balanced_accuracy": 0.5 * (recall + specificity),
                "matthews_corrcoef": float(mcc),
            }

        return {
            # Full confusion matrices at the three thresholds that matter:
            # the deployment operating point, the best-F1 point, and whatever
            # fixed threshold the caller asked about. Reported always, because
            # scalar summaries hide failure structure such as an inverted
            # ranking.
            "confusion_matrix": {
                "operating": confusion_at(operating_index),
                "best_f1": confusion_at(best_index),
                "fixed": confusion_at(fixed_index),
            },
            "real_count": total_negative,
            "ghost_count": total_positive,
            "auprc": auprc,
            "auroc": auroc,
            "best_f1": float(f1[best_index]),
            "best_threshold": float(thresholds[best_index]),
            "max_false_positive_rate": float(max_false_positive_rate),
            "operating_threshold": float(thresholds[operating_index]),
            "operating_precision": float(precision[operating_index]),
            "operating_recall": float(recall[operating_index]),
            "operating_false_positive_rate": float(
                false_positive_rate[operating_index]
            ),
            "fixed_threshold": float(fixed_threshold),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
        }


def format_confusion_matrix(block, title="confusion matrix"):
    """Render one confusion-matrix block as a readable ASCII table."""

    tp = block["true_positive"]
    fp = block["false_positive"]
    tn = block["true_negative"]
    fn = block["false_negative"]
    width = max(9, max(len(f"{value:,d}") for value in (tp, fp, tn, fn)))
    lines = [
        f"{title} @ threshold {block['threshold']:.4f}",
        f"{'':>14}{'pred real':>{width + 2}}{'pred ghost':>{width + 2}}",
        f"{'actual real':>14}{tn:>{width + 2},d}{fp:>{width + 2},d}"
        f"    <- {block['false_positive_rate'] * 100:.2f}% false rejection",
        f"{'actual ghost':>14}{fn:>{width + 2},d}{tp:>{width + 2},d}"
        f"    <- {block['recall'] * 100:.2f}% ghost recall",
        "",
        f"  precision {block['precision']:.4f}   recall {block['recall']:.4f}   "
        f"f1 {block['f1']:.4f}",
        f"  specificity {block['specificity']:.4f}   "
        f"balanced_acc {block['balanced_accuracy']:.4f}   "
        f"mcc {block['matthews_corrcoef']:+.4f}",
    ]
    return "\n".join(lines)


def format_all_confusion_matrices(result):
    """Render the operating / best-F1 / fixed matrices from a compute() dict."""

    matrices = result.get("confusion_matrix") or {}
    titles = {
        "operating": "operating point (<= max real FPR)",
        "best_f1": "best-F1 point",
        "fixed": "fixed threshold",
    }
    sections = [
        format_confusion_matrix(matrices[key], titles[key])
        for key in ("operating", "best_f1", "fixed")
        if key in matrices
    ]
    return "\n\n".join(sections)
