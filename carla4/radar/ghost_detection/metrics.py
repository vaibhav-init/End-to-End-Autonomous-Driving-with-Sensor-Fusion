"""Dependency-light streaming binary metrics and threshold selection."""

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
        return {
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
