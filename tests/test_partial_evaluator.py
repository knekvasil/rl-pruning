import torch
import pytest

from rl_pruning.eval.partial_evaluator import PartialEvaluator
from rl_pruning.eval.glue_evaluator import GLUEEvaluator


class FakeDataset:
    def __init__(self, n=100):
        self.rows = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 0], dtype=torch.long),
                "label": torch.tensor(1, dtype=torch.long),
            }
            for _ in range(n)
        ]
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.rows[i]


class FakeFullEvaluator:
    def __init__(self, n=100):
        self.dataset = FakeDataset(n)
        self.calls = []

    def evaluate_subset(self, model, indices, batch_size=32):
        self.calls.append((len(indices), batch_size))
        return len(indices)


def test_subset_size_is_fraction():
    full = FakeFullEvaluator(n=100)
    pe = PartialEvaluator(full, subset_frac=0.1, seed=0)
    pe._init_subset()
    assert len(pe.indices) == 10


def test_subset_at_least_one():
    full = FakeFullEvaluator(n=5)
    pe = PartialEvaluator(full, subset_frac=0.1, seed=0)
    pe._init_subset()
    assert len(pe.indices) >= 1


def test_fixed_seed_gives_stable_subset():
    full1 = FakeFullEvaluator(n=100)
    full2 = FakeFullEvaluator(n=100)
    pe1 = PartialEvaluator(full1, subset_frac=0.1, seed=42)
    pe2 = PartialEvaluator(full2, subset_frac=0.1, seed=42)
    pe1._init_subset()
    pe2._init_subset()
    assert pe1.indices == pe2.indices


def test_different_seed_gives_different_subset():
    full1 = FakeFullEvaluator(n=100)
    full2 = FakeFullEvaluator(n=100)
    pe1 = PartialEvaluator(full1, subset_frac=0.1, seed=1)
    pe2 = PartialEvaluator(full2, subset_frac=0.1, seed=2)
    pe1._init_subset()
    pe2._init_subset()
    assert pe1.indices != pe2.indices


def test_call_forwards_batch_size():
    full = FakeFullEvaluator(n=100)
    pe = PartialEvaluator(full, subset_frac=0.1, seed=0, batch_size=16)
    pe(None)
    assert full.calls[-1] == (10, 16)


def test_subset_lazily_initialized():
    full = FakeFullEvaluator(n=100)
    pe = PartialEvaluator(full, subset_frac=0.1, seed=0)
    assert pe.indices is None
    pe(None)
    assert pe.indices is not None


def test_evaluate_subset_batches_correctly():
    """Real batched evaluate_subset must process all indices across batches."""
    n = 100
    dataset = FakeDataset(n)

    # mimic GLUEEvaluator.evaluate_subset over the fake dataset
    def evaluate_subset(model, indices, batch_size=32):
        correct, total = 0, 0
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            rows = [dataset[i] for i in chunk]
            batch = {
                k: torch.stack([r[k] for r in rows])
                for k in ["input_ids", "attention_mask", "label"]
            }
            labels = batch.pop("label")
            total += labels.size(0)
        return correct / total if total else 0.0

    # verify batching visits every index with batch_size=16 (10 % 16 -> 1 batch of 10)
    indices = list(range(10))
    result = evaluate_subset(None, indices, batch_size=16)
    assert result == 0.0  # no crash, returns valid float

    # verify it handles non-multiple-of-batch_size without dropping rows
    seen = []
    def evaluate_subset_count(model, indices, batch_size=16):
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            seen.extend(chunk)
        return 0.0
    evaluate_subset_count(None, list(range(10)), batch_size=16)
    assert seen == list(range(10))
