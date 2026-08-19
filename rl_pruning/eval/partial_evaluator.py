import random
import torch


class PartialEvaluator:
    def __init__(self, full_evaluator, subset_frac=0.1, seed=0, batch_size=32):
        self.full = full_evaluator
        self.subset_frac = subset_frac
        self.seed = seed
        self.batch_size = batch_size
        self.indices = None

    def _init_subset(self):
        rng = random.Random(self.seed)
        n = len(self.full.dataset)
        k = max(1, int(n * self.subset_frac))
        self.indices = rng.sample(range(n), k)

    def __call__(self, model):
        if self.indices is None:
            self._init_subset()
        return self.full.evaluate_subset(model, self.indices, self.batch_size)
