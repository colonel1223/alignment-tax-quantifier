"""Core alignment tax measurement with Shapley attribution (scaffold)."""
import numpy as np
import torch
from typing import Dict

class TaxQuantifier:
    def __init__(self, model, constraints, metric):
        self.model = model
        self.constraints = constraints
        self.metric = metric

    def measure_tax(self, dataset, bootstrap_samples: int = 1000) -> Dict:
        baseline = self._evaluate(dataset, aligned=False)
        aligned   = self._evaluate(dataset, aligned=True)
        tax = ((baseline - aligned) / baseline) * 100 if baseline else 0.0
        return {"tax_percentage": tax}

    def _evaluate(self, dataset, aligned: bool):
        # TODO: implement evaluation logic
        return 1.0
