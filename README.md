# alignment-tax-quantifier

**What does safety actually cost?**

## The problem

Teams add RLHF, constitutional AI methods, or output filtering and assume the cost is "worth it" without measuring what "it" is. This project measures it.

## What's here

`core/tax_quantifier.py` — Python framework for benchmarking alignment overhead: FLOPs, wall-clock time, and downstream task performance regression across safety methods and model scales.

`docs/` — Interactive documentation and results visualization via GitHub Pages.

## Structure

```
├── core/
│   └── tax_quantifier.py    # Benchmarking framework
├── docs/
│   ├── index.html           # Results visualization
│   └── .nojekyll
└── README.md
```

## Usage

```python
from core.tax_quantifier import AlignmentTaxQuantifier

quantifier = AlignmentTaxQuantifier(method="rlhf", baseline="vanilla")
report = quantifier.benchmark(model="gpt2", metrics=["flops", "wall_clock", "task_regression"])
```

## Status

Active development. Core framework functional, expanding method coverage and scale testing.

## License

MIT
