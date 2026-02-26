# alignment-tax-quantifier

**What does safety cost?**

Benchmarking framework for measuring the computational overhead of AI alignment methods. Every safety intervention has a cost in FLOPs, memory, and downstream task performance. This tool produces the numbers.

## Quick start

```python
from core import AlignmentTaxQuantifier

q = AlignmentTaxQuantifier()
q.benchmark()
print(q.report())
```

## Methods

| Method | FLOP Overhead | Memory | Task Regression | Source |
|--------|:---:|:---:|:---:|:---:|
| RLHF | +40% | +110% | -2.0pp | Ouyang et al. (2022) |
| Constitutional AI | +60% | +30% | -1.0pp | Bai et al. (2022) |
| DPO | +15% | +50% | -1.5pp | Rafailov et al. (2023) |
| Output filtering | +5% | +10% | -0.5pp | — |
| Activation steering | +2% | +1% | -0.8pp | Turner et al. (2023) |
| ROME/MEMIT | +0% | +0% | -0.3pp | Meng et al. (2022) |

## Modules

| Module | What |
|--------|------|
| `core/tax_quantifier.py` | Benchmark engine with 6 methods, multi-scale |
| `core/scaling.py` | Power-law scaling fits, extrapolation |
| `core/pareto.py` | Pareto frontier for method selection |

## Tests

```bash
cd tests && python test_tax.py
```

## License

MIT

---

## Research Ecosystem

This framework is part of a unified AI safety research program. See [colonel1223.net](https://colonel1223.net) for the full portfolio.

| Related Project | Connection |
|----------------|------------|
| [conformal-multimodal](https://github.com/colonel1223/conformal-multimodal) | Provides distribution-free uncertainty quantification for alignment measurements |
| [CHIMERA](https://github.com/colonel1223/CHIMERA) | Hallucination bounds interact with alignment tax — safer models hallucinate differently |
| [SHOGGOTH](https://github.com/colonel1223/SHOGGOTH) | Visualizes the alignment dynamics this framework measures |
| [Research Papers](https://github.com/colonel1223/ai-research-modern-alchemy) | Scaling law analysis and economic modeling of alignment costs |
