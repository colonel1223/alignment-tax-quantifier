# alignment-tax-quantifier

Measure and benchmark the computational overhead that safety constraints introduce into ML training pipelines.

## Motivation

Every guardrail has a cost — additional compute, longer training time, reduced throughput, degraded task performance. This tool quantifies that cost across different safety interventions so teams can make informed engineering tradeoffs rather than guessing.

## Features

- Benchmark RLHF, Constitutional AI, and filtering-based safety methods against baseline training
- Track FLOPs overhead, wall-clock time delta, and downstream task performance regression
- Generate comparison reports across intervention types and model scales
- Pluggable architecture for custom safety methods

## Quickstart

```bash
pip install -r requirements.txt
python benchmark.py --method rlhf --model gpt2 --baseline results/baseline.json
```

## Output

Produces structured JSON reports and matplotlib visualizations comparing alignment tax across methods.

## License

MIT
