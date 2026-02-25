# alignment-tax-quantifier

**What does safety actually cost?**

## The problem

The AI safety conversation is full of qualitative arguments and short on numbers. Teams add RLHF, constitutional AI methods, or output filtering and assume the cost is "worth it" without ever measuring what "it" is.

This tool measures it. FLOPs overhead. Wall-clock training time delta. Downstream task performance regression. Across methods, across model scales.

## Why this matters

You can't have an honest conversation about alignment tradeoffs without data. A 40% compute overhead might be acceptable for a chatbot. It's a dealbreaker for real-time medical inference. The numbers change the engineering decisions — but only if you have the numbers.

## Features

- Benchmark RLHF, Constitutional AI, and filtering-based safety methods against unmodified baselines
- Track FLOPs, wall-clock time, and task performance at each scale
- Structured JSON reports and comparison visualizations
- Pluggable — bring your own safety method

## Quickstart

```bash
pip install -r requirements.txt
python benchmark.py --method rlhf --model gpt2 --baseline results/baseline.json
```

## License

MIT
