# Translation project validation

Both translation notebooks were executed in order with `PROJECT_FULL=1` on CPU: 150 training epochs per model. The recorded English-Italian grammar corpus is synthetic. The exact-match scores describe held-out combinations of this grammar and do not measure unrestricted translation quality.

The execution engine used Python cells directly with Matplotlib's Agg backend because this runtime could not open a Jupyter kernel socket. Cell outputs and figures were captured from the executed code. Use `python scripts/execute_translation.py --inprocess` for the same route, or omit that option in a normal Jupyter environment.

`python -m ruff check src tests scripts` passed. Fifteen unit tests passed, including the existing NLP tests and seven translation checks. The test run used `LOKY_MAX_CPU_COUNT=1` to avoid restricted process introspection in the host environment. Causal invariance, source-padding invariance and attention normalization were also checked inside the notebooks.

Actual package versions and execution records are in `results/translation_environment.json` and `results/translation_execution_report.json`. The optional real-corpus TSV route was not benchmarked. No saved weights from the supplied TensorFlow examples were used.
