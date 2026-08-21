"""Microbenchmarks / profiling drivers — single-config tools the A/B evaluator doesn't import.

`bench.eval` (both targets: hopper vs FA3, blackwell vs the FA4 oracle) depends only on
`bench.scenarios`. The tools in here are single-config drivers for NCU/nsys, a num_splits sweep, and
a reference-PTX dumper. They import FROM the harness (`bench.scenarios` / `bench.eval`), never the
other way round, so nothing here is on the evaluator's dependency path.

Run them as modules from the repo root, e.g.

    python -m bench.microbenchmarks.prof_one
    ncu --set detailed python -m bench.microbenchmarks.prof_fa3
"""
