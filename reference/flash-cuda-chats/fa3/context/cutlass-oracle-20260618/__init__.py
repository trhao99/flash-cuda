"""cutlass_oracle — build high-performance CUTLASS GEMMs, inspect their PTX/SASS, and
distill a pure-CUDA/PTX equivalent that matches CUTLASS performance + output.

See TRACE_prefill_hd128_bf16.md for the traced CUTLASS templates this targets.
"""
