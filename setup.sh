#!/usr/bin/env bash
# One-shot env setup for fa3-cuda / fa4_blackwell benches.
#
# Install order matters: vendored `flash-attn-4` pins `nvidia-cutlass-dsl==4.6.0.dev0`,
# whose cu13 wheel layout breaks the flavor loader (missing
# `cu13/lib/libcute_dsl_runtime.so` → FA4 baseline import fails). This script installs
# a known-good cutlass-dsl[cu13] first, then FA4 with --no-deps.
#
# Usage:
#   ./setup.sh              # torch + FA4 CuteDSL baseline + cuDNN + FlashInfer + matplotlib
#   ./setup.sh --with-vllm  # also install vLLM (Hopper FA3 baseline)
#   ./setup.sh --skip-torch # assume torch is already installed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WITH_VLLM=0
SKIP_TORCH=0
CUTLASS_DSL_VER="${CUTLASS_DSL_VER:-4.6.1}"
FA4_PATH="./reference/flash-attention/flash_attn/cute"

for arg in "$@"; do
  case "$arg" in
    --with-vllm) WITH_VLLM=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  echo "[warn] no venv/conda env detected; installing into the active Python:"
  python -c 'import sys; print(" ", sys.executable)'
fi

echo "[check] CUDA toolkit / nvcc"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not on PATH (need CUDA toolkit >= 13 for Blackwell / FA4)." >&2
  exit 1
fi
nvcc --version | sed -n 's/^Cuda compilation tools,/  &/p'

echo "[check] FA4 source at ${FA4_PATH}"
if [[ ! -f "${FA4_PATH}/pyproject.toml" ]]; then
  echo "error: ${FA4_PATH} missing. Init submodules:" >&2
  echo "  git submodule update --init reference/cutlass reference/flash-attention" >&2
  exit 1
fi

PYTHON_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

purge_cutlass() {
  echo "[clean] remove prior nvidia-cutlass-dsl installs (avoids empty cu13/ leftovers)"
  pip uninstall -y \
    nvidia-cutlass-dsl \
    nvidia-cutlass-dsl-libs-base \
    nvidia-cutlass-dsl-libs-cu12 \
    nvidia-cutlass-dsl-libs-cu13 \
    nvidia-cutlass-dsl-libs-core \
    2>/dev/null || true
  rm -rf "${PYTHON_SITE}/nvidia_cutlass_dsl"
}

if [[ "$SKIP_TORCH" -eq 0 ]]; then
  echo "[1/4] torch (+ CUDA runtime wheels already pulled by a cu13 torch build)"
  pip install 'torch>=2.4'
else
  echo "[1/4] skip torch (--skip-torch)"
  python -c 'import torch; print("  torch", torch.__version__, "cuda", torch.version.cuda)'
fi

echo "[2/4] nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VER} (FA4 CuteDSL runtime)"
purge_cutlass
pip install "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VER}"

RUNTIME_SO="${PYTHON_SITE}/nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"
if [[ ! -f "$RUNTIME_SO" ]]; then
  echo "error: expected ${RUNTIME_SO} after cutlass-dsl install." >&2
  echo "       Set CUTLASS_DSL_VER to a cu13 wheel that ships cu13/lib/ (e.g. 4.6.1)." >&2
  exit 1
fi

echo "[3/4] flash-attn-4 from ${FA4_PATH} (--no-deps so it cannot downgrade cutlass to .dev0)"
pip install \
  'einops' \
  'typing_extensions' \
  'apache-tvm-ffi>=0.1.12,<0.2' \
  'torch-c-dlpack-ext'
# quack-kernels also pins cutlass==4.6.0.dev0
pip install --no-deps 'quack-kernels>=0.5.3'
pip install --no-deps "${FA4_PATH}[cu13]"

echo "[4/4] bench extras (cuDNN + FlashInfer references + plot)"
pip install 'nvidia-cudnn-frontend>=1.14' 'flashinfer-python>=0.6' 'matplotlib>=3.7'

if [[ "$WITH_VLLM" -eq 1 ]]; then
  echo "[extra] vllm (Hopper FA3 baseline _vllm_fa3_C)"
  pip install vllm
fi

echo "[verify] FA4 CuteDSL baseline"
python - <<'PY'
import site
from pathlib import Path

from flash_attn.cute.interface import _flash_attn_fwd

so = Path(site.getsitepackages()[0]) / "nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"
assert so.is_file(), f"missing {so}"
assert _flash_attn_fwd is not None
print("  _flash_attn_fwd OK")
print(f"  cutlass cu13 runtime OK ({so})")
PY

echo
echo "Setup complete. Examples:"
echo "  python -m bench.eval --target blackwell"
echo "  python -m bench.eval --target hopper      # needs vllm: re-run with --with-vllm"
echo
echo "Note: do not plain-pip the cute[cu13] extra afterward — it re-pulls cutlass 4.6.0.dev0."
echo "      Re-run ./setup.sh instead (or: CUTLASS_DSL_VER=4.6.1 ./setup.sh --skip-torch)."
