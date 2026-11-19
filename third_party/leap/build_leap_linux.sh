#!/bin/bash
# Build LEAP (leapct) v1.26 on Linux x86_64 (also WSL2), CPU-only or CUDA, and install it into a Python env.
#
# Usage: third_party/leap/build_leap_linux.sh cpu|cuda <work_dir> [python]
#   cpu|cuda: cpu uses LEAP's src/cpu_CMakeLists.txt (no GPU); cuda uses LEAP's default CUDA build
#   work_dir: empty or new directory to clone LEAP into (outside the repo)
#   python:   interpreter whose site-packages receives leapct (default: .venv/bin/python)
#
# Requires: git, cmake >= 3.18, g++ with OpenMP (gcc's libgomp); for cuda also the CUDA toolkit (nvcc >= 11.7,
#           on PATH) and a working NVIDIA driver (nvidia-smi).
# Environment: CUDA_ARCH (default 75 = Turing, e.g. Quadro RTX 5000), JOBS (default: all cores).
#
# Same steps as build_leap_mac.sh (see there and docs/decisions.md), minus the Homebrew libomp flags:
#   1. clone LLNL/LEAP at LEAP_TAG
#   2. apply cpu_build_fix.patch and joseph_modular_cpu_fix.patch. Both patch CPU code paths; the CUDA build
#      compiles them too (used when LEAP runs on the CPU). The GPU projectors are unpatched and untested
#      for our geometry: docs/LINUX_SETUP.md says how to test them.
#   3. cpu: use src/cpu_CMakeLists.txt. cuda: keep LEAP's src/CMakeLists.txt and set CMAKE_CUDA_ARCHITECTURES
#      (CMake 3.23 and older use it; newer CMake builds LEAP's all-major set instead, whose sm_70 code
#      also runs on 7.5 cards).
#   4. copy libleapct.so + the ctypes Python modules into site-packages, as LEAP's manual_install.py does
#      (leaptorch.py is skipped: it needs PyTorch, which is not a project dependency)
# UNTESTED: written on a Mac from LEAP's CMake files; first run is on the collaborator's Linux PC.
set -euo pipefail

LEAP_TAG=v1.26
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MODE="${1:?usage: $0 cpu|cuda <work_dir> [python]}"
WORK="${2:?usage: $0 cpu|cuda <work_dir> [python]}"
PY="${3:-$REPO/.venv/bin/python}"
CUDA_ARCH="${CUDA_ARCH:-75}"
JOBS="${JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
CMAKE="$(dirname "$PY")/cmake"
[ -x "$CMAKE" ] || CMAKE=cmake

case "$MODE" in
  cpu) ;;
  cuda)
    command -v nvcc >/dev/null || { echo "nvcc not found: add the CUDA toolkit to PATH, e.g. export PATH=/usr/local/cuda/bin:\$PATH"; exit 1; }
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || { echo "nvidia-smi does not see a GPU: check the NVIDIA driver first"; exit 1; }
    ;;
  *) echo "first argument must be cpu or cuda"; exit 1 ;;
esac
[ -x "$PY" ] || { echo "python not found: $PY"; exit 1; }

mkdir -p "$WORK"
cd "$WORK"
[ -d LEAP ] || git clone --quiet https://github.com/LLNL/LEAP.git
cd LEAP
git checkout --quiet "$LEAP_TAG"
git checkout --quiet -- .
git apply "$HERE/cpu_build_fix.patch"
git apply "$HERE/joseph_modular_cpu_fix.patch"

ARGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
if [ "$MODE" = cpu ]; then
  cp src/cpu_CMakeLists.txt src/CMakeLists.txt
  TAGLINE="cpu"
else
  ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_CUDA_COMPILER="$(command -v nvcc)")
  TAGLINE="cuda $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') sm_$CUDA_ARCH"
fi

rm -rf build && mkdir build && cd build
"$CMAKE" .. "${ARGS[@]}"
"$CMAKE" --build . -j "$JOBS"
cd ..

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp build/lib/libleapct.so src/leapctype.py src/leap_filter_sequence.py src/leap_preprocessing_algorithms.py "$SITE/"
echo "$LEAP_TAG $(git rev-parse --short HEAD) $TAGLINE +cpu_build_fix.patch +joseph_modular_cpu_fix.patch" > "$SITE/leapct_build_info.txt"
echo "Installed LEAP $LEAP_TAG ($TAGLINE) into $SITE"
