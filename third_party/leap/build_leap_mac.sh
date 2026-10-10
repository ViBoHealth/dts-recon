#!/bin/bash
# Build LEAP (leapct) in CPU-only mode on macOS / Apple Silicon and install it into a Python env.
#
# Usage: third_party/leap/build_leap_mac.sh <work_dir> [python]
#   work_dir: empty or new directory to clone LEAP into (outside the repo)
#   python:   interpreter whose site-packages receives leapct (default: .venv/bin/python)
#
# Requires: Xcode command line tools, Homebrew libomp (`brew install libomp`),
#           cmake (installed in the venv: `uv pip install cmake`).
#
# Steps (see reports/M0_mac_setup.md and docs/decisions.md):
#   1. clone LLNL/LEAP at LEAP_TAG
#   2. apply cpu_build_fix.patch (guards three GPU-only calls missing #ifndef __USE_CPU in v1.24-v1.26)
#   3. use LEAP's own src/cpu_CMakeLists.txt (the default one requires CUDA)
#   4. configure with explicit OpenMP flags (Apple clang ships no OpenMP; FindOpenMP needs them for keg-only libomp)
#      and CMAKE_POLICY_VERSION_MINIMUM=3.5 (CMake >= 4 rejects LEAP's cmake_minimum_required(3.0))
#   5. copy libleapct.dylib + the ctypes Python modules into site-packages, as LEAP's manual_install.py does
#      (leaptorch.py is skipped: it needs PyTorch, which is not a project dependency)
set -euo pipefail

LEAP_TAG=v1.26
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:?usage: $0 <work_dir> [python]}"
PY="${2:-$REPO/.venv/bin/python}"
CMAKE="$(dirname "$PY")/cmake"
[ -x "$CMAKE" ] || CMAKE=cmake
OMP="$(/opt/homebrew/bin/brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)"
[ -f "$OMP/lib/libomp.dylib" ] || { echo "libomp not found at $OMP; run: brew install libomp"; exit 1; }

mkdir -p "$WORK"
cd "$WORK"
[ -d LEAP ] || git clone --quiet https://github.com/LLNL/LEAP.git
cd LEAP
git checkout --quiet "$LEAP_TAG"
git checkout --quiet -- .
git apply "$HERE/cpu_build_fix.patch"
cp src/cpu_CMakeLists.txt src/CMakeLists.txt

rm -rf build && mkdir build && cd build
"$CMAKE" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib"
"$CMAKE" --build . -j "$(sysctl -n hw.ncpu)"
cd ..

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp build/lib/libleapct.dylib src/leapctype.py src/leap_filter_sequence.py src/leap_preprocessing_algorithms.py "$SITE/"
echo "$LEAP_TAG $(git rev-parse --short HEAD) +cpu_build_fix.patch" > "$SITE/leapct_build_info.txt"
echo "Installed LEAP $LEAP_TAG (CPU) into $SITE"
