# Linux setup and GPU check

These are setup steps for a Linux x86_64 PC, with an optional NVIDIA GPU. They were written for a Linux Mint 21.2 / Ubuntu 22.04 PC with a Quadro RTX 5000 and CUDA 12.3. The same CPU steps should work in WSL2 on Windows.

**Status: untested.** These steps and `third_party/leap/build_leap_linux.sh` were written on a Mac, from LEAP's build files. Your first run is also their first test. If a step fails, stop and send the full output; please don't work around it.

What the run is for:
1. **First Linux install.** Build the tool with the CPU-only LEAP and run the test suite. So far it has only been run on an Apple Silicon Mac.
2. **First GPU test.** Rebuild LEAP with CUDA and run the same tests. The tests check LEAP's projector and backprojector against our own NumPy reference. Two LEAP bugs were found and patched on the CPU side; nobody has checked whether its GPU code gives the same answers for this geometry.

You need a few GB of free disk space (two Python environments and two LEAP builds) and no sudo rights. All commands run in a normal terminal.

## 0. Check the GPU driver (skip for CPU only)

```bash
nvidia-smi
nvidia-smi -L
nvcc --version
```

- `nvidia-smi` should list the GPU(s), with driver and CUDA versions. If it says it can't talk to the driver, the NVIDIA kernel module isn't loaded; fix that before step 4. (A hardware report from this PC showed no `nvidia` module loaded and software OpenGL.)
- `nvcc --version` should report release 12.3. If `nvcc` isn't found, add the toolkit to `PATH`, for example `export PATH=/usr/local/cuda-12.3/bin:$PATH` (or `/usr/local/cuda/bin`), and put that line in `~/.bashrc`.

## 1. Get the code

You need access to the private GitHub repository `ViBoHealth/dts-recon-shared` (Gil sends the invitation). Then:

```bash
git clone https://github.com/ViBoHealth/dts-recon-shared.git
cd dts-recon-shared
```

If git asks for a password, GitHub needs a token or SSH key instead: the simplest route is the GitHub CLI (`gh auth login`, then `gh auth setup-git`).

## 2. Python environment

The project uses `uv`, which installs its own Python and needs no sudo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once; then open a new terminal
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python -e '.[dev,leap,dicom]'
```

Don't install `cmake` into the environment: the build script then uses the system `cmake` (3.22 is fine).

## 3. CPU build and tests (first Linux install)

```bash
third_party/leap/build_leap_linux.sh cpu ~/leap_build_cpu .venv-cpu/bin/python
.venv-cpu/bin/pytest -rs
```

- The build clones LEAP v1.26 into `~/leap_build_cpu`, applies our two patches and installs LEAP into `.venv-cpu`.
- Expected result: **92 passed, 0 skipped**, in well under 2 minutes. On the Mac it is 92 passed in about 23 s.
- If the summary lists skipped tests, the LEAP tests didn't run. `-rs` prints the reasons; send them.

## 4. CUDA build and tests (GPU check)

Use a second environment, so the CPU result stays available for comparison:

```bash
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python -e '.[dev,leap,dicom]'
CUDA_ARCH=75 third_party/leap/build_leap_linux.sh cuda ~/leap_build_gpu .venv-gpu/bin/python
```

- `CUDA_ARCH=75` targets Turing cards (Quadro RTX 5000). The CUDA build takes longer than the CPU build.
- Check that LEAP sees the GPU:

  ```bash
  .venv-gpu/bin/python -c "from leapctype import tomographicModels as T; lp = T(); print('GPUs:', lp.number_of_gpus(), 'using:', lp.get_gpu())"
  ```

  `using:` must be 0 or higher. A value of -1 means LEAP is running on the CPU.
- Run the tests:

  ```bash
  .venv-gpu/bin/pytest -rs
  ```

  With a CUDA build, LEAP uses the GPU automatically, so every test that uses LEAP now runs on the GPU. **A failure here is a result, not a setup problem:** for example, `test_projection_geometry` failing would mean the GPU projector puts beads in a different place than our reference. Send the output as it is.

If the import fails with `libcufft.so... cannot open shared object file`, the CUDA libraries aren't on the loader path: `export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH` and try again.

## 5. Speed comparison (optional)

The M0 benchmark reconstructs a 256³ volume from 41 projections in several configurations. Run it once per build:

```bash
.venv-cpu/bin/python scripts/m0_benchmark.py && mv reports/m0_benchmark_results.json ~/bench_cpu.json
.venv-gpu/bin/python scripts/m0_benchmark.py && mv reports/m0_benchmark_results.json ~/bench_gpu.json
```

## 6. What to send back

- the output of `nvidia-smi -L` and `nvcc --version`;
- the two `leapct_build_info.txt` files:

  ```bash
  cat .venv-cpu/lib/python3.12/site-packages/leapct_build_info.txt .venv-gpu/lib/python3.12/site-packages/leapct_build_info.txt
  ```
- the full output of both `pytest -rs` runs;
- the output of the GPU check command in step 4;
- if you ran step 5, `~/bench_cpu.json` and `~/bench_gpu.json`;
- the full output of any step that failed.
