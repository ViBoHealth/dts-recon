# Packaging dts-recon for other insiders (proposal, 2026-11-18)

**Status: proposal only.** Nothing has been built or published, and no dependency has been added. Each step
below waits for Gil's approval.

**Goal:** colleagues without access to Gil's Mac can install `dts` and run it on x-ray data of their own.

Evidence labels used below:
- **measured:** checked in this repo or on this Mac on 2026-11-18.
- **read:** taken from an upstream page or file.
- **inferred:** an estimate, not checked.

## 0. Gil's answers (2026-11-18)

- **Users are probably on Windows PCs.** A Mac-only build is not acceptable.
- **CUDA access:** unknown. Gil will check.
- **GitHub:** Gil can give access to the company GitHub.
- **Their data:** regularly spaced 2D grids of projections, 5 × 5 up to 10 × 10, with **known geometry**.
- **Licence:** internal use only, so no licence for dts-recon.
- **File format:** raw detector output as TIFF. DICOM input is unlikely. The importer reads TIFF first; DICOM output (`--dicom`) stays as it is.
- **Repository:** the company GitHub repository is private.
- **Dependencies approved:** Pillow (declared in `pyproject.toml`), delocate and auditwheel (build machine only). Recorded in `docs/decisions.md` item 30 and `CLAUDE.md`.

- **GPU (Milaan, 2026-11-19):** one collaborator has a **Linux PC with a CUDA GPU**:
  - NVIDIA Quadro RTX 5000 (Turing, compute capability 7.5; the capability is inferred from the model), 16 GB;
  - CUDA toolkit 12.3 with compiler;
  - driver 580.178.04 (supports CUDA ≤ 13.0).

  So native Linux is a real target as well as WSL2, and a CUDA build has at least one user.

  From Milaan's hardinfo report (read 2026-11-19):
  - **System:** Linux Mint 21.2 (Ubuntu 22.04 base), kernel 6.8.0-138, x86_64, glibc 2.35, so manylinux_2_28 wheels apply.
  - **CPU and memory:** Intel i9-9900K (8 cores / 16 threads), 64 GB RAM, about 810 GB free disk.
  - **Software:** Python 3.10.12, gcc 11.4.0 (supported by CUDA 12.3), cmake 3.22.1, make 4.3.
  - **Not in the report:** git, uv/conda, nvcc, sudo rights.
  - **Two GPU entries:** the PCI list shows two "Quadro RTX 5000" VGA entries. That may be two cards; confirm with `nvidia-smi`.
  - **NVIDIA driver not loaded when the report ran.** The kernel module list (512 modules) has no `nvidia`, `nvidia_uvm` or `nvidia_drm` (only `i2c_nvidia_gpu`), and OpenGL renders in software (`llvmpipe`). If `nvidia-smi` fails, CUDA will not work until the driver is rebuilt or reinstalled for the running kernel. That is an admin task on Milaan's side (inferred: a kernel update without a matching driver module).
  - **Viewer:** with software OpenGL, napari will be slow. 2D slice viewing should be fine (inferred), and Fiji is an alternative.

These change the recommendation below. The first version (wheel for macOS + Linux first) is replaced.

### 0.1 What Milaan's machine changes (2026-11-19)

1. **It can be the GPU validation machine.** The GPU side of M0/M1 has never run: the CUDA LEAP build and the round-trip, projector and adjoint tests on a GPU (`pytest -m gpu`). It was planned for a rented GPU. Milaan's machine can do it, if Milaan agrees. Until that passes, no CUDA build goes to users (section 2, CUDA row).
2. **A CUDA LEAP build for Linux.**
   - Source and patches are the same as the CPU build.
   - It uses LEAP's default `src/CMakeLists.txt`, which is the CUDA one; the Mac script swaps in `cpu_CMakeLists.txt`.
   - Target `CMAKE_CUDA_ARCHITECTURES=75`, with CUDA 12.3 (inferred; not built yet).
   - Apply both patches here too. They change CPU code only, but they keep the CPU fallback correct and write the build marker the adapter checks.
   - Prefer building from source over conda-forge's `leapct`. That package is unpatched, pulls a CUDA PyTorch, and carries no CPU fix.
3. **First native Linux install test.** Milaan's PC is also where the Linux wheel (CPU first) gets its first test outside Gil's Mac.
4. **Expected gain (inferred):**
   - Our OS-SART keeps the data on the CPU and calls LEAP's GPU projector per subset, so each call copies data to and from the GPU.
   - The ASD-POCS TV step is NumPy, on the CPU.
   - So the speed-up will be well below the raw GPU speed. Measure it rather than assume it.
   - 16 GB should hold the dental-size volume (0.54 GB per copy at 850 × 600 × 265), since LEAP splits CPU-resident data into chunks for the GPU (read: `leapctype` docstring).

## 1. Recommendation (revised)

**Recommended: one Linux x86_64 build of dts-recon with the patched LEAP compiled in, run on Windows through WSL2.**
- WSL2 is Microsoft's built-in Linux layer for Windows 10/11. It installs with `wsl --install`, which needs admin rights once.
- Inside it, users install the Linux wheel with `uv tool install <wheel>`. No compiler or LEAP build is needed.
- napari works through WSLg on Windows 11 (inferred, not tried). The TIFF and DICOM output also opens in Windows tools such as Fiji.
- If they have NVIDIA GPUs, CUDA works inside WSL2 with the normal Windows driver (inferred from NVIDIA's WSL support, not tried). That makes a later CUDA build usable on the same PCs.
- Data should sit inside the WSL file system. Reading large frame sets from `C:\` through `/mnt/c` is slow (inferred).

**Fallback 1: the same build as a CPU Docker image.**
- Built from the same Linux wheel, run with Docker Desktop (which itself runs on WSL2).
- Use this if users would rather not touch a Linux shell, or if IT supports Docker but not WSL.
- Check first: Docker Desktop needs a paid subscription for companies with 250 or more employees or $10M or more in revenue.

**Fallback 2: a native Windows build.**
- This means building LEAP with Microsoft's compiler and our patch, which has never been built there (section 2).
- Keep it for later, and only if WSL2 and Docker are both blocked on their PCs.

**Mac:** keep a macOS arm64 wheel for Gil. It costs little, since the Mac build already works. Its only portability fix is bundling libomp.

Both the WSL2 and Docker paths need the same groundwork first (section 5). The groundwork is also most of the effort.

**Why the Linux wheel is the core:**
- One build serves WSL2, Docker and any Linux machine.
- LEAP is loaded through `ctypes`, not as a Python extension, so one wheel works on every Python ≥ 3.10.
- Its own size is about 1–2 MB.
- On Linux, LEAP builds with gcc and its own OpenMP, which avoids both the Mac's Homebrew problem and Windows' OpenMP problem.

**Expected run times for 10 × 10 grids (inferred, not measured):**
- Iterative cost scales with the number of projections. 100 projections is about 2.8× the 36 measured on the Mac.
- So OS-SART would take about 5–6 min on the M5 Max for the dental-size volume, and probably 2–4× longer on a typical office PC CPU.
- That is the case for CUDA if they have it (section 2, CUDA row).
- No 10 × 10 grid has been simulated yet. Studies went up to about 40 projections (M4). By the working rules, a 10 × 10 simulation with known ground truth should run before users rely on it.

## 2. Platforms

| Platform | Realistic? | What blocks it (evidence) |
|---|---|---|
| macOS arm64 | **Yes, now** | Works today on Gil's Mac. To make it portable: `libleapct.dylib` links Homebrew's libomp by absolute path (`/opt/homebrew/opt/libomp/lib/libomp.dylib`, measured with `otool -L`), so it fails on a Mac without Homebrew libomp. Fix: bundle libomp in the wheel (delocate), or link conda's `llvm-openmp`. `build_leap_mac.sh` is Mac-only (it calls `brew` and `sysctl`). |
| macOS x86_64 (Intel) | Possible, low priority | Same build with an x86_64 target. Untested. Ask whether anyone still has one. |
| Linux x86_64, CPU | **Yes**, with a new build script | LEAP's CPU build on Linux uses gcc + libgomp, which is simpler than on the Mac (inferred). Both patches touch only C++ sources, so they should apply unchanged (inferred). Needs a manylinux build and `auditwheel` to bundle libgomp. Not yet built or tested here. |
| Linux x86_64, CUDA | **Not yet** | The adapter skips its patch check when LEAP reports a GPU (`leap_adapter.py`: `lp.get_gpu() < 0 and not _cpu_patch_installed()`). The CUDA modular projector has **never been round-trip tested** for our geometry, because the M0/M1 GPU side is open (`docs/STATUS.md`). conda-forge has `leapct` 1.26 (read: anaconda.org, uploaded 2026-05-14). It is CUDA-only (the recipe skips CPU builds), needs a CUDA PyTorch, and is unpatched (read: leapct-feedstock `recipe.yaml`). So it is unusable on Macs, but a possible base for a later CUDA variant. Measured CPU times (above) suggest dental-size users don't need a GPU. |
| Windows via WSL2 (or Docker Desktop) | **Yes: the primary target** (section 0) | Runs the Linux x86_64 build unchanged. Needs WSL2 enabled (admin rights once). Not yet tried on a PC. |
| Windows, native | **No, for now** | LEAP has a Windows build path (`leapctype.py` looks for `libleapct.dll`), but neither patch has been built there. The patch adds `#pragma omp parallel for ... collapse(2)`, which MSVC's default OpenMP 2.0 does not support (needs `/openmp:llvm`; inferred from MSVC docs, not tried). Fallback 2 only. |

## 3. Distribution options

All sizes are installed sizes. Dependency sizes were measured in `.venv`:
- core runtime dependencies (numpy, scipy, matplotlib, pydantic, tifffile, xraydb, imageio, Pillow, …): about **200 MB**;
- napari + Qt (`[viz]`): about **480 MB** more;
- the whole dev venv: 1.0 GB.

| | Wheel with bundled patched LEAP | Own conda package | Docker image (CPU) | Docker (CUDA) | Hosted service |
|---|---|---|---|---|---|
| Build effort (inferred) | 2–3 days: build LEAP inside the wheel build, bundle the OpenMP runtime (delocate / auditwheel), test on clean machines | 3–4 days: recipe for patched `libleapct` (CPU) + `dtsrecon`, a private channel to host it | 1–2 days: two-stage Dockerfile (gcc builds LEAP, slim runtime), multi-arch build | after the GPU validation session, plus 1–2 days | weeks: upload, auth, storage, job queue, ops |
| User installs with | `uv tool install dtsrecon-0.1.0-py3-none-<platform>.whl` (Python 3.10+ needed; uv can provide it) | `mamba install -c <channel> dtsrecon` (needs conda/mamba) | `docker run -v <scan dir>:/data <image> dts recon /data -o /data/out` | same + NVIDIA driver and container toolkit | browser upload |
| Size | ~1–2 MB own + ~200 MB deps from PyPI | similar | ~0.6–0.9 GB image (inferred) | 3–5 GB (inferred) | none for the user |
| Viewer | napari via `[viz]` | napari via conda-forge | none; view files on the host | none | web viewer (to build) |
| Updates reach users | new wheel file / private index; `uv tool upgrade` | `mamba update` | `docker pull <tag>` | same | instantly |
| Main risk | macOS dylib relinking and notarization quirks; one build per platform | users without conda; channel hosting | Docker Desktop memory cap on Macs; file permissions on mounts | GPU projector not yet validated | uploads data that is commercially confidential (and later clinical) to a server; most effort |

**Not recommended now:**
- **Hosted service:** too much work for a handful of insiders, and it moves their data off their machines.
- **conda:** it handles OpenMP more cleanly, but it adds a channel to run and a tool users may not have. Worth revisiting if a CUDA variant built on conda-forge's `libleapct` becomes the GPU route.

**Interim option (no packaging):** clone the repo and run a build script, as on Gil's Mac today. This is fragile: it needs Xcode CLT, Homebrew libomp, cmake and git access to LLNL/LEAP. Use it only for one or two people who are helping test, not as the distribution.

## 4. What a user needs to bring their own data

### 4.1 Input formats accepted now

| Input | How it gets in | State |
|---|---|---|
| **`dts-raw` scan** (`docs/raw_format.md`): `scan.yaml` + headerless frames (uint16/uint32/float32, either byte order) | `dts info / calibrate / recon` | The only format the CLI reads. It is our own **PLACEHOLDER** strawman, not a detector's real output. |
| **2018 emitter-array layout**: one 14-bit TIFF per emitter (`A2.tif` …), air folders, attenuation PNGs, keys files (`read_keys`: `A2 x y` or `atten1RRCCC x y`), MJPEG AVI, Tecplot | `dtsrecon.io.emitter_array` functions, driven by `scripts/real_v1.py` and `scripts/real_v2.py` | **Script-only.** No CLI command. The scripts are not in the package (`packages.find where = src`). The configs in `configs/real/` hold absolute paths to Gil's machine (`/Users/gtravish/Downloads/...`). |
| 8-bit display images (intraoral POC, webp) with self-calibrated geometry | `scripts/poc_intraoral.py` | Script-only. Every acquisition value is assumed. |

### 4.2 What the user must supply for a `dts-raw` scan

- **Detector:** `n_cols`, `n_rows`, pixel pitch u/v (mm), `dtype`, byte order, bit depth.
  - Frames are row-major `[row, col]`.
  - The column index runs along +x.
- **Frames:**
  - at least one **dark** and one **flat**: the validator raises if either kind is missing, so already flat-corrected images cannot be used as they are;
  - projections with `position_index` running 0…N−1 with no gaps;
  - encoder x/y (mm) for every projection.
- **Geometry:**
  - `sid_mm` plus encoder x/y, which places sources in the detector frame (brief §4.1: origin at the detector centre, +z toward the source, stationary flat detector);
  - or an explicit per-position geometry YAML (`det_center`, `det_u`, `det_v`, `src`).
  - `dts calibrate` only works with our two-plane bead phantom (`default_calibration_phantom`), which has **not been manufactured** (`docs/STATUS.md`). So for now users must supply geometry themselves.
- **Optional:**
  - `object_z_range_mm`. Without it the default is 10–210 mm, a dental **PLACEHOLDER** (`pipeline.DEFAULT_Z_RANGE`).
  - exposure µA/ms, for scaling the flats.
- **Machine:** at least 16 GB RAM for dental-size volumes at 0.4 mm. The measured peak is 5.4 GB (OS-SART) and 6.9 GB (ASD-POCS).

### 4.3 Missing before outsiders can use it on their own data

1. **An importer for regular grids with known geometry** (Gil: 5 × 5 to 10 × 10, geometry known).
   - Proposal: `dts import <image folder> --grid 10x10 --pitch-mm <x>,<y> --sid-mm <S> --order raster|serpentine --pixel-mm <p> [--darks ...] [--flats ...]`, which writes `dts-raw`.
   - The pieces exist already: `geometry/generators.py` has `grid_source_positions(grid_shape, grid_extent_mm, source_to_detector_mm, order)` with serpentine and raster orders. A per-position CSV stays an option for grids that are not exactly regular.
   - Images are matched to grid positions by file order, or by a filename pattern such as `r03_c07.tif`.
   - Allow "already normalized" input: no darks/flats, and the user states that the images are transmission or line integrals.
   - The emitter-array loader could become `dts import --layout emitter-2018 --keys keys.txt`, with the paths moved out of `configs/real/`.
   - Still to learn from the users: file format and bit depth, detector size and pixel pitch, whether there are darks and flats, how positions are named, and where the grid origin is relative to the detector centre.
2. **`dts validate <scan>`**, run before a reconstruction. It should check:
   - the manifest and frame sizes;
   - the geometry (`ProjectionGeometry.validate`: orthonormal detector vectors, sources on +z);
   - source coverage (`dts plot-geometry` already does this for configs, not for scans);
   - saturation and defect fractions;
   - that the z range lies under the sources;
   - whether the volume fits in memory, before a long run starts.
3. **Error messages.**
   - Manifest problems come out as raw pydantic tracebacks.
   - A missing LEAP prints `leapct is not installed`, with no next step.
   - These should become one-line messages that say which field or file is wrong and what to do.
4. **Defaults that break or mislead outside the repo.**
   - `pipeline.DEFAULT_CONFIG` is `Path(__file__).parents[2] / "configs" / ...`. Installed from a wheel, that path points outside the package, so `dts mock-scan` without `--config` fails (inferred from the code; the configs are not package data).
   - The default binning (to 0.4 mm), the 20 mm margin and the z range are dental values.
5. **User docs.** `docs/USAGE.md` is mock-data only. It needs:
   - a "your own data" page: units, frame conventions, a worked `positions.csv` example, what "placeholder" means in outputs;
   - a page on how to read the results (methods, artefacts, depth scale).
6. **Provenance in outputs.**
   - Recon JSON/TIFF metadata does not record the `dtsrecon` version or the LEAP build string. `__version__` exists (`0.0.1`) but is not written.
   - Without them, users' results can't be traced back to a release.

## 5. What must be in place first (both paths)

1. **Reproducible LEAP build.**
   - Pin v1.26 by commit (`0c8846f`, measured from `leapct_build_info.txt`) and by archive SHA-256.
   - Builds currently clone from GitHub at build time. Instead, vendor the v1.26 source archive, or pin it by checksum.
   - Add a Linux build script next to `build_leap_mac.sh`.
   - Ship LEAP's Python modules (`leapctype.py`, `leap_filter_sequence.py`, `leap_preprocessing_algorithms.py`) inside the package, for example `dtsrecon/engines/_leap/`, so the default library lookup next to `leapctype.py` finds the bundled library and nothing collides with a user's own `leapct`.
   - Upstream status: v1.26 is still LEAP's latest release (read: GitHub releases), so both bugs are still present upstream. The fixes have not been reported upstream (decisions items 2 and 6). If upstream fixes them, the patch step goes away.
2. **A self-test that proves the patch, not a marker file.**
   - Today the adapter trusts `leapct_build_info.txt`.
   - Add `dts selftest`: a ≤ 32³ projector/backprojector round-trip against the NumPy reference projector (< 0.1 px), plus the adjoint check, in seconds.
   - Run it in the wheel/image CI and tell users to run it once after installing.
3. **Tests that run against the installed package.**
   - Today 92 tests pass in 23 s on this Mac (measured).
   - The 6 LEAP test files are `skipif(not leap_adapter.available())`, so a broken LEAP bundle would show up as *skipped*, not failed.
   - Add a "LEAP required" switch (for example an env var) that turns those skips into failures in packaging CI.
   - Run `pytest` against the installed wheel on a clean machine or container per platform.
4. **A git remote and CI.**
   - There is no remote yet (`git remote -v` is empty). It is needed to build per-platform wheels and images, and to host releases.
   - The macOS wheel can be built on this Mac, and the Linux one in a manylinux container, so CI is useful but not strictly needed for a first release.
5. **Versioning.**
   - Move to `0.1.0` and up, with git tags.
   - Wheel and image tags match the version.
   - Write the version and LEAP build string into every output (section 4.3 item 6).
   - Keep a short CHANGELOG for users.
6. **Licences.**
   - dts-recon: internal use, no licence (Gil, 2026-11-18). A one-line "internal use only" note in the README is enough.
   - LEAP is MIT (read: LICENSE, "Copyright (c) 2013-2023 LLNS, LLC and other LEAP Project Developers"). MIT still requires its notice to travel with copies, so the wheel/image includes LEAP's LICENSE file and a note that the binary is modified (our two patches). This is one file, with no action needed from Gil.
   - TIGRE is BSD-3-Clause (read: its GitHub page). It is not bundled, since it isn't used yet. This closes the brief §3.1 open item for TIGRE, subject to Gil's own check against commercial plans.
   - napari is BSD, but `napari[all]` pulls **PyQt5 (GPL / commercial)**. Don't bundle Qt in any image. Leave `[viz]` as a user-installed extra, or switch it to PySide (LGPL).
   - xraydb's licence still needs checking (not read).
7. **Keep data out of the package.**
   - The wheel only contains `src/dtsrecon` (setuptools `where = src`), so `data/` (8.6 GB here, including the 2018 and WeTransfer real data) cannot get into it. Add a check that the wheel contains only `dtsrecon/**` and licence files.
   - A Docker build context would include `data/` unless there is a `.dockerignore`. Add one that allowlists `src/`, `pyproject.toml`, `third_party/leap/` and licences.
   - The repo itself is not data-free:
     - 20 figures derived from the 2018 hand data are tracked under `reports/figures/real_data_v1/` and `real_data_v2/`;
     - `configs/real/*.yaml`, `scripts/real_v2.py` and `reports/real_data_v2.md` contain `/Users/gtravish/...` paths.
     - If insiders get repo access (not just the wheel), decide whether that is acceptable.
8. **Undeclared dependency.** `Pillow` is imported in `src/` (`io/emitter_array.py`, `io/viewer.py`) but not declared. It currently arrives through matplotlib. Declaring it needs Gil's approval (it is not on the CLAUDE.md list).

## 6. Steps and rough effort (inferred, one developer)

**Groundwork (both paths), about 4–6 days:**
1. Package fixes (about 1 day):
   - configs as package data via `importlib.resources`;
   - version and LEAP build string in outputs;
   - one-line error messages for manifest and LEAP problems;
   - declare Pillow;
   - add LICENSE files.
2. `dts selftest` and the "LEAP required" test mode (about 0.5 day).
3. `dts validate` (about 1 day).
4. `dts import` for image folder + geometry CSV, with an "already normalized" option (about 1.5–2 days).
5. A "your own data" user page and a worked example generated from `dts mock-scan` (about 1 day).

**Path A, wheel: about 2–3 days on top:**
1. Vendor LEAP v1.26 source and patches. Build `libleapct` in the wheel build, either with a small setuptools build step or with scikit-build-core (scikit-build-core is a new build-time dependency, so ask first).
2. macOS arm64:
   - build with `MACOSX_DEPLOYMENT_TARGET` set (for example 14.0);
   - `delocate-wheel` to bundle libomp;
   - check `otool -L` shows no `/opt/homebrew` paths;
   - test in a fresh venv, ideally on a second Mac.
3. Linux x86_64: build in a `manylinux_2_28` container, `auditwheel repair`, test in a plain Debian/Ubuntu container.
4. Publish: GitHub release assets in a private repo, or a shared drive. Install doc: `uv tool install dtsrecon-…whl` and `uv tool install 'dtsrecon[viz] @ file://…'`.

**Path B, Docker fallback: about 1–2 days on top:**
1. Two-stage Dockerfile: Debian + gcc + cmake builds patched LEAP, then a `python:3.12-slim` runtime. No Qt.
2. Build multi-arch images (`docker buildx`, linux/arm64 + linux/amd64), then run `dts selftest` and `pytest` inside each image.
3. Publish to a private registry (GHCR), or hand over a `docker save` tarball. Document the mount, memory (raise Docker Desktop's cap to ≥ 12 GB) and permissions.

**Later (not part of this plan):**
- the CUDA variant, after the GPU session validates LEAP's CUDA modular projector with the round-trip test;
- Windows native;
- a hosted service.

## 7. Open questions for Gil

**Revised order of work (2026-11-18):**
1. Groundwork.
2. A 10 × 10 grid simulation with ground truth.
3. The Linux wheel, tested in WSL2 on one real PC.
4. The Docker image if needed.
5. The macOS wheel for Gil.
6. CUDA: GPU validation on Milaan's machine (0.1 item 1), then a CUDA Linux build for users with NVIDIA GPUs.

**Still open:**
- Whether the other users have NVIDIA GPUs. Milaan has one (section 0).
- For Milaan (distribution, Python, gcc and cmake are now known from the hardinfo report):
  - output of `nvidia-smi` (does the driver load? one GPU or two?);
  - `nvcc --version`;
  - sudo rights;
  - whether git and uv or conda are installed;
  - whether the PC can reach GitHub;
  - whether they agree to run the GPU validation tests.
- Whether their PCs allow WSL2 or Docker Desktop (IT policy, admin rights).
- The details of their image files (4.3 item 1).
- Which GitHub organisation and repository to use.

**Status of the questions below** (answers in section 0):
- Answered: 4 (no licence), 5 (private company GitHub), 6 (approved).
- Partly answered: 1 (Windows PCs), 2 (regular grids, known geometry, raw TIFF from the detector).
- Partly answered (2026-11-19): 8. Milaan has a CUDA GPU on Linux; the other users are not known yet.
- Still open: 3, 7, 9, 10.

1. **Users and machines:** how many people, which operating systems (Apple Silicon Macs? Intel? Linux? Windows?), and are they comfortable with Python/uv, or would Docker suit them better? This decides whether the wheel or Docker comes first.
2. **Their data:** which detectors and file formats (TIFF, raw, DICOM)? Do they have darks and flats? How do they know their geometry: encoders, a CAD model, or a phantom? This decides what `dts import` must read.
3. **Repo or tool:** will insiders get the source repo, or only the built tool? (See 5.7: real-data figures and Gil's paths are in the repo.)
4. **Licence and terms** for dts-recon itself, and any confidentiality wording to ship with it.
5. **Hosting:** may we create a private GitHub remote, for CI, releases and a container registry? Or would you rather use files on a shared drive?
6. **New dependencies:** Pillow (runtime; already installed through matplotlib). Build-only tools: delocate, auditwheel, optionally cibuildwheel and scikit-build-core.
7. **Upstream LEAP:** report the two CPU bugs now (decisions items 2 and 6)? An upstream fix would remove the patch from packaging.
8. **GPU:** does any user need CUDA? If not, packaging stays CPU-only, and the GPU session is only needed for TIGRE.
9. **Legacy 2018 tools:** turn them into a `dts import --layout emitter-2018` command, or keep them as repo-only scripts?
10. **Support:** who answers users' questions, and how are updates announced?
