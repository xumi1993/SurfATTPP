# Contributing to SurfATTPP

Thank you for your interest in contributing! This document explains how to report issues, propose changes, and submit code.

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Building and Testing](#building-and-testing)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Reporting Bugs

Please open an issue on [GitHub](https://github.com/xumi1993/SurfATTPP/issues) and include:

- A minimal, self-contained reproducer (input parameter file, dataset size, command line)
- The full error message or stack trace
- Your platform, compiler version, MPI implementation, and HDF5 version
- Whether the problem is reproducible with a single MPI rank

---

## Requesting Features

Open an issue labelled **enhancement** with:

- A clear description of the use-case
- References to relevant literature if the request is algorithm-related
- Whether you are willing to implement it (highly appreciated)

---

## Development Workflow

The repository uses two long-lived branches:

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases only |
| `devel` | Active development — target all PRs here |

1. Fork the repository on GitHub.
2. Clone your fork and add the upstream remote:
   ```bash
   git clone https://github.com/<your-username>/SurfATTPP.git
   cd SurfATTPP
   git remote add upstream https://github.com/xumi1993/SurfATTPP.git
   ```
3. Create a feature branch off `devel`:
   ```bash
   git fetch upstream
   git checkout -b feature/my-feature upstream/devel
   ```
4. Make your changes, commit, and push to your fork.
5. Open a pull request targeting `devel`.

---

## Code Style

SurfATTPP is written in **C++20**. Please follow the conventions already present in the codebase:

- **Naming**: `snake_case` for variables, functions, and files; `PascalCase` for classes.
- **Header guards**: `#pragma once` (not `#ifndef` guards).
- **Formatting**: 4-space indentation, no tabs. Keep lines under ~100 characters where practical.
- **MPI safety**: any code that writes to shared-memory arrays (`vs3d`, `vp3d`, …) must be guarded by `if (mpi.is_main())` and followed by `mpi.barrier()` + `mpi.sync_from_main_rank(...)`.
- **INVERSION_MODE guard**: sensitivity tensors (`sen_vs_loc`, `sen_gc_loc`, …) are only allocated in `INVERSION_MODE`. Do not access them unconditionally.
- **No `std::cout` / `exit()` in library code**: use `ATTLogger::logger()` for messages and `mpi.abort()` for fatal errors.
- **Precision agnosticism**: use `real_t`, `_0_CR`, `_1_CR` instead of literal `0.0` / `1.0` so the code compiles correctly in both single- and double-precision modes.

---

## Building and Testing

### Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Optional CMake flags:

| Flag | Default | Effect |
|------|---------|--------|
| `-DUSE_SINGLE_PRECISION=ON` | OFF | Build with `float` instead of `double` |
| `-DBUILD_TESTS=ON` | OFF | Build unit-test executables under `tests/` |

### Unit tests

```bash
cmake -DBUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

### Integration tests

Integration tests live in a separate repository ([SurfATT-tests](https://github.com/xumi1993/SurfATT-tests)) and are run automatically by the [CI workflow](.github/workflows/test.yml) on every push and pull request to `devel`.

To run them locally:

```bash
git clone https://github.com/xumi1993/SurfATT-tests.git ../SurfATT-tests
export SURFATT_BIN=$(realpath bin)
cd ../SurfATT-tests/testcase01
bash run.sh
```

All nine test cases (`testcase01`–`testcase09`) must pass before a PR is merged.

---

## Submitting a Pull Request

- Keep PRs focused — one logical change per PR.
- Reference the related issue number in the PR description (`Closes #123`).
- All CI checks (build on macOS & Linux, nine integration tests) must be green.
- Add a brief summary of what changed and why in the PR description; include any noteworthy algorithmic decisions.
- For new executables, update `README.md` with the new binary name, description, and usage example.
