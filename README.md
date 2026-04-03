# SurfATTPP

[![Language](https://img.shields.io/badge/-C++-00599C?logo=cplusplus&logoColor=white)](https://github.com/topics/cpp)
[![License](https://img.shields.io/github/license/xumi1993/seispy)]()
[![Build SurfATT](https://github.com/xumi1993/SurfATT-iso/actions/workflows/build.yml/badge.svg)](https://github.com/xumi1993/SurfATT-iso/actions/workflows/build.yml)

This is an innovative package for **Surf**ace wave **A**djoint **T**ravel-time **T**omography written in modern C++20 with highlights:

- Calculation of surface wave travel time based on **Eikonal equation** with fast sweeping method ([Tong, 2021a](https://doi.org/10.1029/2021JB021818))
- Computation of sensitivity kernels through **adjoint method** ([Tong, 2021b](https://doi.org/10.1029/2021JB022365))
- **Multi-grid model parametrization** utilization in optimization ([Tong et al., 2019](https://doi.org/10.1093/gji/ggz151))
- Consideration of **surface topography** ([Hao et al., 2024a](https://doi.org/10.1029/2023JB027454))
- **MPI parallelism** with shared-memory windows for large-scale distributed computing
- **HDF5** model I/O with parallel read/write support
- Configurable **single or double precision** at compile time

## Gallery

### Travel time field and sensitivity kernel on curved surface ([Hao et al., 2024a](https://doi.org/10.1029/2023JB027454))
![jgrb56585-fig-0001-m](https://github.com/xumi1993/SurfATT-iso/assets/7437523/49e205a3-7529-4079-a8c2-471c6e7075fc)
-------

### Tomographic results of S-wave velocity beneath Hawaii Island
![Fig2](https://github.com/xumi1993/SurfATT-iso/assets/7437523/f9a0155b-7b83-4970-914d-f13dc42b11e5)

## Dependencies

| Library | Version | Role |
|---------|---------|------|
| MPI | — | Parallelism (required) |
| HDF5 | ≥ 1.10 | Model I/O (required) |
| Eigen | ≥ 3.4 | Linear algebra (required) |
| yaml-cpp | ≥ 0.7 | Parameter file parsing (bundled) |
| spdlog | ≥ 1.10 | Logging (bundled) |

## Installation

Please refer to the [installation guide](https://surfatt.xumijian.me/installation/dependence.html) for detailed instructions.

### Build from source

```bash
mkdir build && cd build
cmake .. && make -j
```

Use `-DUSE_SINGLE_PRECISION=ON` to build with single-precision floats (default: double).

## Executables

| Executable | Description |
|------------|-------------|
| `bin/SURFATT_tomo` | Main inversion / forward modelling |
| `bin/SURFATT_rotate_src_rec` | Rotate source-receiver CSV to local coordinates |
| `bin/SURFATT_rotate_topo` | Rotate topography NetCDF grid to local coordinates |

## How to use SurfATTPP

The main executable `bin/SURFATT_tomo` for inverting surface dispersion data for S-wave velocity is run with `mpirun` as:

```bash
mpirun -np 4 bin/SURFATT_tomo -i input_params.yml
```

### A quick example

A case named `examples/00_checkerboard_iso` presents an example of inversion for 2×3×2 checkers using ambient noise surface wave data from 25 stations. Execute `run_this_example.sh` to run this example under 8 processors:

```bash
cd examples/00_checkerboard_iso
bash run_this_example.sh
```

