# Checkerboard Test for Radial Anisotropy

This example demonstrates the checkerboard forward simulation and inversion for **radial anisotropy** models.

## Key Features

- **Model Type**: Radial anisotropy (MODEL_RADIAL_ANI, model_para_type=2)
- **Parameters**: vsv (vertical S-wave velocity) and vsh (horizontal S-wave velocity)
- **Derived Parameters**: 
  - Vs = √((2vsv + vsh)/3) — RMS S-wave velocity
  - zeta = vsh²/vsv² — Radial anisotropy strength
- **Checkerboard**: 2×3×2 anomalies in x, y, z directions

## Running the Example

```bash
./run_this_example.sh
```

This will:
1. Create 2×3×2 radial anisotropy checkerboard perturbations
   - Vs perturbation: 0.08 (8%)
   - zeta perturbation: 0.1 (10%)
   - Margin between anomalies: 0.2°
2. Forward simulate Rayleigh-wave travel times
3. Run inversion to recover the model

## Configuration

### Forward Simulation Command
```bash
mpirun -np 8 ../../bin/surfatt_cb_fwd -i input_params.yml -n 2/3/2 -r -m 0.2 -p 0.08/0.1
```

- `-n 2/3/2`: Number of checkerboard anomalies in x, y, z
- `-r`: Enable radial anisotropy mode
- `-m 0.2`: Margin in degrees
- `-p 0.08/0.1`: Perturbations (Vs, zeta)

### Inversion Parameters
- Model type: Radial anisotropy (model_para_type=2)
- Max iterations: 40
- Optimization: LBFGS with line search
- Data: Rayleigh-wave phase velocity

## Output Files

Generated in `OUTPUT_FILES/`:
- `initial_model.h5`: Initial model with vsv and vsh
- `model_*.h5`: Inverted models with vsv, vsh, Vs, and zeta
- Sensitivity kernels for radial anisotropy parameters
- Inverted checkerboard recovery

## Differences from Isotropic Case (00_checkerboard_iso)

| Aspect | Isotropic | Radial Anisotropy |
|--------|-----------|-------------------|
| Model type | model_para_type=0 | model_para_type=2 |
| Parameters | Single Vs | vsv, vsh (or Vs, zeta) |
| CB flag | None | `-r` |
| Perturbation | `-p pert_vel` | `-p pert_vs/pert_zeta` |
| Output | vs | vsv, vsh, Vs, zeta |

## Comparison with Azimuthal Anisotropy (01_checkerboard_azi_ani)

- **Azimuthal** (`-a`): Varies with fast-axis direction in horizontal plane
- **Radial** (`-r`): Varies in depth, controlled by vsh/vsv ratio
- Cannot use `-a` and `-r` together

## Notes

- The radial anisotropy mode reuses the same checkerboard pattern for Vs perturbations
- zeta perturbations are applied independently
- The model is saved with all four fields: vsv, vsh, Vs, and zeta for analysis
