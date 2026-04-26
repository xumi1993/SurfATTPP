#!/bin/bash

NPROC=8

mkdir -p OUTPUT_FILES
cp src_rec_file_ph.csv OUTPUT_FILES/src_rec_file_forward_RL_PH.csv
cp src_rec_file_ph.csv OUTPUT_FILES/src_rec_file_forward_LV_PH.csv
cp src_rec_file_gr.csv OUTPUT_FILES/src_rec_file_forward_RL_GR.csv
cp src_rec_file_gr.csv OUTPUT_FILES/src_rec_file_forward_LV_GR.csv

# Create 2x3x2 radial anisotropy checkerboards with Vs and zeta perturbations
# -r: enable radial anisotropy mode
# -p 0.08/0.1: Vs perturbation=0.08, zeta perturbation=0.1
# -m 0.2: margin between anomalies
mpirun -np $NPROC ../../bin/SURFATT_cb_fwd -i input_params.yml -n 2/3/2 -r -m 0.2 -p 0.08/0.1

# inversion
mpirun -np $NPROC ../../bin/SURFATT_tomo -i input_params.yml
