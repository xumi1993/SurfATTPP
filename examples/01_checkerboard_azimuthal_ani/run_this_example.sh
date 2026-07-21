#!/bin/bash

NPROC=8

mkdir -p OUTPUT_FILES
cp src_rec_file_rl_ph.csv OUTPUT_FILES/src_rec_file_forward_RL_PH.csv

# create checkers and forward simulate surface traveltimes
mpirun -np $NPROC ../../bin/SURFATT_cb_fwd -i input_params.yml -n 3/3/2 -a 2/2/2/135 -p 0.08/0.08 -m 0.2 -s 4

# inversion 
mpirun -np $NPROC ../../bin/SURFATT_tomo -i input_params.yml
