#!/bin/bash
set -e

NPROC=8

pos_str=19.5/-155.5
angle=-30

gmt grdcut @earth_relief_01m -R-157/-152/18/21 -Ghawaii.nc

../../bin/surfatt_rotate_src_rec -i src_rec_file_raw.csv -a $angle -c $pos_str -o src_rec_file_rotated.csv

../../bin/SURFATT_rotate_topo -i hawaii.nc -a $angle -c $pos_str -x -0.75/0.8 -y -0.75/1 -o hawaii_rotated.nc
