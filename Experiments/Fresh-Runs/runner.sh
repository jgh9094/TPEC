#!/bin/bash

# Base directory
BASE_DIR="/home/hernandezj45/Repos/GECCO-2026-TPEC/Experiments/Fresh-Runs"

# cd into RF directory
cd $BASE_DIR/RF
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into DT directory
cd $BASE_DIR/DT
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into ET directory
cd $BASE_DIR/ET
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into GB directory
cd $BASE_DIR/GB
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into KSVC directory
cd $BASE_DIR/KSVC
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into LSGD directory
cd $BASE_DIR/LSGD
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

# cd into LSVC directory
cd $BASE_DIR/LSVC
sbatch bo.sb
sbatch p0.sb
sbatch p25.sb
sbatch p50.sb
sbatch p5.sb
sbatch p10.sb

echo "All jobs submitted!"