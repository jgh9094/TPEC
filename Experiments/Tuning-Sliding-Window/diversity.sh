#!/bin/bash

# Base directory
BASE_DIR="/home/hernandezj45/Repos/GECCO-2026-TPEC/Experiments/Tuning"

# cd into RF directory
cd $BASE_DIR/RF
sbatch diversity.sb

# cd into DT directory
cd $BASE_DIR/DT
sbatch diversity.sb

# cd into ET directory
cd $BASE_DIR/ET
sbatch diversity.sb

# cd into GB directory
cd $BASE_DIR/GB
sbatch diversity.sb

# cd into KSVC directory
cd $BASE_DIR/KSVC
sbatch diversity.sb

# cd into LSGD directory
cd $BASE_DIR/LSGD
sbatch diversity.sb

# cd into LSVC directory
cd $BASE_DIR/LSVC
sbatch diversity.sb

echo "All jobs submitted!"