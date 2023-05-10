#!/bin/bash
#SBATCH --partition=paratera
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

make test/euler.exe -j
