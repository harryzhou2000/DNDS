#!/bin/bash
make test/eikonal.exe -j|| exit 1


NP=1
if [[ $# -gt 0 ]]; then
    NP=${1}
fi


mpirun.openmpi -np ${NP} test/eikonal.exe