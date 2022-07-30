#!/bin/bash
make test/euler.exe -j|| exit 1


NP=1
if [[ $# -gt 0 ]]; then
    NP=${1}
fi


mpirun.openmpi -np ${NP} test/euler.exe