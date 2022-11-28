#!/bin/bash
make test/eulerSA.exe -j|| exit 1


NP=1
if [[ $# -gt 0 ]]; then
    NP=${1}
fi


mpirun.mpich -np ${NP} test/eulerSA.exe