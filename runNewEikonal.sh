#!/bin/bash
make test/eikonal.exe -j|| exit 1


NP=1
if [ $# == 1 ]; then
    NP=${1}
fi


mpirun -np ${NP} test/eikonal.exe