#!/bin/bash

NP=1
NSEE=1
if [[ $# -gt 0 ]]; then
    NP=${1}
fi
if [[ $# -gt 1 ]]; then
    NSEE=${2}
fi

bash backEuler.sh ${NP}

./watch.sh ${NSEE}