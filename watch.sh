#!/bin/bash

NSEE=1
if [[ $# -gt 0 ]]; then
    NSEE=${1}
fi


watch -cn${NSEE} tail -n20 log.txt 
