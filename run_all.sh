#!/bin/bash

controllers=("mpc")
robots=("helix" )
targets=("pos1" "pos2" "pos3" "pos4")

for robot in "${robots[@]}"; do
    for ctrl in "${controllers[@]}"; do
        for t in "${targets[@]}"; do
            python run.py --headless --experiment set --control_scheme $ctrl --robot $robot --target $t
        done

        if [ "$robot" != "spirob" ]; then
            python run.py --headless --experiment tracking --control_scheme $ctrl --robot $robot
        fi
    done
done
