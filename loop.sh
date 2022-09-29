#!/bin/bash

experiment_prefix='4_final/'
experiment_suffix='_genotype'

for i in {0..9}
do
    status=-1
    while [ ! $status -eq 0 ]
    do 
        python3 specialist_solution_v2.py "$experiment_prefix$i$experiment_suffix"
        status=$?
        echo "bash done: current status $status"
    done
done

exit 0
	
