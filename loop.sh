#!/bin/bash

experiment_name='specialist_solution_test'

while [ ! -f $experiment_name"/neuroended" ]
do 
	python3 specialist_solution_v2.py
done

exit 0
	
