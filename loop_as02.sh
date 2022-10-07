#!/bin/bash
#enemies=(4) #(4 1)


#for enemie in ${enemies[@]}
#do
#    for strategy in "phenotype" "genotype"
#    do
#        for i in {0..9}
#        do
status=-1
while [ ! $status -eq 0 ]
do
    python3 generalist_pymoo.py "pymoo_first_long_run"
    status=$?
    echo "bash done: current status $status"
done
#        done
#    done
#done

exit 0
	
