#!/bin/bash




enemies=(4) #(4 1)


for enemie in ${enemies[@]}
do
    for strategy in "phenotype" "genotype"
    do
        for i in {0..9}
        do
            status=-1
            while [ ! $status -eq 0 ]
            do 
                python3 specialist_solution_v2.py "${enemie}_final_v2/${i}_${strategy}" $strategy $enemie
                status=$?
                echo "bash done: current status $status"
            done
        done
    done
done

exit 0
	
