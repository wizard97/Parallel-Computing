#!/bin/bash

#number of nodes in cluster
numnodes=7
kstart=8
kend=16
numcores=16
r=1

# compile it first
mpicc -std=gnu99 -O3 -Wall -fopenmp daw268_hw5.c -lm -o daw268_hw5

for i in $(seq 1 $numnodes); do
    for j in $(seq 1 $numcores); do
        echo "Running: mpirun -mca plm_rsh_no_tree_spawn 1 -np $i -use-hwthread-cpus -hostfile hostfile --map-by node:PE=$j daw268_hw5 $j $r $kstart $kend"
        mpirun -mca plm_rsh_no_tree_spawn 1 -np $i -use-hwthread-cpus -hostfile hostfile --map-by node:PE=$j daw268_hw5 $j $r $kstart $kend
    done
done
