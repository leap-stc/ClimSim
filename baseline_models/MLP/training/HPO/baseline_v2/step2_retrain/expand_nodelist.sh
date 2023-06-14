#!/bin/bash

# Define the node array
slurm_nodes=$1

# remove non-number characters
# then, pack it into an array
slurm_nodes=($(echo $slurm_nodes | tr -dc '0-9,-' | tr ',' ' ')) 	 

# Use a loop to generate the node numbers
expanded=""
for node in "${slurm_nodes[@]}"; do
  if [[ $node == *-* ]]; then
    start="$(echo $node | cut -d'-' -f1)"
    end="$(echo $node | cut -d'-' -f2)"
    node=`eval "echo {$start..$end}"`
    expanded="$expanded $node"
  else
    expanded="$expanded $node"
  fi
done

expanded=($expanded)
expanded=(${expanded[@]/#/nid})
echo ${expanded[@]} | sed 's/ /\n/g'

