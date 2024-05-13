#!/bin/bash

# The number of data entries in your Python script
NUM_ENTRIES=90
# Loop through each index and run the Python script
for ((i=0; i<$NUM_ENTRIES; i++))
do
    python execute_sim.py $i --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error > log_data/sim_$i.txt 2>&1
done
