#!/bin/bash

# The number of data entries in your Python script
NUM_ENTRIES=90
# Loop through each index and run the Python script
for ((i=0; i<$NUM_ENTRIES; i++))
do
    python execute_gpt.py $i > log_data/gpt_$i.txt 2>&1
done