#!/bin/bash

# Modifying working directory
cd parallel-processing/hw4
echo "Working directory is:"
pwd
echo ""

mpirun -npernode 8 ./get_data