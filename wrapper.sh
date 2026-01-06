#!/bin/bash

cd "$(dirname "$0")"

echo "--- Hostname Information ---"
hostname
echo "--------------------------"
echo "Process ID: $$"

export OMP_NUM_THREADS=2

command="./pddrive_spawn"
if [ ! -f $command ] ; then
    echo "Error: pdqrdriver binary not found!"
    exit 1
fi
echo "Executing command: $command $*"
$command $*
#command="./hello"
#$command