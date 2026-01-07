#!/bin/bash

cd "$(dirname "$0")"

echo "--- Hostname Information ---"
hostname
echo "--------------------------"
echo "Process ID: $$"

last_arg="${@: -1}"
result="${*:1:$(( $# - 1 ))}"


export OMP_NUM_THREADS=$last_arg
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

command="./build/EXAMPLE/pddrive_spawn"
if [ ! -f $command ] ; then
    echo "Error: pdqrdriver binary not found!"
    exit 1
fi
echo "Executing command: $command $result"
$command $result
