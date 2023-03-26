#!/bin/bash

cd ../python_scripts

for depth in $(seq 1 6); do
    for mss in $(seq -f "%02g" 1 10); do
        echo "-d ${depth} --mss 0.${mss} --min-years 3 -p -S basin 0.8 --monthly --sim-all" >> psweep_progress.txt
        pipenv run python fit_plrt_model.py -d $depth --mss "0.$mss" --min-years 3 -p -S basin 0.8 --monthly --sim-all
    done
done
