#!/bin/bash

pushd ../python_scripts

while read args; do
    echo "python fit_plrt_model.py $args" #>> model_runner.txt
    pipenv run python fit_plrt_model.py $args
done < ../shell_scripts/runner_args.txt

popd
