#!/bin/bash

dirs="$(/bin/ls --indicator-style=none)"


for dir in $dirs; do
  if [ -d $dir ]; then
    echo $dir
    cd $dir
    mv $dir/* .
    rmdir $dir
    cd ..
  fi
done
