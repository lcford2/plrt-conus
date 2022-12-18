#!/bin/bash

function my_dot () {
    # echo $1
    # echo $1 | sed -e 's/dot/png/'
    png_file="$(echo $1 | sed -e 's/\.dot/\.png/')"
    dot -Tpng $1 -o $png_file
}

export -f my_dot

if [ "$1" == "all" ]; then
    find ../results/ -name '*.dot' -exec /bin/bash -c 'my_dot "$0"' {} \;
else
    find $1 -name '*.dot' -exec /bin/bash -c 'my_dot "$0"' {} \;
fi
