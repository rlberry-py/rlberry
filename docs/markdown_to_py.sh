#!/usr/bin/env -S bash


mkdir python_scripts

set -e

shopt -s globstar
list_files=$(ls $PWD/**/*.md)

for script in $list_files; do
    echo "Processing " $script
    sed -n '/^```python/,/^```/ p' < $script | sed '/^```/ d' > python_scripts/${script##*/}.py
done

# read -p "Do you wish to execute all python scripts? (y/n)" yn
# case $yn in
#     [Yy]* ) for f in python_scripts/*.py; do python3 $f ; done ;;
#     * ) exit;;
# esac
