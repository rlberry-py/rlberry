#!/bin/bash

NAME=$1
SCRIPT_PATH=${@:2}


usage="$(basename "$0") [-h]  NAME FILE1 FILE2...

FREEZE an environment and a script in a tar archive
the NAME is a string that will be used as name for the archive file
The Files must be a python files, glob accepted
"

Run(){
cat << EOF > get_rlberry_source.py
import rlberry
from pathlib import Path

rlberry_source = str(Path(rlberry.__file__).parents[1])
print(rlberry_source)
EOF

rlberry_source=$(python get_rlberry_source.py 2>&1)



FREEZE_PATH=freeze_dir

# ADDITIONAL_DEPS={}

echo "Creating the environment"

python3 -m venv $FREEZE_PATH

new_pip=$FREEZE_PATH/bin/pip3

echo "Installing rlberry and necessary deps"

$new_pip install wheel

if [[ -f "$rlberry_source/setup.py" ]]; then
    echo "using custom rlberry source"

    $new_pip install -r $rlberry_source/requirements.txt
    $new_pip install $rlberry_source
else
    echo "using last pip version rlberry"
    $new_pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[torch_agents]
fi

echo "Packing the env and creating archives"

venv-pack -p $FREEZE_PATH -o $FREEZE_PATH.tar.gz

python3 --version > python_version

tar -cf $NAME.tar $FREEZE_PATH.tar.gz $SCRIPT_PATH python_version

read -r -p "Do you want to clean the temporary files? (Y/N): " answer
if [ $answer = "Y" ]
then
    /bin/rm $FREEZE_PATH.tar.gz
    /bin/rm -r $FREEZE_PATH
    /bin/rm -r get_rlberry_source.py
    /bin/rm python_version
fi
}

case "$1" in
     -h) echo "$usage"
       exit
       ;;
     "") echo "Missing an argument"
       echo " "
       echo "$usage" >&2
       exit 1
       ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      echo "$usage" >&2
      exit 1
      ;;
     *)
     Run
      ;;
esac
