#!/bin/bash

usage="$(basename "$0") [-h] FILE

Unfreeze a frozen environment and a script
The FILE must be a tar file constructed using freeze.sh
"
FROZEN_FILE=$1
FREEZE_PATH=freeze_dir

Run(){

script=$(tar -tf test.tar | grep -v freeze_dir)


echo "decompressing the frozen env"


tar -xf $FROZEN_FILE >/dev/null 2>&1
mkdir env_dir
tar -zxvf $FREEZE_PATH.tar.gz -C env_dir >/dev/null 2>&1
/bin/rm $FREEZE_PATH.tar.gz

echo "the unfrozen scripts are " $script
echo "the unfrozen virtual env is env_dir"

echo "you can activate the environment with 'source env_dir/bin/activate'"

source env_dir/bin/activate
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
