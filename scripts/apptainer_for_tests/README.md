
To test rlberry, you can use this script to create a container that install the latest version, run the tests, and send the result by email.
(or you can check inside the .sh file to only get the part you need)

:warning: **WARNING** :warning: : In both files, you have to update the paths and names

## .def
Scripts to build your apptainer.
2 scripts :
- 1 with the "current" version of python (from ubuntu:last)
- 1 with a specific version of python to choose

## .sh
Script to run your apptainer and send the report
use chmod +x [name].sh to make it executable

To run this script you need to install "mailutils" first (to send the report by email)
