#!/bin/bash

# Set the email recipient and subject
recipient="email1@inria.fr,email2@inria.fr"
attachment="test_report.txt"

# Build, then run, the Apptainer container and capture the test results
cd [Path_to_your_apptainer_folder]
#build the apptainer from the .def file  into an .sif image
apptainer build --fakeroot --force rlberry_apptainer_base.sif rlberry_apptainer_base.def
#Run the "runscript" section, and export the result inside a file (named previously)
apptainer run --fakeroot --overlay my_overlay/ rlberry_apptainer.sif > "$attachment"


# Send the test results by email
exit_code=$(cat [path]/exit_code.txt)  # Read the exit code from the file

if [ $exit_code -eq 0 ]; then
    # Initialization when the exit code is 0 (success)
    subject="Rlberry : Succes Monthly Test Report"
    core_message="Success. Please find attached the monthly test report."
else
    # Initialization when the exit code is not 0 (failed)
    subject="Rlberry : Failed Monthly Test Report"
    core_message="Failed. Please find attached the monthly test report."
fi

echo "$core_message" | mail -s "$subject" -A "$attachment" "$recipient"  -aFrom:"Rlberry_Monthly_tests"

