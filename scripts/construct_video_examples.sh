#!/bin/bash

# Script used to construct the videos for the examples that output videos.
# Please use a script name that begins with video_plot as with the other examples
# and in the script, there should be a line to save the video in the right place
# and a line to load the video in the headers. Look at existing examples for
# the correct syntax.


for f in ../examples/video_plot*.py ;
do
python $f
done
