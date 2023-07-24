#!/bin/bash

# Script used to construct the videos for the examples that output videos.
# Please use a script name that begins with video_plot as with the other examples
# and in the script, there should be a line to save the video in the right place
# and a line to load the video in the headers. Look at existing examples for
# the correct syntax. Be careful that you must remove the _build folder before
# recompiling the doc when a video has been updated/added.


for f in ../examples/video_plot*.py ;
do
# construct the mp4
python $f
name=$(basename $f)
# make a thumbnail. Warning : the video should have the same name as the python script
# i.e. video_plot_SOMETHING.mp4 to be detected
ffmpeg -i ../docs/_video/${name%%.py}.mp4  -vframes 1 -f image2 ../docs/thumbnails/${name%%.py}.jpg
done
