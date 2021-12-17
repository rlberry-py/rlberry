# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Script used to construct the videos for the examples that output videos.
# Please use a script name that begins with video_plot as with the other examples
# and in the script, there should be a line to save the video in the right place
# and a line to load the video in the headers. Look at existing examples for
# the correct syntax. Be careful that you must remove the _build folder before
# recompiling the doc when a video has been updated/added.
video:
	 # Make videos
	 $(foreach file, $(wildcard ../examples/**/video_plot*.py), \
	 @echo $(basename $(notdir $(file)));\
	 python $(file)) ;\
	 # Make thumbnails
	 $(foreach file, $(wildcard _video/*.mp4), \
	 ffmpeg -y -i $(file) -vframes 1 -f image2 \
		 thumbnails/$(basename $(notdir $(file))).jpg ;\
		 )
	 # Remove unused metadata json
	 @rm _video/*.json

thumbnail_images:
	$(foreach file, $(wildcard _video/*.mp4), \
	ffmpeg -y -i $(file) -vframes 1 -f image2 \
		thumbnails/$(basename $(notdir $(file))).jpg ;\
		)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
