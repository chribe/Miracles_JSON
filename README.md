This file is a first version of the json file which should be used for the file writer to create nexus data for miracles.

To test modifications, use:

- https://scipp.github.io/chexus/
- https://github.com/ess-dmsc/kafka-to-nexus/tree/main
- https://github.com/chribe/jsondisplay

The functions in mcstas2json.py have been implemented using AI. The test.ipynb converts the mcstas file in part of the json structure (sample_relative_components.json), which has then to be copied into the main structure to obtain the full file.

The coordinates in this version are all directely depending on the sample position. No dependency chains are implemented.
