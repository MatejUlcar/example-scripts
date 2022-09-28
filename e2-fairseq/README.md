## e2

This example contains:
* `fairseq-container.def` script to build a singularity container, based on the nvidia's cuda docker; fairseq library is additionally installed.
* `train.xrsl` xrsl script used to submit a job via ARC client, in this case to the vega HPC server; it contains all the training settings.
* `train.sh` bash executable script that is run by the server; all the training is started by this script, it executes the singularity container from which it runs the training itself.


