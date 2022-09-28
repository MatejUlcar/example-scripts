## e1

This example contains:
* `prepare-nvidia-hf-container.def` script to build a singularity container, based on the nvidia's pytorch docker; huggingface transformer library is additionally installed in this script.
* `maister.xrsl` xrsl script used to submit a job via ARC client, in this case to the maister HPC server; it contains all the training settings.
* `maister-train.sh` bash executable script that is run by the server; all the training is started by this script, it executes the singularity container from which it runs the training itself.
* `ds_config.json` a basic configuration file for deepspeed.

