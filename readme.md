# Setup to Run Experiments

### GCS setup
- you will need a GCS developer access key in the form of a JSON file
- download the `json` key file from the `mdiabetes-prod-analysis-data` bucket
- save this file on your computer, ex `/home/user/keys/mdiabetes-gcs-key.json`
- export the path to this file as `$GOOGLE_APPLICATION_CREDENTIALS`
    - for example, in you `~/.bashrc` or `~/.profile` add the following line:
        `export GOOGLE_APPLICATION_CREDENTIALS=$HOME/keys/mdiabetes-gcs-key.json`
- using developer keys is a security measure so no unknown/unauthenticated parties can write to our GCS buckets

### `env` file
- there is some private data associated with the code that cannot be uploaded to git
- the code is expecting these variables to be defined in a `.env` file
- for example, the GCS bucket names because they are public
- this repository contains an `.env_sample` which exports two variables:
    - `MDIABETES_GCS_BUCKET_PRIVATE`: name of private bucket to store participant batches 
    - `MDIABETED_GCS_BUCKET_PUBLIC`: name of public bucket to store all other files
- you need do the following
    - clone the repo
    - download the `env` file from the `mdiabetes-prod-analysis-data` bucket of the `mdiabetes-prod-analysis` Google Cloud project

### `arogya_content/` directory
- download from the `mdiabetes-prod-analysis-data` bucket
- contains message/question banks, questionnaires, etc

### yaml input file
- each experiment has many hyperparameters which are defined in a `.yaml` file
- you can see examples in the `yaml/` directory

# Running Experiments
- confirm you have cloned the repo, install requirements, setup GCS, and made a `yaml` file
- to run a given experiment (specified by yaml) for 1 week:
    - `$ python3 mdiabetes.py -f input.yaml`
- *PRODUCTION RUN*: this is automated via the `systemd` service and timer units
    - verify success by checking for file in bucket
