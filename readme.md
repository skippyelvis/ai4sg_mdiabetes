# Setup to Run Experiments
along with cloning this repo, you will need to download the relevant private files

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
- the code is expecting these variables to be defined in a `env` file
- for example, the GCS bucket names because they are public
- you need do the following
    - clone the repo
    - download the `env` file from the `mdiabetes-prod-analysis-data` bucket of the `mdiabetes-prod-analysis` Google Cloud project

### `arogya_content/` directory
- download from the `mdiabetes-prod-analysis-data` bucket
- contains message/question banks, questionnaires, etc

### `local_storage/` directory
- download the `uo-ai-storage` folder from the `participant-files` bucket and save locally as `local_storage`
- from the `uploads` folder of the same bucket, 
	- download all the `to_participants` files into a folder `local_storage/prod/outfiles`
	- download all the `participant_responses` files into a folder `local_storage/prod/responses`

### yaml input file
- each experiment has many hyperparameters which are defined in a `.yaml` file
- you can see examples in the `yaml/` directory

# Running Experiments
- confirm you have cloned the repo, install requirements, setup GCS, and made a `yaml` file
- to run a given experiment (specified by yaml) for 1 week:
    - `$ python3 mdiabetes.py -f input.yaml`
- *PRODUCTION RUN*: this is automated via the `systemd` service and timer units
    - verify success by checking for file in bucket
