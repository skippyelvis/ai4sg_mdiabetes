# Files

### `mdiabetes.py`
- main experiment logic
- `main` function ran once per week
- logic:
    - load/compute participant states
    - group participants --> core, ai
    - generate weekly actions --> messages/questions
    - check weekly responses
    - update model
    - store new states, model, actions

### `model.py`
- neural network source code
- dqn agent source code

### `content.py`
- logic to handle content from arogya world
- `MessageHandler`: reads in message file from AW and creates action space
- `QuestionHandler`: reads in question file from AW and creates question space
- `Questionnaire`: reads baseline questionnaire files from AW
- `StatesHandler`: computes states for participants off questionnaires

### `storage.py`
- util class to store and load data
- also helpers to delete data after experiments 

### `logger.py`
- util class to log data to console

# Setup to Run Experiments

### Python requirements
- install the required packages from `environment.yaml`
- optionally, skip this step and install packages as necessary when you see error messages like `torch: no module found`

### GCS setup
- you will need a GCS developer access key in the form of a JSON file
- save this file on your computer, ex `/home/user/keys/mdiabetes-gcs-key.json`
- export the path to this file as `$GOOGLE_APPLICATION_CREDENTIALS`
    - for example, in you `~/.bashrc` or `~/.profile` add the following line:
        `export GOOGLE_APPLICATION_CREDENTIALS = $HOME/keys/mdiabetes-gcs-key.json`
- using developer keys is a security measure so no unknown/unauthenticated parties can write to our GCS buckets

### `.env` file
- there is some private data associated with the code that cannot be uploaded to git
- the code is expecting these variables to be defined in a `.env` file
- for example, the GCS bucket names because they are public
- this repository contains an `.env_sample` which exports two variables:
    - `MDIABETES_GCS_BUCKET_PRIVATE`: name of private bucket to store participant batches 
    - `MDIABETED_GCS_BUCKET_PUBLIC`: name of public bucket to store all other files
- you need do the following
    - clone the repo
    - rename `.env_sample` to `.env`: `ai4sg_mdiabetes$ cp .env_sample .env`
    - open `.env` and replace the bucket names with their real world names

### yaml input file
- each experiment has many hyperparameters which are defined in a `.yaml` file
- you can see examples in the `yaml/` directory
- important hyperparameters:
    - `dry_run`: whether or not to save the generated data
    - `storage.experiment`: the name of this experiment, and the folder in which we save all the data
        - remember to use different names for different experiments, otherwise they will overwrite
    - `dqn`: hyperparameters for dqn optimization
    - `model`: hyperparameters for model architecture

# Running Experiments
- confirm you have cloned the repo, install requirements, setup GCS, and made a `yaml` file
- to run a given experiment (specified by yaml) for 1 week:
    - `$ python3 mdiabetes.py -f input.yaml`
- to run a given experiment (specified by yaml) for 5 weeks:
    - `$ python3 mdiabetes.py -f input.yaml -n 5`
- if you want to continue this experiment for another 2 weeks, run
    - `$ python3 mdiabetes.py -f input.yaml -n 2`
- to run multiple different experiments at the same time
    - `$ python3 mdiabetes.py -f input1.yaml -n 3` will simulate 3 weeks of input1
    - `$ python3 mdiabetes.py -f input2.yaml -n 4` will simulate 4 weeks of input2
        - as long as input1 and input2 use different `storage.experiment` parameters, the
        data for both will be saved 

# Cleaning up experiments:
- locally: `$ rm -rf local_storage/storage_name`
- cloud: `$ python3 storage.py -c -f storage_name`
