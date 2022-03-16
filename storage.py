from google.cloud import storage
import re
import os
from dotenv import load_dotenv
import csv
import torch
import shutil
import yaml
from logger import StorageLogger

load_dotenv()

class PathHandler:

    def __init__(self, bucket, bucket_folder, local_folder, folder_name, storage_name, 
            file_prefix, file_ext, filename_offset):
        self.bucket = bucket
        self.bucket_folder = bucket_folder
        self.folder_name = folder_name           # unique identifier of this experiment 
        self.storage_name = storage_name         # name of data to store in this folder
        self.file_prefix = file_prefix           # optional human-friendly file prefix
        if len(self.file_prefix) > 0:
            self.file_prefix += "_"
        self.file_ext = file_ext.strip('.')      # extension type
        self.filename_offset = filename_offset   # offset the starting number of files
        p = lambda *x: os.path.join(*x)
        self.local_base = local_folder
        self.local_folder = p(local_folder, self.folder_name[0], self.storage_name[0])
        os.makedirs(self.local_folder, exist_ok=True)
        self.cloud_folder = p(self.bucket_folder, self.folder_name[1], self.storage_name[1])

    def filename(self, local_or_cloud, index):
        fname = f"{self.file_prefix}{index}.{self.file_ext}"
        path = os.path.join(local_or_cloud, fname)
        return path

    def count_files(self, local_or_cloud):
        assert local_or_cloud in (self.local_folder, self.cloud_folder)
        if local_or_cloud == self.local_folder:
            return len(os.listdir(local_or_cloud))
        elif local_or_cloud == self.cloud_folder:
            cli = storage.Client()
            blobs = cli.list_blobs(self.bucket, prefix=local_or_cloud)
            blobs = list(blobs)
            N = len(blobs)
            return max(0, N)

def load_yaml(fname):
    try:
        loader = yaml.SafeLoader
    except:
        loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    data = yaml.load(open(fname, "r"), Loader=loader)
    return data

class Storage(PathHandler):

    def __init__(self, local, cloud, *args):
        super().__init__(*args)
        self.local = local
        self.cloud = cloud

    def save_data(self, data, index):
        if data is None:
            return
        localf = self.filename(self.local_folder, index)
        cloudf = self.filename(self.cloud_folder, index)
        if self.local:
            self.save_local(data, localf)
        if self.cloud:
            self.save_cloud(data, cloudf, localf)

    def load_indexed_data(self, index):
        localf = self.filename(self.local_folder, index)
        cloudf = self.filename(self.cloud_folder, index)
        if self.local and not self.cloud:
            # StorageLogger("loading local", localf)
            return self.load_local(localf)
        # StorageLogger("loading cloud", cloudf)
        return self.load_cloud(cloudf, localf)

    def save_local(self, data, fname):
        if self.file_ext == "pt":
            torch.save(data, fname)
        elif self.file_ext == "csv":
            with open(fname, "w") as fp:
                wr = csv.writer(fp)
                for row in data:
                    wr.writerow(row)
        elif self.file_ext == "yaml":
            shutil.copy(data, fname)

    def load_pt(self, fname):
        return torch.load(fname)

    @staticmethod
    def load_csv(fname):
        data = []
        with open(fname, "r") as fp:
            rd = csv.reader(fp)
            for row in rd:
                try:
                    row = [int(r) for r in row]
                    data.append(row)
                except:
                    pass
        return data

    def load_local(self, fname):
        if not os.path.exists(fname):
            return None
        if self.file_ext == "pt":
            return self.load_pt(fname)
        elif self.file_ext == "csv":
            return self.load_csv(fname)
        elif self.file_ext == "yaml":
            return load_yaml(fname)

    def save_cloud(self, data, cloud_fname, local_fname):
        if not self.local:
            os.makedirs(self.local_folder, exist_ok=True)
        cli = storage.Client()
        bucket = cli.bucket(self.bucket)
        blob = bucket.blob(cloud_fname)
        if not os.path.exists(local_fname):
            self.save_local(data, local_fname)
        blob.upload_from_filename(local_fname)
        if not self.local:
            shutil.rmtree(self.local_folder)

    def load_cloud(self, cloud_fname, local_fname):
        if not self.local:
            os.makedirs(self.local_folder, exist_ok=True)
        cli = storage.Client()
        bucket = cli.bucket(self.bucket)
        try:
            blob = bucket.blob(cloud_fname)
            blob.download_to_filename(local_fname)
            data = self.load_local(local_fname)
            if not self.local:
                shutil.rmtree(self.local_folder)
        except:
            data = None
            os.remove(local_fname)
        return data

    def count_files(self):
        f = self.cloud_folder
        if not self.cloud:
            f = self.local_folder
        return super().count_files(f)

    def delete_files(self):
        if self.local:
            shutil.rmtree(self.local_folder)
        if self.cloud:
            cli = storage.Client()
            blobs = cli.list_blobs(self.bucket, prefix=self.cloud_folder)
            for blob in blobs:
                blob.delete()

LOCAL_STORAGE_FOLDER = "local_storage"
PUBLIC_BUCKET = os.getenv("MDIABETES_GCS_BUCKET_PUBLIC")
PRIVATE_BUCKET = os.getenv("MDIABETES_GCS_BUCKET_PRIVATE")
CLOUD_STORAGE_FOLDER_AI = "uo-ai-storage"
CLOUD_STORAGE_FOLDER_PROD = "uploads"
CLOUD_STORAGE_FOLDER_BATCHES = "Batches"
CLOUD_STORAGE_FOLDER_RESPONSES = "uploads"
LOCAL = True
CLOUD = True

def make_storage_group(experiment="preprod", local=LOCAL, cloud=CLOUD):
    # participant state vectors
    StateStor = Storage(
            local,    # store data locally? 
            cloud,    # store data on cloud?
            PUBLIC_BUCKET,   # gcs bucket to use
            CLOUD_STORAGE_FOLDER_AI,   # top-level folder in bucket
            LOCAL_STORAGE_FOLDER,           # top-level local folder
            [experiment, experiment], # middle-level folder on [local, cloud]
            ["states", "states"],     # lowest-level folder on [local, cloud]
            "",      # prefix for filenames, ex "batch", postfixed with "_"
            ".pt",   # file extension to save as
            0        # filename offset
    )
    # participant IDs
    IDStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI, 
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["ids", "ids"], "", ".pt", 0)
    # participant cluster
    ClusterStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI, 
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["clusters", "clusters"], "", ".pt", 0)
    # weekly (ID, action) mapping
    ActionStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["actions", "actions"], "", ".pt", 0)
    # timeline of participants (ID, weeks_since_join)
    TLStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["timeline", "timeline"], "", ".pt", 0)
    # dqn model weights, memory
    DQNStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["dqn", "dqn"], "", ".pt", 0)
    # debugging info each week (ai metrics)
    DebugStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["debug", "debug"], "", ".pt", 0)
    # where to store yaml hyperparameter file
    YamlStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_AI,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["yaml", "yaml"], "", ".yaml", 0)
    # incoming participant (ID, phone) mapping
    BatchStor = Storage(local, cloud, PRIVATE_BUCKET, CLOUD_STORAGE_FOLDER_BATCHES,
            LOCAL_STORAGE_FOLDER, [experiment, ""], ["batch", ""], "batch", ".csv", 0)
    # weekly participant responses (ID, q1, r1, q2, r2)
    RespStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_RESPONSES,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["responses", ""], "participant_responses_week", ".csv", 0)
    # weekly outbound message/question file (ID, m1, q1, m2, q2)
    MsgQsnStor = Storage(local, cloud, PUBLIC_BUCKET, CLOUD_STORAGE_FOLDER_RESPONSES,
            LOCAL_STORAGE_FOLDER, [experiment, experiment], ["outfiles", ""], "to_participants_week", ".csv", 0)
    Stor = {
            "states": StateStor,
            "ids": IDStor,
            "clusters": ClusterStor,
            "actions": ActionStor,
            "timelines": TLStor,
            "dqns": DQNStor,
            "debugs": DebugStor,
            "batches": BatchStor,
            "responses": RespStor,
            "outfiles": MsgQsnStor,
            "yaml": YamlStor
            }
    return Stor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="storage folder", default="preprod")
    parser.add_argument("-c", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-t", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(args)

    Stor = make_storage_group(args.f)

    # for testing, remove all stored files
    if args.c:
        if args.y or input("clean gcs? (y/n) >  ").lower() == "y":
            for k, v in Stor.items():
                if k in ["batches", "responses", "outfiles"]:
                    continue
                v.delete_files()

    # for testing, upload sample file to all buckets
    if args.t:
        file = [[1,2], [3,4], [5,6]]
        filet = torch.tensor(file)
        for k, v in Stor.items():
            if v.file_ext == "pt":
                v.save_data(filet, 0)
            elif v.file_ext == "csv":
                v.save_data(file, 0)
