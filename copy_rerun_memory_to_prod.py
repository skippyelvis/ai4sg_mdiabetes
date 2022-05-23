from torch import load as L, save as S
import os

path = "/home/users/jwolf5/mdiabetes/PROD/local_storage"
prod_path = f"{path}/prod/dqn"
rern_path = f"{path}/behavior_dataset/dqn"

for file in os.listdir(rern_path):
    # load up the rerun memory and prod memory
    prodf = f"{prod_path}/{file}"
    rernf = f"{rern_path}/{file}"
    rern = L(rernf)
    prod = L(prodf)
    # only fix the necessary data
    if 'reward' not in rern['0']['memory']:
        continue
    # now copy memory['reward'] from renr-->prod
    for k in ['0', '1', '2']:
        old = prod[k]['memory']['reward'].clone()
        prod[k]['memory']['reward'] = rern[k]['memory']['reward'].clone()
    # then save prod
    print(prodf)
    S(prod, prodf)
