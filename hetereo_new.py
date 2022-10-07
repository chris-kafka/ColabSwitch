import numpy as np
import os
from af import *
import re
import random
import argparse


parser1 = argparse.ArgumentParser(description='Hallucination Design')
parser1.add_argument('out_dir', type=str,help='Output location')
args1 = parser1.parse_args()
path_out = "/nfs2/data/chrisfrank/AD/Hallucination/"


dir = path_out + args1.out_dir

if not os.path.exists(dir):
    os.mkdir(dir)


import wandb
wandb.init(project="Hetero2", name=args1.out_dir[:-1])
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}






# setup the model
clear_mem()
model = mk_design_model(protocol="hetero")
print('Starting up and compiling JAX model....')
model.prep_inputs(length_1=100, length_2=100,rog=5)
#model.prep_inputs(length=100,copies=3)

for i in range(0,200):#
    name_it = ''.join(args1.out_dir[:-1] + '_' + str(i))
    wandb.init(project="hallucination1", name=name_it)
    model.restart(mode="soft")
    model.opt["weights"].update({"rog": 1.0, 'bh':0.0})
    model.design_logits(iters=100)




   # model.restart(mode="zeros")
   # model.opt["weights"].update({"pae":1.0,"plddt":1.0,"con":0.3})
   # model.opt["con"].update({"binary":True, "cutoff":21.6875, "num":model._len, "seqsep":0})
   # model.design_logits(iters=100)



    # define positions we want to constrain (input PDB numberi
    model.save_pdb(f"{dir}/out_hallo{i}.pdb",f"{dir}/out_hallo{i}.txt")
    model.plot_pdb()
    model.animate()