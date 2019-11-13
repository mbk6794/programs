import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dir')

args = parser.parse_args()
dirpath = args.dir
os.mkdir(dirpath)
#if os.path.isfile("amp.amp"):
#    os.system("mv amp.amp convergence.png amp-log.txt total.png w_1_1_1.png w_1_1_1_* total_hypo.txt total_labels.txt new_geo.extxyz w*md.traj w*md.txt w*md.xyz "+dirpath+"/.")
#elif os.path.isfile("amp-untrained-parameters.amp"):

os.system("mv *.amp convergence.png amp-log.txt total.png w_1_1_1.png w_1_1_1_* total_hypo.txt total_labels.txt new_geo.extxyz w*md.traj w*md.txt w*md.xyz "+dirpath+"/.")
os.system("cp xyz2qchem.py serialjob.py qchemsp2extxyz.py w_1_1_1.extxyz ./"+dirpath+"/.")

she = glob.glob("*sh.e*")
she.sort()
with open(she[-1],'r') as f:
    flines = f.readlines()
for i in range(len(flines)):
    if 'did not converge' in flines[i]:
        amp = False
    else:
        amp = True

os.chdir(dirpath)
if amp:
    os.system("cp amp.amp ../.")
else:
    os.system("cp amp-untrained-parameters.amp ../.")

os.chdir("..")
 
