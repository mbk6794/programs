import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='It will draw a histrogram (number vs coupling constant')
parser.add_argument('-m', '--module', help='choose the module [pandas, numpy]')
parser.add_argument('-f', '--file', default=None, help='file name of .csv or .npz')
parser.add_argument('-mode','--mode',default=None)
args = parser.parse_args()

if args.module == 'pandas':
    molecule = pd.read_csv(args.file,sep=",")
    coupling = np.abs(np.array(molecule["Coupling(eV)"])*1000)

elif args.module == 'numpy':
    data = np.load(args.file)
    lst = data.files
    coupling = np.abs(data['C']*1000).reshape(-1)

if args.mode == 'log':
    coupling = np.log10(coupling / 1000)

medium_size = 10
bigger_size = 12

fig = plt.figure(figsize=(6,6))
plt.rc('font', size=bigger_size)
plt.title("TET coupling histogram")
if args.mode == 'log':
    plt.xlabel("log eV")
else:
    plt.xlabel("meV")
plt.ylabel("data")
#plt.ylim((0,2000))
plt.hist(coupling, bins=316)
if args.mode == 'log':
    plt.savefig("coupling_4_log.png")
else:
    plt.savefig("coupling_4.png")

with open("coupling_4",'w') as f:
    f.write("max: {:f}\n".format(max(coupling)))
    f.write("min: {:f}\n".format(min(coupling)))
print(np.where(coupling == max(coupling)))
