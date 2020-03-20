import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='randomly pick from original CM')
parser.add_argument('-f','--fin',help='csv file name')
parser.add_argument('-nt','--ntotal',type=int,help='how many data will be need')
parser.add_argument('-nf','--nfile',type=int,help='if you consider symmetry of molecule, how many files will be created')

args = parser.parse_args()

num = int(args.ntotal / args.nfile)

molecule_dataframe = pd.read_csv(args.fin, sep=",")
molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
molecule_dataframe = molecule_dataframe.head(num)

header = molecule_dataframe.keys()
molecule_dataframe.to_csv("pick.csv",sep=',',header=header,index=False) 
