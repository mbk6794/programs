
# coding: utf-8

# In[1]:


import numpy as np
import glob
import os


# In[2]:


infiles = glob.glob('*.in')
infiles.sort()
outfiles = []
for name in infiles:
    outfiles.append(name[:-2]+'out')


# In[3]:


leninfiles = len(infiles)
n_atoms_per_molecule = 10
n_atoms_per_cluster = 2 * n_atoms_per_molecule
atomic_numbers = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10}
CM = np.zeros((leninfiles,n_atoms_per_cluster,n_atoms_per_cluster))


# In[8]:


symbols = np.zeros(n_atoms_per_cluster)
xyz = np.zeros((leninfiles,n_atoms_per_cluster, 3))

for i in range(leninfiles):
    with open(infiles[i],'r') as f:
        flines = f.readlines()
    position = flines[2:2+n_atoms_per_cluster]
    for j in range(len(position)):
        position[j] = position[j].split(' ')
        while '' in position[j]:
            position[j].remove('')
        symbols[j] = atomic_numbers[position[j][0]]
        xyz[i][j] = position[j][1:4]


# In[10]:


import math
#lenCV = int(n_atoms_per_molecule ** 2)
lenCV = int(n_atoms_per_cluster * (n_atoms_per_cluster + 1) * 0.5)
CV = np.zeros((leninfiles, lenCV))

for sample in range(leninfiles):
    cnt=0
    for row in range(0, n_atoms_per_cluster):
        for col in range(row, n_atoms_per_cluster):
            if row == col:
                CM[sample][row][col] = 0.5 * pow(symbols[row],2.4)
                CV[sample][cnt] = CM[sample][row][col]
                cnt+=1
            elif row < col:
                CM[sample][row][col] = (symbols[row] * symbols[col]) / (pow(pow(xyz[sample][row][0]-xyz[sample][col][0],2)+pow(xyz[sample][row][1]-xyz[sample][col][1],2)+pow(xyz[sample][row][2]-xyz[sample][col][2],2),0.5))
                CV[sample][cnt] = CM[sample][row][col]
                cnt+=1

CVscaling = CV / np.amax(CV)


# In[12]:


coupling = []
for name in outfiles:
    with open(name, 'r') as f:
        flines = f.readlines()
    for idx in range(len(flines)):
        if 'Coupling(eV)' in flines[idx]:
            line = flines[idx+2]
            break
    cp = line.split(' ')
    while '' in cp:
        cp.remove('')
    cp = eval(cp[5])
    coupling.append(cp)

from itertools import combinations
comb = []
for i in range(0, n_atoms_per_cluster):
    for j in range(i, n_atoms_per_cluster):
        comb.append((i+1,j+1))



f=open('complete_CoulombVector_Coupling','w')
size = len(comb)

for i in range(size):
    for j in [0,1]:
        if j == 0:
            f.write('{:2d}'.format(comb[i][j])+'-')
        else:
            f.write('{:2d}'.format(comb[i][j])+',')
f.write('Coupling(eV)')
f.write('\n')
for i in range(leninfiles):
    for j in range(lenCV):
        f.write('{:.7f}'.format(CVscaling[i][j])+',')
    f.write(str(coupling[i]))
    f.write('\n')
f.close()

