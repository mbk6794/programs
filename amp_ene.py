#!/home/joonho/anaconda3/bin/python

import argparse

import ase.io
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from my_mplot2d import *
from my_chem import *
import os

fig = plt.figure(figsize=(15,10))
#ax = plt.axes()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

def draw(fdata, y, h):
    fdata = fdata[:-7]
    nlen = len(y)
    h_conv = np.array(h) * ev2kj 
    y_conv = np.array(y) * ev2kj 
    diff =  np.subtract(h_conv,y_conv) 
    rmse = np.sqrt((diff**2).mean())
    max_res = abs(max(diff, key=abs))
    #max_res = max(diff, key=abs)
    print("{:10.3f} {:10.3f}".format(rmse,max_res))
    ones = np.zeros((len(y_conv)))

    #my_font('amp')
    mpl.rcParams.update({'font.size':22})
    ax1.title.set_text('AMP model error test-set')
    #plt.ylabel('PE(kJ/mol)', fontsize=20)
    #plt.xlabel('data', fontsize=20)
    #ax.tick_params(axis='both', which='major', labelsize=15)
    
    #plt.scatter(x, y, 'r', x, y_bar, 'b')
    ax1.set_xlabel('data', fontsize=20)
    ax1.set_ylabel('PE(kJ/mol)', fontsize=20)
    ax2.set_xlabel('data', fontsize=20)
    ax2.set_ylabel('PE(kJ/mol)', fontsize=20)
    p1 = ax1.scatter(range(nlen), y_conv, c='r', marker='o', label='true value')
    p2 = ax1.scatter(range(nlen), h_conv, c='b', marker='^', label='hypothesis')
    p3 = ax2.plot(range(nlen), diff, label='difference')
    ax2.plot(range(nlen), ones)
    plt.legend([p1,p2],['true value', 'hypothesis'])
    plt.savefig(fdata+'.png')

    with open(fdata+'_labels.txt','w') as f:
        f.write('kJ/mol\n')
        for i in range(len(y_conv)):
            f.write(str(y_conv[i])+'\n')

    with open(fdata+'_hypo.txt','w') as g:
        g.write('kJ/mol\n')
        for j in range(len(h_conv)):
            g.write(str(h_conv[j])+'\n')

    return
    
def train_images(images, HL, E_conv):
    Hidden_Layer=tuple(HL)
    calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=Hidden_Layer))
    calc.model.lossfunction = LossFunction(convergence={'energy_rmse': E_conv})
    #calc.model.lossfunction = LossFunction(force_coefficient=-0.1)
    calc.train(images=images, overwrite=True)

def run_md(fdata, atoms):
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md import VelocityVerlet
    fdata = fdata[:-7]

    traj = ase.io.Trajectory(fdata+".traj", 'w')

    calc = Amp.load("amp.amp")
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    MaxwellBoltzmannDistribution(atoms, 100. * units.kB) #Boltzmann constant eV/K
    traj.write(atoms)
    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
   
    for step in range(200):
        pot = atoms.get_potential_energy()  # 
        kin = atoms.get_kinetic_energy()
        with open(fdata+'.txt', 'a') as f: 
            f.write("{}: Total Energy={}, POT={}, KIN={}\n".format(step,pot+kin, pot, kin))
        dyn.run(5)
        ase.io.write(fdata+'.xyz', ase.io.read(fdata+'.traj'), append=True)
        traj.write(atoms)


def run_amp(fdata, job, ndata, HL, E_conv):
    total_images = ase.io.read(fdata, index=':')
    if job == "train":
        if not ndata:
            #images = ase.io.read(fdata, index=':')
            images = total_images
            print("Start training using all the data %d" % len(total_images))
        
            #images = ase.io.read(fdata, index=':
            images = total_images[:ndata]
            print("number of traing data %d/%d" % (len(images), len(total_images)))
        train_images(images, HL, E_conv)

    elif job == "test":
        #images = ase.io.read(fdata, index='ndata:')
        images = total_images[ndata:]

        import glob
        amps = glob.glob('*.amp')
        if 'amp.amp' in amps:
            calc = Amp.load("amp.amp")  
        else: 
            calc = Amp.load("amp-untrained-parameters.amp")
        #print(images[0])
        y=[]
        y_bar=[]
        for mol in images:
            y.append(mol.get_potential_energy())
            #print(mol.get_potential_energy())
            mol.set_calculator(calc)
            y_bar.append(mol.get_potential_energy())
            #print(mol.get_potential_energy())
        #print(images[0])

        draw(fdata, y, y_bar)
    elif job == 'md':
        if not ndata:
            atoms = ase.io.read(fdata, index='0')
        else:
            atoms = ase.io.read(fdata, index=ndata)
        run_md(fdata, atoms)
    return

def main():
    parser = argparse.ArgumentParser(description='run amp with extxyz ')
    parser.add_argument('fin', help='extxyz input file')
    parser.add_argument('job', default='train', choices=['train','test','md'], help='job option: "train", "test", "md"')
    parser.add_argument('-n', '--dlimit', type=int,  help='data range for training and test')
    parser.add_argument('-hl', '--hidden_layer', nargs='*', type=int, default=[8,8,8], help='Hidden Layer of lists of integer')
    parser.add_argument('-el', '--e_convergence', default=0.001, type=float, help='energy convergence limit')
    #group_train = parser.add_mutually_exclusive_group()
    #group_test  = parser.add_mutually_exclusive_group()
    #group_train.add_argument('-l', '--data_limit', type=int, help='the number of data to be trained')
    #group_test.add_argument('-m', '--data_limit2', type=int, help='the start index of data to be tested')
    
    #parser.add_argument('-s', '--sets', default=1, type=int, help='nsets to divide data into training and test set')
    args = parser.parse_args()

    run_amp(args.fin, args.job, args.dlimit, args.hidden_layer, args.e_convergence)
    return

if __name__ == '__main__':
    main()

