#!/gpfs/home/minbk/anaconda3/bin/python

import os

import argparse

import ase.io
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
import amp.utilities 

import numpy as np
from my_mplot2d import *
from my_chem import *

def train_images(images, HL, E_conv):
    Hidden_Layer=tuple(HL)
    print("Hidden Layer: {}".format(Hidden_Layer))
    print("Energy convergence: {} kJ/mol".format(E_conv/96.487))
    calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=Hidden_Layer), cores=20)
    calc.model.lossfunction = LossFunction(convergence={'energy_rmse': E_conv/96.487})
    #calc.model.lossfunction = LossFunction(force_coefficient=-0.1)
    calc.train(images=images, overwrite=True)
    return
   
def re_train_images(images, HL, E_conv):
    Hidden_Layer=tuple(HL)
    print("Hidden Layer: {}".format(Hidden_Layer))
    print("Energy convergence: {} kJ/mol".format(E_conv/96.487))
    calc = Amp.load("amp.amp")
    calc.model.lossfunction = LossFunction(convergence={'energy_rmse': E_conv/96.487})
    calc.train(images=images, overwrite=True)

def run_amp(fdata, job, ndata, HL, E_conv):

    total_images = ase.io.read(fdata, index=':')
    if job == "train":
        if not ndata:
            #images = ase.io.read(fdata, index=':')
            images = total_images
            print("Start training using all the data %d" % len(total_images))
        else:
            #images = ase.io.read(fdata, index=':ndata')
            images = total_images[:ndata]
            print("number of training data %d/%d" % (len(images), len(total_images)))
        train_images(images, HL, E_conv)

    elif job == "retrain":
        if not ndata:
            images = total_images
            print("Start re-training using all the data %d" % len(total_images))
        else:
            images = total_images[:ndata]
            print("number of re-training data %d/%d" % (len(images), len(total_images)))
        re_train_images(images, HL, E_conv)

    elif job == "test":
        #images = ase.io.read(fdata, index='ndata:')
        images = total_images[ndata:]

        calc = Amp.load("amp.amp")
        #print(title)
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

        h_conv = np.array(y_bar) * kcal2kj
        y_conv = np.array(y) * kcal2kj
        diff =  np.subtract(h_conv,y_conv) 
        rmse = np.sqrt((diff**2).mean())
        max_res = abs(max(diff, key=abs))
        with open("score", 'w') as f:
            f.write("{}\n".format(0.9*max_res+0.1*rmse))

        os.system("cat HL score > ./pop_fit.txt")
        if os.path.isfile('HL'):
            os.remove('HL')
          
      
    return

def main():
    #dir1=os.getcwd()
    #dir1=dir1[-5:]
    #if os.path.isfile('../'+dir1+'done') == True:
    #    os.remove('../'+dir1+'done')
    #if os.path.isfile('score') == True:
    #    os.remove('score')
    parser = argparse.ArgumentParser(description='run amp with extxyz ')
    parser.add_argument('fin', help='extxyz input file')
    parser.add_argument('job', default='train', choices=['train', 'test', 'retrain'], help='job option: "train" vs "test"')
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
    #run_amp(args.fin, 'test', args.dlimit, args.hidden_layer, args.e_convergence)
    #dir1=os.getcwd()
    #dir1=dir1[-5:]
    #os.system("cat pop_fit.txt >> ../pop_fit.txt") 
    #with open('../'+dir1+'done','w') as f:
    #    f.write(dir1+'\ncalculation is done')
    #os.system('rm run_py.sh.e*')
    return

if __name__ == '__main__':
    main()

