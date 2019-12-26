import os
import time
import random
import glob
import numpy as np

def init_pop(npop, layer, neuron, gen, is_converge, old_pop=None):
    if gen == 0:
        pop = np.zeros((npop, layer, 3))
        pop[0,:5] = np.array([[50, 2, 0.2],[50, 2, 0.5],[50, 2, 0.5],[50, 2, 0.5],[50, 2, 0.5]])
        pop[0,5:] = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
    else:
        pop = np.zeros((npop, layer, 3))
        pop[0] = old_pop[0]

    for j in range(1,npop):
        for k in range(layer):
            pop[j,k,0] = np.random.randint(0,neuron+1) # the number of neurons
            if pop[j,k,0] != 0:
                pop[j,k,1] = np.random.randint(0,3) # 0: sigmoid, 1: tanh, 2: ReLU
                pop[j,k,2] = round(np.random.random(),2) # dropout rate
        for _ in range(layer-1):# if the number of neuron is zero, move it to the end of array
            for k in range(layer-1): 
                if pop[j,k,0] == 0:
                    pop[j,k] = pop[j,k+1]
                    pop[j,k+1] = [0,0,0]
        if np.linalg.norm(pop[j,:,0]) == 0: # if there is no neuron, make a layer with one neuron 
            pop[j,k,0] = 1
            pop[j,k,1] = np.random.randint(0,3) 
            pop[j,k,2] = round(np.random.random(),2)
    return pop

def train(pop, trainout):
    npop, _, _ = pop.shape
    train_dr = glob.glob("{:s}*".format(trainout))
    train_dr = list(filter(os.path.isdir, train_dr))
    for i in range(len(train_dr), npop):
        strHL = ''
        strAF = ''
        strDR = ''
        for j in range(len(pop[i,:,0])):
            if pop[i,j,0] != 0:
                strHL += str(int(pop[i,j,0]))+' '
                strAF += str(int(pop[i,j,1]))+' '
                strDR += str(round(pop[i,j,2],2))+' '
        f = open("{:s}{:02d}.sh".format(trainout, i),'w')
        f.write("#!/usr/bin/csh\n\n")
        f.write("$HOME/anaconda3/bin/python tf_ml.py -j train -f complete_CoulombVector_Coupling -hl {:s}-af {:s}-dr {:s}-t c -epoch 3000 -bs 100 -o {:s}{:02d}".format(strHL, strAF, strDR, trainout, i))
        f.close()
        os.system("qsub -cwd {:s}{:02d}.sh".format(trainout, i))

def test(pop, trainout):
    npop, _, _ = pop.shape
    train_dr = glob.glob("{:s}*".format(trainout))
    train_dr = list(filter(os.path.isdir, train_dr))
    for i in range(len(train_dr), npop):
        strHL = ''
        strAF = ''
        for j in range(len(pop[i,:,0])):
            if pop[i,j,0] != 0:
                strHL += str(int(pop[i,j,0]))+' '
                strAF += str(int(pop[i,j,1]))+' '
        f = open("{:s}{:02d}/test{:02d}.sh".format(trainout,i,i),'w')
        f.write("#!/usr/bin/csh\n\n")
        f.write("$HOME/anaconda3/bin/python tf_ml.py -j test -hl {:s}-af {:s}-o test{:02d}".format(strHL, strAF, i))
        f.close()
        os.chdir("{:s}{:02d}".format(trainout, i))
        os.system("qsub -cwd test{:02d}.sh".format(i))
        os.chdir('..')

def crossover(npop, layer, neuron, pop, parents, best_index):
    next_gen = np.zeros((npop, layer, 3)) 
    next_gen[0] = pop[best_index] # the best gene at the top
    for i in range(len(parents)):
        next_gen[i+1] = pop[parents[i]]
    for j in range(int(len(parents)/2)):
        head1 = pop[parents[2*j], :int(layer/2), :]
        head2 = pop[parents[2*j+1], :int(layer/2), :]
        tail1 = pop[parents[2*j], int(layer/2):, :]
        tail2 = pop[parents[2*j+1], int(layer/2):, :]
        baby1 = np.vstack((head1, tail2))
        baby2 = np.vstack((head2, tail1))
        if np.linalg.norm(baby1[:,0]) == 0:
            baby1 = init_pop(1,layer,neuron,1)[0]
        if np.linalg.norm(baby2[:,0]) == 0:
            baby2 = init_pop(1,layer,neuron,1)[0]
        next_gen[i+2*j+2] = baby1
        next_gen[i+2*j+3] = baby2
         
    return next_gen

def converge(pop):
    for i in range(1,len(pop)):
        diffpop = pop[0]-pop[i]
        if np.linalg.norm(diffpop[:,0]) > 10 and np.linalg.norm(diffpop[:,1]) != 0 and np.linalg.norm(diffpop[:,2]) > 0.5:
            return False
            break
    return True

def tcheck(word):
    while True:
        os.system('qstat > qstat.txt')
        with open('qstat.txt','r') as f:
            contents = f.read()
        if word not in contents:
            break
        time.sleep(60)

def main():
    import argparse

    parser = argparse.ArgumentParser("Micro Genetic Algorithm for Multilayer Perceptron")
    parser.add_argument('-n', '--npop', type=int, help='the number of population (odd number please)')
    parser.add_argument('-layer', '--layer', type=int, help='upper limit of the number of layers')
    parser.add_argument('-neuron', '--neuron', type=int, help='upper limit of the number of neurons in a layer')
    parser.add_argument('-g', '--gen', type=int, help='the number of generation')
    parser.add_argument('-to', '--trainout', default=None, help='csh file name for training')

    args = parser.parse_args()

    npop = args.npop
    layer = args.layer 
    neuron = args.neuron
    gen = args.gen
    trainout = args.trainout
    if trainout == None:
        trainout = "train"
    is_converge = True

    scale_factor = round(np.log10(2*neuron*layer))
    
    for generation in range(gen):
        with open("microga.txt", 'a') as result:
            result.write("Generation : {:03d}\n".format(generation))
        idx = list(range(npop))
        group = np.zeros((int((npop-1)/2), 2))
        g_row, g_col = group.shape
        if generation == 0:
            pop = init_pop(npop, layer, neuron, generation, is_converge)
        else:
            if is_converge == False:
                pop = crossover(npop, layer, neuron, pop, parents, best_index)
            else:
                pop = init_pop(npop, layer, neuron, generation, is_converge, pop)            
        
        train(pop, trainout)
        tcheck(trainout)
        test(pop, trainout)
        tcheck('test')
        with open("microga.txt", 'a') as result:
            result.write(str(pop)+'\n')

        p, w = [], [] # p:performance, w:the size of trainable parameters
        fitness = [] # minimize fitness
        for i in range(npop):
            with open("{:s}{:02d}/score".format(trainout, i), 'r') as f:
                flines = f.readlines()
            p.append(eval(flines[0]))
            w.append(eval(flines[1]))
        p_star = np.array(p) / np.linalg.norm(p) # normalize p vector
        w_star = np.log10(np.array(w)) # scale order of w
        for i in range(npop):
            fitness.append(scale_factor*(1-0.7)*p_star[i]+0.3*w_star[i])

        cp_fitness = fitness.copy()    
        best_index = idx.pop(fitness.index(min(fitness)))
        fitness.remove(min(fitness))
        os.system('cp -r {:s}{:02d} best{:02d}'.format(trainout, best_index, generation))
        with open("microga.txt", 'a') as result:
            result.write(str(cp_fitness)+'\n')
        # Selection
        for r in range(g_row):
            for c in range(g_col):
                group[r,c] = idx.pop(idx.index(random.choice(idx))) # Save indices for tournament
        
        parents = []
        for r in range(g_row):
            parents.append(cp_fitness.index(min(cp_fitness[int(group[r,0])], cp_fitness[int(group[r,1])])))
            if os.path.isdir("parents{:02d}".format(r)):
                os.system("rm -rf parents{:02d}".format(r))
            os.system("cp -r {:s}{:02d} parents{:02d}".format(trainout, parents[r],r))

        os.system("rm -rf {:s}*".format(trainout))
        os.system("mv best{:02d} {:s}00".format(generation, trainout))
        for r in range(g_row):
            os.system("mv parents{:02d} {:s}{:02d}".format(r, trainout, r+1))

        is_converge = converge(pop)
        if is_converge == True:
            with open("microga.txt", 'a') as result:
                result.write(str("***Restart Micro Population***")+'\n')

    result.close()

if __name__ == '__main__':
    main()
