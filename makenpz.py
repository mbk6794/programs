import numpy as np
import os
import glob
import argparse

h2ev = 27.2114

def makenpz(dirpath, sidx, eidx, num_atoms):
    chem_dic = {'H':1,'C':6,'O':8}
    dp = glob.glob('{:s}[0-9][0-9][0-9]'.format(dirpath))
    dp.sort()
        dp = dp[sidx:]
    else:
        dp = dp[sidx:eidx]

    dictionary = {'R':[],'D':[],'E':[],'Z':[],'C':[]}
    for d in dp:
        os.chdir('{:s}/done'.format(d))
        os.system('pwd')
        outputs = glob.glob('*out')
        outputs.sort()
        for o in outputs:
            with open(o, 'r') as f: 
                flines = f.readlines()
            for i in range(len(flines)):
                if 'Standard Nuclear Orientation' in flines[i]:
                    coor = flines[i+3:i+3+num_atoms]
                    image_coor = []
                    nuc_charge = []
                    for j in range(len(coor)):
                        coor[j] = coor[j].split(' ')
                        while '' in coor[j]:
                            coor[j].remove('')
                        for k in range(2,5):
                            coor[j][k] = eval(coor[j][k])
                        image_coor.append([coor[j][2:5]])
                        nuc_charge.append(chem_dic[coor[j][1]])
                    dictionary['Z'].append(nuc_charge)
                    dictionary['R'].append(image_coor)

                if 'Total energy in the final basis set' in flines[i]:
                    energy = flines[i]
                    energy = energy.split('=')
                    energy = eval(energy[-1]) * h2ev
                    dictionary['E'].append(energy)

                if 'Coupling(eV)' in flines[i]:
                    coupling = flines[i+2]
                    coupling = coupling.split(' ')
                    while '' in coupling:
                        coupling.remove('')
                    coupling = eval(coupling[5]) 
                    dictionary['C'].append(coupling)
        
                if 'Cartesian Multipole Moments' in flines[i]:
                    dip = flines[i+5]
                    image_dip = []
                    dip = dip.split(' ')
                    while '' in dip:
                        dip.remove('')
                    for j in [1,3,5]:
                        image_dip.append(eval(dip[j]))
                    dictionary['D'].append(image_dip)
                    break
        
        dictionary['Z'] = np.array(dictionary['Z']).reshape((-1,num_atoms))
        dictionary['R'] = np.array(dictionary['R']).reshape((-1,num_atoms,3))             
        dictionary['E'] = np.array(dictionary['E']).reshape((-1,1))
        dictionary['C'] = np.array(dictionary['C']).reshape((-1,1))
        dictionary['D'] = np.array(dictionary['D']).reshape((-1,3))
        
        os.chdir('../..')
    print(dictionary.items())
    np.savez('./{:s}.npz'.format(dirpath),**dictionary)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program that makes .npz file')
    parser.add_argument('-d','--dir',help='iterative directory path')
    parser.add_argument('-s','--start',type=int,default=0,help='starting number of directory')
    parser.add_argument('-e','--end',type=int,default=-1,help='end number of directory')    
    parser.add_argument('-n','--num_atoms',type=int,help='the number of atoms')  

    args = parser.parse_args()
    makenpz(args.dir, args.start, args.end, args.num_atoms)
 
