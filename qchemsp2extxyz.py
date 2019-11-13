import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--fin')
parser.add_argument('-n','--num_atoms', type=int)
args = parser.parse_args()

infiles = glob.glob("{:s}*in".format(args.fin))
infiles.sort()

h2ev = 27.2114
atomic_number = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}

xyz = []
en = []
for i in range(len(infiles)):
    with open(infiles[i],'r') as f:
        flines = f.readlines()
    xyz.append(flines[2:2+args.num_atoms])

    with open(infiles[i][:-2]+"out",'r') as g:
        while True:
            line = g.readline()
            if 'Total energy in the final basis set' in line:
                line = line.split('=')
                en.append(eval(line[1])*h2ev)
                break
            if not line:
                break

with open('{:s}.extxyz'.format(args.fin),'w') as h:
    for i in range(len(xyz)):
        h.write('{:d}\nLattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties="species:S:1:pos:R:3:forces:R:3" energy={:.10f}\n'.format(args.num_atoms,en[i]))
        for j in range(len(xyz[i])):
            h.write('{:s} 0.0 0.0 0.0\n'.format(xyz[i][j][:-1]))
