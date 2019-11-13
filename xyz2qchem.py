import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--fin')
parser.add_argument('-n','--num_atoms',type=int)

args = parser.parse_args()

with open(args.fin,'r') as f:
    flines = f.readlines()

numatoms= args.num_atoms

data = []
for i in range(200):
    data.append(flines[i*(numatoms+2)+2:(i+1)*(numatoms+2)])
    for j in range(len(data[i])):
        data[i][j] = data[i][j].split(' ')
        while '' in data[i][j]:
            data[i][j].remove('')
    with open('{:s}_{:03d}.in'.format(args.fin[:-4],i),'w') as g:
        g.write("$molecule\n0 1\n")
        for j in range(len(data[i])):
            g.write(data[i][j][0]+"  "+data[i][j][1]+"  "+data[i][j][2]+"  "+data[i][j][3]+"\n")
        g.write("$end\n$rem\njobtype sp\nmethod revpbe0\nbasis g3large\ndft_d empirical_grimme3\nmem_static 2000\nmem_total 3000\n$end")
