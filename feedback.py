import os, glob, time

num_images = 2400
step = 0

runw01 = False #True
runw02 = True
runw03 = True

newdir = glob.glob("new_[0-9][0-9][0-9]")
newdir.sort()
step = len(newdir)

while step < 100: # 100 loops
    if os.path.isfile('num_images'):
        with open('num_images','r') as z:
            line = z.readline()
        num_images = eval(line)
    with open('fml.sh','w') as f:
        f.write("#!/usr/bin/csh\n\n")
        if step == 0:
            f.write("/gpfs/home/minbk/anaconda3/bin/python qchem_amp_etest.py w_1_1_1.extxyz train -n {:d} -hl 4 4 4 4 4 -el 0.0001\n".format(2400))
        else:
            f.write("/gpfs/home/minbk/anaconda3/bin/python qchem_amp_etest.py new_geo.extxyz retrain -n {:d} -hl 4 4 4 4 4 -el 0.0001\n".format(num_images))
        f.write("/gpfs/home/minbk/anaconda3/bin/python subplot3.py w_1_1_1.extxyz test -n {:d}\n".format(2400))
        f.write("/gpfs/home/minbk/anaconda3/bin/python amp_ene.py w_1_1_1.extxyz test -n 0\n") 
        f.write("/gpfs/home/minbk/anaconda3/bin/python convergence.py -el 0.0001\n")
        if runw01 == True:
            f.write("/gpfs/home/minbk/anaconda3/bin/python amp_ene.py w_1_1_1.extxyz md -n {:d} -mdout w01md\n".format(0))
        if runw02 == True:
            f.write("/gpfs/home/minbk/anaconda3/bin/python amp_ene.py w_1_1_1.extxyz md -n {:d} -mdout w02md\n".format(1))
        if runw03 == True:
            f.write("/gpfs/home/minbk/anaconda3/bin/python amp_ene.py w_1_1_1.extxyz md -n {:d} -mdout w03md\n".format(2))
        f.write("/gpfs/home/minbk/anaconda3/bin/python mv.py -d new_{:03d}\n".format(step))
    os.system("qsub -cwd -pe numa 20 fml.sh")


    while True:
        time.sleep(60)
        os.system("qstat > qstat.txt")
        with open("qstat.txt",'r') as g:
            line = g.read()
            if "fml" not in line:
                break
        os.remove('qstat.txt')
    shfin = glob.glob("*.sh.o*")
    shfin.sort()
    shf = shfin.pop()
    shfnum = shf[-7:]
    os.system("mv *"+shfnum+" new_{:03d}".format(step))
    os.chdir("new_{:03d}".format(step))

    # Convert the output of md to input for qchem
    if runw01 == True:
        os.system("python3 xyz2qchem.py -f w01md.xyz -n 3")
    if runw02 == True:
        os.system("python3 xyz2qchem.py -f w02md.xyz -n 6")
    if runw03 == True:
        os.system("python3 xyz2qchem.py -f w03md.xyz -n 9")
    os.system("python3 serialjob.py")
    os.system("cp qchemsp2extxyz.py ./done/.")
    os.chdir("./done")

    # Make .extxyz from the outputs of qchem
    w1isbreak = glob.glob("w01md*out")
    w2isbreak = glob.glob("w02md*out")
    w3isbreak = glob.glob("w03md*out")
    num_images += len(w1isbreak) + len(w2isbreak) + len(w3isbreak)
    with open('../../num_images','w') as h:
        h.write(str(num_images))
    if len(w1isbreak) < 200:
        os.system("python3 qchemsp2extxyz.py -f w01md -n 3")
    if len(w2isbreak) < 200:
        os.system("python3 qchemsp2extxyz.py -f w02md -n 6")
    if len(w3isbreak) < 200:
        os.system("python3 qchemsp2extxyz.py -f w03md -n 9")

    if os.path.isfile('w01md.extxyz') == True:
        os.system("cat w01md.extxyz >> new_geo.extxyz")
    else:
        runw01 = False
    if os.path.isfile('w02md.extxyz') == True:
        os.system("cat w02md.extxyz >> new_geo.extxyz")
    else:
        runw02 = False
    if os.path.isfile('w03md.extxyz') == True:
        os.system("cat w03md.extxyz >> new_geo.extxyz")
    else:
        runw03 = False
    if runw01 == False and runw02 == False and runw03 == False:
        break

    os.system("cp new_geo.extxyz ../../.")
    os.chdir("../../")
    if step == 0:
        os.system("cat new_{:03d}/w_1_1_1.extxyz >> new_geo.extxyz".format(step))
    else:
        os.system("cat new_{:03d}/new_geo.extxyz >> new_geo.extxyz".format(step))
    step += 1
