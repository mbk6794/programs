import glob
import os
import time

#dpath = glob.glob("btdgro*")
#dpath.sort()
#dp = dpath[6:10]
dp = ["."]

for dpi in dp:
    os.chdir(dpi)
    w1inputfile = glob.glob('w01md*.in')
    w2inputfile = glob.glob('w02md*.in')
    w3inputfile = glob.glob('w03md*.in')
    w1inputfile.sort()
    w2inputfile.sort()
    w3inputfile.sort()
    w1inputfile.reverse()
    w2inputfile.reverse()
    w3inputfile.reverse()
    w1outputfile = glob.glob('w01md*.out')
    w2outputfile = glob.glob('w02md*.out')
    w3outputfile = glob.glob('w03md*.out')
    if len(w1outputfile) != 0:
        for name in w1outputfile:
            w1inputfile.remove(name[:-3]+'in')
    if len(w2outputfile) != 0:
        for name in w2outputfile:
            w2inputfile.remove(name[:-3]+'in')
    if len(w3outputfile) != 0:
        for name in w3outputfile:
            w3inputfile.remove(name[:-3]+'in')
    
    if os.path.isfile('ps.txt'):
        os.system('rm ps.txt')
    if os.path.isfile('qstat.txt'):
        os.system('rm qstat.txt')
    if os.path.isdir('done') == False:
        os.mkdir('done')
    if os.path.isdir('failed') == False:
        os.mkdir('failed')
        
    
    while True:
        thank = []
        cisexp = []
        start = 0
        end = 0
    
        time.sleep(1)
        #os.system('ps > ps.txt')
        os.system('qstat > qstat.txt')
  
        with open('qstat.txt', 'r') as f:
            flines = f.read()
        if 'wsp' not in flines:
            for _ in range(50):
                if len(w1inputfile) != 0:
                    name = w1inputfile.pop()
                    with open('wsp.sh','w') as z:
                        z.write('#!/usr/bin/csh\n\n') 
                        z.write('qchem '+name+' '+name[:-2]+'out')
                    os.system('qsub -cwd wsp.sh')
                else:
                    pass
            for _ in range(50):
                if len(w2inputfile) != 0:
                    name = w2inputfile.pop()
                    with open('wsp.sh','w') as z:
                        z.write('#!/usr/bin/csh\n\n') 
                        z.write('qchem '+name+' '+name[:-2]+'out')
                    os.system('qsub -cwd wsp.sh')
                else:
                    pass
            for _ in range(50):
                if len(w3inputfile) != 0:
                    name = w3inputfile.pop()
                    with open('wsp.sh','w') as z:
                        z.write('#!/usr/bin/csh\n\n') 
                        z.write('qchem '+name+' '+name[:-2]+'out')
                    os.system('qsub -cwd wsp.sh')
                else:
                    pass
            
            while True:
                time.sleep(10)
                os.system('qstat > qstat.txt')
                with open('qstat.txt','r') as y:
                    ylines = y.read()
                if 'wsp' not in ylines:
                    if os.path.isfile('finished'):
                        os.system('rm finished')
                    os.system("grep -r 'Thank you' ./*out > finished")
                    if os.path.isfile('fail'):        
                        os.system('rm fail')
                    os.system("grep -r 'Error in gen_scfman' ./*out > fail")
                    with open('finished', 'r') as g:
                        glines = g.readlines()
            
                    with open('fail', 'r') as h:
                        hlines = h.readlines()
            
                    for i in glines:
                        start = i.index('/')
                        end = i.index(':')
                        thank.append(i[start+1:end])
            
                    for i in hlines:
                        start = i.index('/')
                        end = i.index(':')
                        cisexp.append(i[start+1:end])
            
                    for name in thank:
                        os.system('mv '+name+' ./done/.')
                        os.system('mv '+name[:-3]+'in ./done/.')
            
                    for name in cisexp:
                        os.system('mv '+name+' ./failed/.')
                        os.system('mv '+name[:-3]+'in ./failed/.')
                        if 'w01md' in name:
                            for name_1 in w1inputfile:
                                os.system('mv '+name_1[:-3]+'* ./failed/.')
                            w1inputfile = []
                        if 'w02md' in name:
                            for name_1 in w2inputfile:
                                os.system('mv '+name_1[:-3]+'* ./failed/.')
                            w2inputfile = []
                        if 'w03md' in name:
                            for name_1 in w3inputfile:
                                os.system('mv '+name_1[:-3]+'* ./failed/.')
                            w3inputfile = []
                          
                    #os.system('rm *.sh.e*')
                    #os.system('rm *.sh.o*')
                    break
                
        llen = len(w1inputfile) + len(w2inputfile) + len(w3inputfile)
        if llen == 0:
            break   
    #os.chdir('..') 
