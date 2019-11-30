import os, glob

allfiles = glob.glob('*')
dlist = list(filter(lambda x: os.path.isdir(x) == True, allfiles))

train = []
total = []
HL = []
rmse = []
max_res = []

for d in dlist:
    os.chdir(d)
    with open("summary.txt", 'r') as f:
        flines = f.readlines()
    data = flines[:3]
    for i in range(len(data)):
        if i == 0:
            data[i] = data[i].split(' ')
            data[i] = data[i][-1]
            data[i] = data[i].split('/')
            train.append(data[i][0])
            total.append(data[i][1][:-1])
        elif i == 1:
            data[i] = data[i].split('(')
            data[i] = data[i][1]
            HL.append(data[i][1:-2])
        elif i == 2:
            data[i] = data[i].split(' ')
            while '' in data[i]:
                data[i].remove('')
            rmse.append(data[i][2])
            max_res.append((data[i][-1]))
    os.chdir('..')

with open("TFtest.csv",'w') as g:
    g.write("dir,train,total,rmse,max_res\n")
    for i in range(len(dlist)):
        g.write("{:s},{:s},{:s},{:s},{:s}".format(dlist[i],train[i],total[i],rmse[i],max_res[i]))
