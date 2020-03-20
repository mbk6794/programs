#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.switch_backend('agg')
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
from datetime import datetime
import glob
import os


# In[ ]:


def makefeature(tr, fin=None, intype=None, ndata=0, log=None):
    if tr == "train":
        for name in range(len(fin)):
            molecule_dataframe = pd.read_csv(fin[name],sep=",")
            if ndata == 0:
                molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
            else:
                dt_limit = int(ndata / 0.8 / len(fin))
                if dt_limit > len(molecule_dataframe):
                    molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
                else:
                    molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
                    molecule_dataframe = molecule_dataframe.head(dt_limit)

            r, c = np.array(molecule_dataframe).shape
            n = 0
    
            while True: #Determine the number of atoms in this system
                if n*(n+1)/2 == (c-1):
                    break
                n += 1
    
            if intype == 'a':
                columns = []
               
                for i in range(1,n+1):
                    for j in range(i,n+1):
                        columns.append('{:2d}-{:2d}'.format(i,j))
                
            elif intype == 'b':
                columns = []
              
                for i in range(1,n+1): 
                    columns.append('{:2d}-{:2d}'.format(i,i)) 
                for i in range(1,int(n/2)+1):
                    for j in range(int(n/2)+1,n+1):
                        columns.append('{:2d}-{:2d}'.format(i,j))
        
            elif intype == 'c':
                columns = []
         
                for i in range(1,int(n/2)+1):
                    for j in range(int(n/2)+1,n+1):
                        columns.append('{:2d}-{:2d}'.format(i,j))
    
            feature = np.zeros((len(molecule_dataframe),len(columns)))

            for i in range(len(columns)):
                for sample in range(len(molecule_dataframe)):
                    feature[sample][i] = np.array(molecule_dataframe[columns[i]])[sample]
         
            target = np.zeros((len(molecule_dataframe)))
            for sample in range(len(molecule_dataframe)):
                if log == None:
                    target[sample] = abs(np.array(molecule_dataframe["Coupling(eV)"])[sample]) 
                else:
                    target[sample] = np.log10(abs(np.array(molecule_dataframe["Coupling(eV)"])[sample])) 
    
            target = target.reshape((len(molecule_dataframe),1))

            if name == 0:
                Feature = feature
                Target = target
            else:
                Feature = np.vstack((Feature, feature)) 
                Target = np.vstack((Target, target))           

        Data = np.hstack((Feature, Target))
        Data = np.random.permutation(Data)
        Feature = Data[:,:-1]
        Target = Data[:,-1].reshape((len(Data),1))
        if ndata != 0:
            lentrain = ndata
        elif ndata == 0 or ndata >= len(Data):
            lentrain = int(len(Feature) * 0.8)
        lentest = int((len(Data)-lentrain)/2)
        lenval = int((len(Data)-lentrain)/2)
        lentotal = lentrain + lentest + lenval
        
        tr_images, tr_labels, val_images, val_labels, te_images, te_labels = Feature[:lentrain, :], Target[:lentrain, :], Feature[lentrain:lentrain+lenval, :], Target[lentrain:lentrain+lenval, :] ,Feature[lentrain+lenval:lentotal, :], Target[lentrain+lenval:lentotal,:]
        return tr_images, tr_labels, val_images, val_labels, te_images, te_labels, lentrain, lenval, lentotal

    elif tr == 'retrain' or tr == 'test':
        if fin == None:
            tr_img = glob.glob('tr_images*.txt')[0]
            tr_lab = glob.glob('tr_labels*.txt')[0]
            val_img = glob.glob('val_images*.txt')[0]
            val_lab = glob.glob('val_labels*.txt')[0]
            te_img = glob.glob('te_images*.txt')[0]
            te_lab = glob.glob('te_labels*.txt')[0]
            # make training_images
            with open(tr_img,'r') as f: 
                flines = f.readlines()
            data = flines[:]
            for i in range(len(data)):
                data[i] = data[i].split(' ')
                data[i].pop()
                for j in range(len(data[i])):
                    data[i][j] = eval(data[i][j])
            tr_images = np.array(data)

            # make validation_images
            with open(val_img,'r') as f: 
                flines = f.readlines()
            data = flines[:]
            for i in range(len(data)):
                data[i] = data[i].split(' ')
                data[i].pop()
                for j in range(len(data[i])):
                    data[i][j] = eval(data[i][j])
            val_images = np.array(data)
            
            # make training_labels
            with open(tr_lab,'r') as f: 
                flines = f.readlines()
            data = flines[1:]
            for i in range(len(data)):
                data[i] = eval(data[i])
            if log == None:
                tr_labels = np.array(data)
            else:
                tr_labels = np.log10(np.array(data))
            tr_labels = tr_labels.reshape((len(tr_labels),1))
            # make validation_labels
            with open(val_lab,'r') as f: 
                flines = f.readlines()
            data = flines[1:]
            for i in range(len(data)):
                data[i] = eval(data[i])
            if log == None:
                val_labels = np.array(data)
            else:
                val_labels = np.log10(np.array(data))
            val_labels = val_labels.reshape((len(val_labels),1))
       
            # make test_images
            with open(te_img,'r') as f: 
                flines = f.readlines()
            data = flines[:]
            for i in range(len(data)):
                data[i] = data[i].split(' ')
                data[i].pop()
                for j in range(len(data[i])):
                    data[i][j] = eval(data[i][j])
            te_images = np.array(data)
        
            # make test_labels
            with open(te_lab,'r') as f: 
                flines = f.readlines()
            data = flines[1:]
            for i in range(len(data)):
                data[i] = eval(data[i])
            if log == None:
                te_labels = np.array(data)
            else:
                te_labels = np.log10(np.array(data))
            te_labels = te_labels.reshape((len(te_labels),1))

        else:
            for name in range(len(fin)):
                molecule_dataframe = pd.read_csv(fin[name],sep=",")
                if ndata == 0:
                    molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
                else:
                    dt_limit = int(ndata / 0.8 / len(fin))
                    if dt_limit > len(molecule_dataframe):
                        molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
                    else:
                        molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
                        molecule_dataframe = molecule_dataframe.head(dt_limit)
    
                r, c = np.array(molecule_dataframe).shape
                n = 0
        
                while True: #Determine the number of atoms in this system
                    if n*(n+1)/2 == (c-1):
                        break
                    n += 1
            
                if intype == 'a':
                    columns = []
                   
                    for i in range(1,n+1):
                        for j in range(i,n+1):
                            columns.append('{:2d}-{:2d}'.format(i,j))
                        
                elif intype == 'b':
                    columns = []
                  
                    for i in range(1,n+1): 
                        columns.append('{:2d}-{:2d}'.format(i,i)) 
                    for i in range(1,int(n/2)+1):
                        for j in range(int(n/2)+1,n+1):
                            columns.append('{:2d}-{:2d}'.format(i,j))
                        
                elif intype == 'c':
                    columns = []
             
                    for i in range(1,int(n/2)+1):
                        for j in range(int(n/2)+1,n+1):
                            columns.append('{:2d}-{:2d}'.format(i,j))
                        
                feature = np.zeros((len(molecule_dataframe),len(columns)))
            
                for i in range(len(columns)):
                    for sample in range(len(molecule_dataframe)):
                        feature[sample][i] = np.array(molecule_dataframe[columns[i]])[sample]
             
                target = np.zeros((len(molecule_dataframe)))
                for sample in range(len(molecule_dataframe)):
                    if log == None:
                        target[sample] = abs(np.array(molecule_dataframe["Coupling(eV)"])[sample]) 
                    else:
                        target[sample] = np.log10(abs(np.array(molecule_dataframe["Coupling(eV)"])[sample]))
             
                Target = target.reshape((len(molecule_dataframe),1))

                if name == 0:
                    Feature = feature
                    Target = target
                else:
                    Feature = np.vstack((Feature, feature)) 
                    Target = np.vstack((Target, target))            
    
            Data = np.hstack((Feature, Target))
            Data = np.random.permutation(Data)
            Feature = Data[:,:-1]
            Target = Data[:,-1].reshape((len(Data),1))
            if ndata != 0:
                lentrain = ndata
            elif ndata == 0 or ndata >= len(Data):
                lentrain = int(len(Feature) * 0.8)
            lentest = int((len(Data)-lentrain)/2)
            lenval = int((len(Data)-lentrain)/2)
            lentotal = lentrain + lentest + lenval
    
            te_images, te_labels = Feature[:lentest, :], Target[:lentest,:]
            tr_images, tr_labels, val_images, val_labels, te_images, te_labels = Feature[:lentrain, :], Target[:lentrain, :], Feature[lentrain:lentrain+lenval, :], Target[lentrain:lentrain+lenval, :] ,Feature[lentrain+lenval:lentotal, :], Target[lentrain+lenval:lentotal,:]

        return tr_images, tr_labels, val_images, val_labels, te_images, te_labels, len(tr_labels), len(val_labels), len(tr_labels)+len(val_labels)+len(te_labels)

def draw(tr_labels, tr_hypo, te_labels, te_hypo, dirpath, t):
    fig = plt.figure(figsize=(20,10))
    plt.rc('font', size=15)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text("train")
    ax2.title.set_text("test")
    ax1.set_ylabel("hypothesis (eV)")
    ax1.set_xlabel("labels (eV)")
    ax2.set_ylabel("hypothesis (eV)")
    ax2.set_xlabel("labels (eV)")

    tr_lim = max(max(tr_labels), max(tr_hypo)) 
    te_lim = max(max(te_labels), max(te_hypo))
    max_lim = max(tr_lim, te_lim)

    ax1.set_xlim([0,max_lim])
    ax1.set_ylim([0,max_lim])
    ax2.set_xlim([0,max_lim])
    ax2.set_ylim([0,max_lim])
  
    ax1.scatter(tr_labels, tr_hypo, c='r', s=1)
    ax1.plot([0,max_lim], [0,max_lim], c='k')
    ax2.scatter(te_labels, te_hypo, c='r', s=1)
    ax2.plot([0,max_lim], [0,max_lim], c='k')
    
    plt.savefig("{:s}/CM{:s}.png".format(dirpath, t))
# In[ ]:


parser = argparse.ArgumentParser(description='tensorflow training')
parser.add_argument('-j','--job', choices=['train', 'retrain','test'], help='job option: train, retrain or test')
parser.add_argument('-f','--fin', nargs='*', default=None, help='write filename if you want to train first')
parser.add_argument('-hl','--hiddenlayer', nargs='*', type=int, help='hidden layers: ex) 5 5 5 means three layers, five perceptrons each')
parser.add_argument('-af','--actfunc', nargs='*', type=int, default=None, help='the activation function of each layer')
parser.add_argument('-dr','--drop', nargs='*', type=float, default=None, help='the rate of dropout of each layer')
parser.add_argument('-t','--intype', choices=['a','b','c'], help='a: inter-, intra-, diag / b: inter-, diag / c: inter-')
parser.add_argument('-nd','--ndata', type=int, default=None, help='the number of total data set, if None, whole data will be used')
parser.add_argument('-epoch','--epoch', type=int, help='max number of epochs')
parser.add_argument('-bs','--batchsize', type=int, help='batch size')
parser.add_argument('-o','--output', default=None, help='output directory name')
parser.add_argument('-log','--log', default=None, help='if this is true, labels set to logarithm value')

args = parser.parse_args()

tr_images, tr_labels, val_images, val_labels, te_images, te_labels, lentrain, lenval, lentotal = makefeature(args.job, args.fin, args.intype, args.ndata, args.log)

# In[ ]:
_, n_inputs = tr_images.shape
HL = tuple(args.hiddenlayer)
if args.drop != None:
    DR = tuple(args.drop)
else:
    DR = ()
afdic = {0:tf.nn.sigmoid, 1:tf.nn.tanh, 2:tf.nn.relu}
AF = []
if args.actfunc != None:
    for i in args.actfunc:
        AF.append(afdic[i])
    AF = tuple(AF)
else:
    AF = ()

n_hidden = []
n_output = 1


# In[ ]:


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")
is_training = tf.placeholder(tf.bool)

# In[ ]:


with tf.name_scope("dnn"):
    for i in range(len(HL)):
        if i == 0:
            if len(AF) == 0:
                n_hidden.append(tf.layers.dense(X, HL[i], name="hidden{:d}".format(i+1), activation=tf.nn.relu))
            else:
                n_hidden.append(tf.layers.dense(X, HL[i], name="hidden{:d}".format(i+1), activation=AF[i]))
            if len(DR) == 0:
                n_hidden[i] = tf.layers.dropout(n_hidden[i], 0.2, is_training)
            else:
                n_hidden[0] = tf.layers.dropout(n_hidden[0], DR[i], is_training)                    
        elif i>0 and i<len(HL)-1:
            if len(AF) == 0:
                n_hidden.append(tf.layers.dense(n_hidden[i-1], HL[i], name="hidden{:d}".format(i+1), activation=tf.nn.relu))
            else:
                n_hidden.append(tf.layers.dense(n_hidden[i-1], HL[i], name="hidden{:d}".format(i+1), activation=AF[i]))
            if len(DR) == 0:
                n_hidden[i] = tf.layers.dropout(n_hidden[i], 0.5, is_training)
            else:
                n_hidden[i] = tf.layers.dropout(n_hidden[i], DR[i], is_training) 
        elif i == len(HL)-1:
            n_hidden.append(tf.layers.dense(n_hidden[i-1], n_output, name="output"))


# In[ ]:

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.squared_difference(n_hidden[-1], y), name="loss")


# In[ ]:


learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[ ]:


init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)


# In[ ]:


n_epochs = args.epoch
batch_size = args.batchsize


# In[ ]:


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
def batch(X, y, batch_size):
    if batch_size >= len(X):
        yield X, y
    else:
        idx = np.arange(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(idx, n_batches):
            X_batch, y_batch = X[batch_idx,:], y[batch_idx,:]
            yield X_batch, y_batch


# In[ ]:

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
with tf.Session(config=config) as sess:
    t = datetime.utcnow().strftime("%m%d%H%M%S")
    if args.output == None:
        dirpath = "{:s}_{:d}".format(t, lentotal)
    else:
        dirpath = args.output
    os.mkdir(dirpath)
    ckpt = tf.train.get_checkpoint_state('validation')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, tf.train.latest_checkpoint('validation'))#ckpt.model_checkpoint_path))
        print('retrain or test')
    else: 
        sess.run(init)
        print('train')
    if args.job == "train" or args.job == "retrain":
        f = open('{:s}/CM_tf{:s}.txt'.format(dirpath,t),'w')            
        for epoch in range(n_epochs):
            for X_batch, y_batch in batch(tr_images, tr_labels, batch_size):
                loss_, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch, is_training: True})
            if epoch%100 == 0:
                loss_tr = sess.run(loss, feed_dict={X:tr_images, y:tr_labels, is_training: True})
                loss_valid, val_hypo = sess.run([loss, n_hidden[-1]], feed_dict={X:val_images, y:val_labels, is_training: True})
                if args.log != None:
                    loss_tr = np.power(10, loss_tr)
                    loss_valid = np.power(10, loss_valid)        
                if os.path.exists("{:s}/validation".format(dirpath)) != True:
                    os.mkdir("{:s}/validation".format(dirpath))
                if epoch == 0:
                    save_path = saver.save(sess, "{:s}/validation/CM_tf{:s}".format(dirpath, t), global_step=epoch)
                    loss_valid_before = loss_valid
                else:
                    if loss_valid < loss_valid_before:
                        save_path = saver.save(sess, "{:s}/validation/CM_tf{:s}".format(dirpath, t), global_step=epoch)
                        loss_valid_before = loss_valid
                f.write("{:d} loss: {:f} validation_loss: {:f}\n".format(epoch, loss_tr, loss_valid))
                
            if loss_tr < 3*1e-6:
                break
    
        f.write("final loss: {:f}\n".format(loss_tr))
        endtrain = datetime.utcnow().strftime("%m%d%H%M%S")
        f.close()
    tr_labels = tr_labels.reshape(lentrain)
    val_labels = val_labels.reshape(lenval)

    strt = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    tr_hypo = sess.run(n_hidden[-1], feed_dict={X:tr_images, is_training: False})
    etrt = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    stet = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    te_hypo = sess.run(n_hidden[-1], feed_dict={X:te_images, is_training: False})
    etet = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    tr_hypo = np.array(tr_hypo).reshape((lentrain,1))
    te_hypo = np.array(te_hypo).reshape((lentotal-(lentrain+lenval),1))

    if args.log != None:
        tr_labels = np.power(10,tr_labels)
        tr_hypo = np.power(10,tr_hypo)
        te_labels = np.power(10,te_labels)
        te_hypo = np.power(10,te_hypo)    
    draw(tr_labels, tr_hypo, te_labels, te_hypo, dirpath, t)
    
    save_path = saver.save(sess, "{:s}/CM_tf{:s}.ckpt".format(dirpath, t))
    writer = tf.summary.FileWriter("{:s}/logs".format(dirpath),sess.graph)

    diff = np.subtract(np.array(te_hypo).reshape(-1), np.array(te_labels).reshape(-1))
    rmse = np.sqrt((diff**2).mean())
    max_res = abs(max(diff))    

    g = open("{:s}/summary.txt".format(dirpath),'w')
    g.write("number of training data {:d}/{:d}\n".format(len(tr_hypo),len(tr_hypo)+len(te_hypo)))
    hl = ''
    for i in range(len(HL)):
        hl += "{:d} ".format(HL[i])    
    g.write("Hidden Layer ( {:s})\n".format(hl))
    g.write("rmse : {:f} max_residual : {:f}\n".format(rmse, max_res))
    if args.job == "train" or args.job == "retrain":
        g.write("start : {:s}, end : {:s}\n".format(t, endtrain)) 
    g.write("log scale rmse : {:f} max_residual : {:f}\n".format(np.log10(rmse),np.log10(max_res)))
    g.write("test time for training set : {:s} ~ {:s}\n".format(strt, etrt))
    g.write("test time for test set : {:s} ~ {:s}\n".format(stet, etet))
    g.close()
    if args.job == 'test':
        h = open("score",'w')
        h.write("{:f}\n".format(rmse))
        n_hl = 0
        for nhl in HL:
            n_hl += nhl
        h.write("{:d}".format(n_hl*2))
        h.close()


# In[ ]:


with open('{:s}/te_hypo{:s}.txt'.format(dirpath, t),'w') as f:
    f.write('test_hypothesis, eV\n')
    for i in range(len(te_hypo)):
        f.write('{:f}\n'.format(te_hypo[i][0]))

with open('{:s}/te_labels{:s}.txt'.format(dirpath, t),'w') as g:
    g.write('test_labels, eV\n')
    for i in range(len(te_labels)):
        g.write('{:f}\n'.format(te_labels[i][0]))

with open('{:s}/tr_hypo{:s}.txt'.format(dirpath, t),'w') as f:
    f.write('train_hypothesis, eV\n')
    for i in range(len(tr_hypo)):
        f.write('{:f}\n'.format(tr_hypo[i][0]))

with open('{:s}/tr_labels{:s}.txt'.format(dirpath, t),'w') as g:
    g.write('train_labels, eV\n')
    for i in range(len(tr_labels)):
        g.write('{:f}\n'.format(tr_labels[i]))

with open('{:s}/tr_images{:s}.txt'.format(dirpath, t),'w') as f:
    for i in range(len(tr_images)):
        for j in range(len(tr_images[i])):
            f.write('{:f} '.format(tr_images[i][j]))
        f.write('\n')

with open('{:s}/te_images{:s}.txt'.format(dirpath, t),'w') as f:
    for i in range(len(te_images)):
        for j in range(len(te_images[i])):
            f.write('{:f} '.format(te_images[i][j]))
        f.write('\n')

if args.job == "train" or args.job == "retrain":
    with open('{:s}/val_hypo{:s}.txt'.format(dirpath, t),'w') as f:
        f.write('validation hypothesis, eV\n')
        for i in range(len(val_hypo)):
            f.write('{:f}\n'.format(val_hypo[i][0]))
    
    with open('{:s}/val_labels{:s}.txt'.format(dirpath, t),'w') as g:
        g.write('validation_labels, eV\n')
        for i in range(len(val_labels)):
            g.write('{:f}\n'.format(val_labels[i]))
    
    with open('{:s}/val_images{:s}.txt'.format(dirpath, t),'w') as f:
        for i in range(len(val_images)):
            for j in range(len(val_images[i])):
                f.write('{:f} '.format(val_images[i][j]))
            f.write('\n')

    os.system("cp diff.py lossfunccall.py *sh.e* *sh.o* {:s}/.".format(dirpath))
    os.chdir(dirpath)
    os.system("~/anaconda3/bin/python lossfunccall.py")
    os.system("~/anaconda3/bin/python diff.py")
