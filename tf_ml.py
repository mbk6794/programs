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


def makefeature(tr, fin=None, intype=None, ndata=0):
    if tr == "train":
        molecule_dataframe = pd.read_csv(fin,sep=",")
     
        molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))
        if ndata != None:
            lentotal = ndata
        else:
            lentotal = len(molecule_dataframe)
        lentrain = int(lentotal*0.8)
        lentest = int((lentotal - lentrain)/2)
        lenval = int((lentotal - lentrain)/2)
    
        if intype == 'a':
            columns = []
           
            for i in range(1,21):
                for j in range(i,21):
                    columns.append('{:2d}-{:2d}'.format(i,j))
                
            Feature = np.zeros((len(molecule_dataframe),len(columns)))
     
            for i in range(len(columns)):
                for sample in range(len(molecule_dataframe)):
                    Feature[sample][i] = molecule_dataframe[columns[i]][sample] 
    
        elif intype == 'b':
            columns = []
          
            for i in range(1,21): 
                columns.append('{:2d}-{:2d}'.format(i,i)) 
            for i in range(1,11):
                for j in range(11,21):
                    columns.append('{:2d}-{:2d}'.format(i,j))
                
            Feature = np.zeros((len(molecule_dataframe),len(columns)))
     
            for i in range(len(columns)):
                for sample in range(len(molecule_dataframe)):
                    Feature[sample][i] = molecule_dataframe[columns[i]][sample] 
    
        elif intype == 'c':
            columns = []
     
            for i in range(1,11):
                for j in range(11,21):
                    columns.append('{:2d}-{:2d}'.format(i,j))
                
            Feature = np.zeros((len(molecule_dataframe),len(columns)))
    
            for i in range(len(columns)):
                for sample in range(len(molecule_dataframe)):
                    Feature[sample][i] = molecule_dataframe[columns[i]][sample]
     
        Target = np.zeros((len(molecule_dataframe)))
        for sample in range(len(molecule_dataframe)):
            Target[sample] = abs(molecule_dataframe["Coupling(eV)"][sample]) 
            #Target[sample] = np.log10(abs(molecule_dataframe["Coupling(eV)"][sample]) * 1000)
     
        Target = Target.reshape((len(molecule_dataframe),1))
        
        tr_images, tr_labels, val_images, val_labels, te_images, te_labels = Feature[:lentrain, :], Target[:lentrain, :], Feature[lentrain:lentrain+lenval, :], Target[lentrain:lentrain+lenval, :] ,Feature[lentrain+lenval:lentotal, :], Target[lentrain+lenval:lentotal,:]
        return tr_images, tr_labels, val_images, val_labels, te_images, te_labels, lentrain, lenval, lentotal

    elif tr == 'retrain' or tr == 'test':
        tr_img = glob.glob('tr_images*.txt')[0]
        tr_lab = glob.glob('tr_labels*.txt')[0]
        te_img = glob.glob('te_images*.txt')[0]
        te_lab = glob.glob('te_labels*.txt')[0]
        val_img = glob.glob('val_images*.txt')[0]
        val_lab = glob.glob('val_labels*.txt')[0]
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
        tr_labels = np.array(data)
        tr_labels = tr_labels.reshape((len(tr_labels),1))
    
        # make test_labels
        with open(te_lab,'r') as f: 
            flines = f.readlines()
        data = flines[1:]
        for i in range(len(data)):
            data[i] = eval(data[i])
        te_labels = np.array(data)
        te_labels = te_labels.reshape((len(te_labels),1))
        
        # make validation_labels
        with open(val_lab,'r') as f: 
            flines = f.readlines()
        data = flines[1:]
        for i in range(len(data)):
            data[i] = eval(data[i])
        val_labels = np.array(data)
        val_labels = val_labels.reshape((len(val_labels),1))
        
        return tr_images, tr_labels, val_images, val_labels, te_images, te_labels, len(tr_labels), len(val_labels), len(tr_labels)+len(val_labels)+len(te_labels)

def draw(tr_labels, tr_hypo, te_labels, te_hypo, dirpath, t):
    fig = plt.figure(figsize=(20,10))
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

    ax1.set_xlim([0,tr_lim])
    ax1.set_ylim([0,tr_lim])
    ax2.set_xlim([0,te_lim])
    ax2.set_ylim([0,te_lim])
  
    ax1.scatter(tr_labels, tr_hypo, c='r', s=1)
    ax1.plot([0,tr_lim], [0,tr_lim], c='k')
    ax2.scatter(te_labels, te_hypo, c='r', s=1)
    ax2.plot([0,te_lim], [0,te_lim], c='k')
    
    plt.savefig("{:s}/CM{:s}.png".format(dirpath, t))
# In[ ]:


parser = argparse.ArgumentParser(description='tensorflow training')
parser.add_argument('-j','--job', choices=['train', 'retrain','test'], help='job option: train, retrain or test')
parser.add_argument('-f','--fin', default=None, help='write filename if you want to train first')
parser.add_argument('-hl','--hiddenlayer', nargs='*', type=int, help='hidden layers: ex) 5 5 5 means three layers, five perceptrons each')
parser.add_argument('-t','--intype', choices=['a','b','c'], help='a: inter-, intra-, diag / b: inter-, diag / c: inter-')
parser.add_argument('-nd','--ndata', type=int, default=None, help='the number of total data set, if None, whole data will be used')
parser.add_argument('-epoch','--epoch', type=int, help='max number of epochs')
parser.add_argument('-bs','--batchsize', type=int, help='batch size')

args = parser.parse_args()

tr_images, tr_labels, val_images, val_labels, te_images, te_labels, lentrain, lenval, lentotal = makefeature(args.job, args.fin, args.intype, args.ndata)

# In[ ]:
_, n_inputs = tr_images.shape
HL = tuple(args.hiddenlayer)
n_hidden = []
n_output = 1


# In[ ]:


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")


# In[ ]:


with tf.name_scope("dnn"):
    if len(HL) == 1:
        n_hidden.append(tf.layers.dense(X, HL[0], name="hidden{:d}".format(1), activation=tf.nn.relu))
        n_hidden.append(tf.layers.dense(n_hidden[0], n_output, name="output", activation=tf.nn.relu))
    for i in range(len(HL)):
        if i == 0:
            n_hidden.append(tf.layers.dense(X, HL[i], name="hidden{:d}".format(i+1), activation=tf.nn.relu))
        elif i>0 and i<len(HL)-1:
            n_hidden.append(tf.layers.dense(n_hidden[i-1], HL[i], name="hidden{:d}".format(i+1), activation=tf.nn.relu))
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
saver = tf.train.Saver(max_to_keep=10)


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
    idx = np.arange(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(idx, n_batches):
        X_batch, y_batch = X[batch_idx,:], y[batch_idx,:]
        yield X_batch, y_batch


# In[ ]:

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
with tf.Session(config=config) as sess:
    t = datetime.utcnow().strftime("%m%d%H%M%S")
    dirpath = "{:s}_{:d}".format(t, lentotal)
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
                loss_, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
            if epoch%100 == 0:
                loss_tr = sess.run(loss, feed_dict={X:tr_images, y:tr_labels})
                loss_valid, val_hypo = sess.run([loss, n_hidden[-1]], feed_dict={X:val_images, y:val_labels})
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
                
            if loss_tr < 1e-10:
                break
    
        f.write("final loss: {:f}\n".format(loss_tr))
        endtrain = datetime.utcnow().strftime("%m%d%H%M%S")
        f.close()
    tr_labels = tr_labels.reshape(lentrain)
    val_labels = val_labels.reshape(lenval)

    strt = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    tr_hypo = sess.run(n_hidden[-1], feed_dict={X:tr_images})
    etrt = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    stet = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    te_hypo = sess.run(n_hidden[-1], feed_dict={X:te_images})
    etet = datetime.utcnow().strftime("%m/%d %H:%M:%S.%f")
    tr_hypo = np.array(tr_hypo).reshape((lentrain,1))
    te_hypo = np.array(te_hypo).reshape((lentotal-(lentrain+lenval),1))
    
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


# In[ ]:


with open('{:s}/te_hypo{:s}.txt'.format(dirpath, t),'w') as f:
    f.write('test hypothesis, eV\n')
    for i in range(len(te_hypo)):
        f.write('{:f}\n'.format(te_hypo[i][0]))

with open('{:s}/te_labels{:s}.txt'.format(dirpath, t),'w') as g:
    g.write('test_labels, eV\n')
    for i in range(len(te_labels)):
        g.write('{:f}\n'.format(te_labels[i][0]))

with open('{:s}/tr_hypo{:s}.txt'.format(dirpath, t),'w') as f:
    f.write('train hypothesis, eV\n')
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
    os.system("cp ../adamop.sh .")
