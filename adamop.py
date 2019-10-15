#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython import display
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.switch_backend('agg')
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse


# In[ ]:


def train(fin):
    molecule_dataframe = pd.read_csv(fin,sep=",")

    molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))

    lentotal = len(molecule_dataframe)
    lentrain = int(lentotal*0.8)

    columns = []

    for i in range(1,11):
        for j in range(11,21):
            columns.append(str(f'{i:2d}')+'-'+str(f'{j:2d}'))
        
    Feature = np.zeros((len(molecule_dataframe),len(columns)))

    for i in range(len(columns)):
        for sample in range(len(molecule_dataframe)):
            Feature[sample][i] = molecule_dataframe[columns[i]][sample]

    Target = np.zeros((len(molecule_dataframe)))
    for sample in range(len(molecule_dataframe)):
        #Target[sample] = abs(molecule_dataframe["Coupling(eV)"][sample]) * 1000
        Target[sample] = np.log10(abs(molecule_dataframe["Coupling(eV)"][sample]) * 1000)

    Target = Target.reshape((lentotal,1))
    
    tr_images, tr_labels, te_images, te_labels = Feature[:lentrain, :], Target[:lentrain, :], Feature[lentrain:, :], Target[lentrain:,:]
    return tr_images, tr_labels, te_images, te_labels, lentrain, lentotal


# In[ ]:


def retrain():
    import glob
    tr_img = glob.glob('tr_images*.txt')[0]
    tr_lab = glob.glob('tr_labels*.txt')[0]
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

    # make training_labels
    with open(tr_lab,'r') as f: 
        flines = f.readlines()
    data = flines[1:]
    for i in range(len(data)):
        data[i] = eval(data[i])
    tr_labels = np.array(data)
    tr_labels = np.log10(tr_labels.reshape((len(tr_labels),1)))

    # make test_labels
    with open(te_lab,'r') as f: 
        flines = f.readlines()
    data = flines[1:]
    for i in range(len(data)):
        data[i] = eval(data[i])
    te_labels = np.array(data)
    te_labels = np.log10(te_labels.reshape((len(te_labels),1)))
    
    return tr_images, tr_labels, te_images, te_labels, len(tr_labels), len(tr_labels)+len(te_labels)


# In[ ]:


parser = argparse.ArgumentParser(description='tensorflow training')
parser.add_argument('-j','--job', choices=['train', 'retrain'], help='job option: "train" vs "retrain"')
parser.add_argument('-f','--file', default=None, help='write filename if you want to train first')

args = parser.parse_args()

if args.job == 'train':
    tr_images, tr_labels, te_images, te_labels, lentrain, lentotal = train(args.file)
    
elif args.job == 'retrain':
    tr_images, tr_labels, te_images, te_labels, lentrain, lentotal = retrain()


# In[ ]:


n_inputs = 10**2
n_hidden1 = 100 
n_hidden2 = 100 
n_hidden3 = 100 
n_hidden4 = 100 
n_hidden5 = 100 
n_output = 1


# In[ ]:


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")


# In[ ]:


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.relu)
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5", activation=tf.nn.relu)
    output = tf.layers.dense(hidden5, n_output, name="output")


# In[ ]:

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator 

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.squared_difference(output, y), name="loss")


# In[ ]:


learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[ ]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[ ]:


n_epochs = 20000
batch_size = 100


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
    print("Session run!")
    import time
    t = time.strftime("%m%d%H%M", time.localtime(time.time()))

    ckpt = tf.train.get_checkpoint_state('.')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else: 
        sess.run(init)

    import os
    os.mkdir(t+"log")
    dirpath = t+"log"
    #f = open(dirpath+'/CM_tf_log'+t+'.txt','w')            
    #for epoch in range(n_epochs):
    #    for X_batch, y_batch in batch(tr_images, tr_labels, batch_size):
    #        loss_, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
    #    if epoch%100 == 0:
    #        f.write(str(epoch)+" loss: "+str(loss_)+"\n")
    #    if loss_ < 1e-5:
    #        break

    #print(epoch, "batch data accuracy:", acc_batch, "valid set accuracy:", acc_valid)
    #f.write("final loss: "+str(loss_))
    #f.close()
   
    tr_hypo = sess.run(output, feed_dict={X:tr_images})
    te_hypo = sess.run(output, feed_dict={X:te_images})
    tr_hypo = np.array(tr_hypo).reshape((lentrain,1))
    te_hypo = np.array(te_hypo).reshape((lentotal-lentrain,1))
    
    tr_labels = tr_labels.reshape(lentrain)
    
    fig = plt.figure(figsize=(20,10))
    #gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1,1])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #ax1 = plt.subplot(gs[0])
    #ax2 = plt.subplot(gs[1])
    ax1.title.set_text("CM_train")
    ax2.title.set_text("CM_test")
    ax1.set_ylabel("hypothesis (meV)")
    ax1.set_xlabel("labels (meV)")
    ax2.set_ylabel("hypothesis (meV)")
    ax2.set_xlabel("labels (meV)")

    tr_lim = max(max(np.power(10,tr_labels)), max(np.power(10,tr_hypo))) 
    te_lim = max(max(np.power(10,te_labels)), max(np.power(10,te_hypo)))

    ax1.set_xlim([0,tr_lim])
    ax1.set_ylim([0,tr_lim])
    ax2.set_xlim([0,te_lim])
    ax2.set_ylim([0,te_lim])
  
    ax1.scatter(np.power(10,tr_labels), np.power(10,tr_hypo), c='r', s=1)
    ax1.plot([0,tr_lim], [0,tr_lim], c='k')
    ax2.scatter(np.power(10,te_labels), np.power(10,te_hypo), c='r', s=1)
    ax2.plot([0,te_lim], [0,te_lim], c='k')
    
    plt.savefig(dirpath+"/CM_log"+t+".png")
    
    #save_path = saver.save(sess, dirpath+"/CM_tf_log"+t+".ckpt")
    #writer = tf.summary.FileWriter(dirpath+"/logs",sess.graph)


# In[ ]:


'''
data = np.log(Target)
plt.figure()
#plt.hist(data, bins=50)
#plt.scatter(range(len(data)), data, c='r', s=1)
plt.show()
'''


# In[ ]:

'''
with open(dirpath+'/te_hypo'+t+'.txt','w') as f:
    f.write('test hypothesis, meV\n')
    for i in range(len(te_hypo)):
        f.write(str(np.power(10,te_hypo[i][0]))+'\n')

with open(dirpath+'/te_labels'+t+'.txt','w') as g:
    g.write('test_labels, meV\n')
    for i in range(len(te_labels)):
        g.write(str(np.power(10,te_labels[i][0]))+'\n')

with open(dirpath+'/tr_hypo'+t+'.txt','w') as f:
    f.write('train hypothesis, meV\n')
    for i in range(len(tr_hypo)):
        f.write(str(np.power(10,tr_hypo[i][0]))+'\n')

with open(dirpath+'/tr_labels'+t+'.txt','w') as g:
    g.write('train_labels, meV\n')
    for i in range(len(tr_labels)):
        g.write(str(np.power(10,tr_labels[i]))+'\n')

with open(dirpath+'/tr_images'+t+'.txt','w') as f:
    for i in range(len(tr_images)):
        for j in range(len(tr_images[i])):
            f.write(str(tr_images[i][j])+' ')
        f.write('\n')

with open(dirpath+'/te_images'+t+'.txt','w') as f:
    for i in range(len(te_images)):
        for j in range(len(te_images[i])):
            f.write(str(te_images[i][j])+' ')
        f.write('\n')
'''
