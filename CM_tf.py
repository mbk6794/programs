#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import display
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import tensorflow as tf
import pandas as pd


# In[2]:


molecule_dataframe = pd.read_csv('40000_CM',sep=",")


# In[3]:


molecule_dataframe = molecule_dataframe.reindex(np.random.permutation(molecule_dataframe.index))


# In[4]:


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
    Target[sample] = np.log(abs(molecule_dataframe["Coupling(eV)"][sample]) * 1000)
        


# In[5]:


Target = Target.reshape((lentotal,1))
tr_images, tr_labels, te_images, te_labels = Feature[:lentrain, :], Target[:lentrain, :], Feature[lentrain:, :], Target[lentrain:,:]


# In[6]:


n_inputs = 10**2
n_hidden1 = 128
n_hidden2 = 128
n_hidden3 = 128
n_hidden4 = 128
n_output = 1


# In[7]:


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")


# In[8]:


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.relu)
    output = tf.layers.dense(hidden4, n_output, name="output")


# In[9]:


with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.squared_difference(output, y), name="loss")


# In[10]:


learning_rate = 0.001

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[11]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[12]:


n_epochs = 100000
batch_size = 10


# In[21]:


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


# In[22]:


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for X_batch, y_batch in batch(tr_images, tr_labels, batch_size):
            loss_, h_, _ = sess.run([loss, output, training_op], feed_dict={X: X_batch, y: y_batch})
        if epoch%100 == 0:
            print(epoch, "loss:", loss_, "output: ", h_)
    #print(epoch, "batch data accuracy:", acc_batch, "valid set accuracy:", acc_valid)
    print("final loss: ", loss_)
    
    hypo = sess.run(output, feed_dict={X:te_images})
    hypo = np.array(hypo).reshape((lentotal-lentrain,1))
    
    tr_labels = tr_labels.reshape(lentrain)
    h_ = h_.reshape(lentrain)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.title.set_text("CM_train")
    ax2.title.set_test("CM_test")
    ax1.set_ylabel("hypothesis")
    ax1.set_xlabel("labels")
    ax2.set_ylabel("hypothesis")
    ax2.set_xlabel("labels")
    
    ax1.scatter(tr_labels, h_, c='r', s=1)
    ax1.plot(tr_labels, tr_labels, c='k')
    ax2.scatter(te_labels, hypo, c='r', s=1)
    ax2.plot(te_labels, te_labels, c='k')
    
    import time
    t = time.strftime("%m%d%H%M", time.localtime(time.time()))
    plt.savefig("CM"+t+".png")
    


# In[ ]:


'''
data = np.log(Target)
plt.figure()
#plt.hist(data, bins=50)
#plt.scatter(range(len(data)), data, c='r', s=1)
plt.show()
'''

