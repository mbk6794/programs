#!/usr/bin/env python
# coding: utf-8

# In[2]:


with open('amp-log.txt', 'r') as f:
    flines = f.readlines()
for i in range(len(flines)):
    if ' Step                Time   Loss (SSD)   EnergyRMSE     MaxResid' in flines[i]:
        sidx = i
    if '...optimization unsuccessful.' in flines[i]:
        eidx = i
        break
    else:
        eidx = -1
        
data = flines[sidx+2:eidx]


# In[8]:


strl = []
for i in range(len(data)):
    if 'Overwriting' in data[i]:
        strl.append(i)
    if 'Saving' in data[i]:
        strl.append(i)


# In[10]:


strl.reverse()
for i in range(len(strl)):
    data.pop(strl[i])


# In[13]:


for i in range(len(data)):
    data[i] = data[i].split(' ')
    while '' in data[i]:
        data[i].remove('')


# In[ ]:


import numpy as np
Loss = []
RMSE = []
Maxres = []
for i in range(len(data)):
    Loss.append(eval(data[i][2]))
    RMSE.append(eval(data[i][3]))
    Maxres.append(eval(data[i][5]))
Loss = np.log10(Loss)
RMSE = np.log10(RMSE)
Maxres = np.log10(Maxres)


# In[23]:


setnum = np.ones(len(Loss)) * np.log10(1e-4 / 22)


# In[25]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.switch_backend('agg')
fig = plt.figure(figsize=(10,10))
plt.xlabel('loss function call')
plt.ylabel('convergence log(eV)')
p1 = plt.plot(range(len(Loss)), Loss, c='grey', label='loss function')
p2 = plt.plot(range(len(RMSE)), RMSE, c='b', lw=2, label='energy rmse')
p3 = plt.plot(range(len(Maxres)), Maxres, c='b', ls=':', label='energy maxresid')
p4 = plt.plot(range(len(setnum)), setnum, c='cornflowerblue', lw=1)
plt.legend()

plt.savefig("convergence.png")


# In[ ]:




