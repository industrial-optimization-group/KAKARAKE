#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import plotly.express as ex

# import plotly.graph_objects as go
from kakarake.SCORE import SCORE_bands, auto_SCORE


# In[2]:


import plotly.io as pio

pio.templates.default = "presentation"


# In[5]:


data = pd.read_csv("./data/DTLZ7_3d_true_front.csv")
# data = pd.read_csv("IBM.csv")
# data = data[data.columns[0:5]]
# data[["f7", "f8", "f9"]] = data[["f7", "f8", "f9"]] * -1
"""data.rename(columns={
    "f1":r"$f_1$",
    "f2":r"$f_2$",
    "f3":r"$f_3$",
    "f4":r"$f_4$",
    "f5":r"$f_5$",
    },
           inplace=True)"""
data


# In[8]:


data.to_numpy()


# In[6]:


fig = auto_SCORE(
    data,
    dist_parameter=0.3,
    clustering_algorithm="GMM",
    clustering_score="BIC",
    solutions=True,
    use_absolute_corr=True,
    # distance_formula = 2,
)[0]
fig


# In[17]:


# fig.write_image("test.svg")


# In[5]:


fig.write_image("dtlz7-score.pdf", width=1400, height=800)


# In[6]:


from matplotlib import cm


# In[9]:


plot_color_gradients(
    "Qualitative",
    ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"],
)


# In[11]:


colors = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]

cm.get_cmap(colors[11], 3)


# In[3]:


data = pd.read_csv("../data/Cex1new.ralph.csv")
data


# In[8]:


fig = ex.parallel_coordinates(data)
fig.update_layout(font=dict(size=28))


# In[9]:


fig.write_image("AD1-parcoords.pdf", width=1400, height=800)


# In[10]:


fig = auto_SCORE(
    data,
    dist_parameter=0.3,
    clustering_algorithm="GMM",
    clustering_score="silhoutte",
    solutions=True,
    # use_absolute_corr = True,
    # distance_formula = 2,
)[0]
fig


# In[11]:


fig.write_image("AD1-score.pdf", width=1400, height=800)


# In[4]:


data = pd.read_csv("../data/c432-141.ralph.csv")


# In[16]:


fig = ex.parallel_coordinates(data)
fig.update_layout(font=dict(size=28))


# In[18]:


fig.write_image("AD2-parcoords.pdf", width=1400, height=800)


# In[6]:


fig = auto_SCORE(
    data,
    dist_parameter=0.3,
    clustering_algorithm="GMM",
    clustering_score="silhoutte",
    solutions=True,
    # use_absolute_corr = True,
    # distance_formula = 2,
)[0]
fig


# In[7]:


fig.write_image("AD2-indScore.pdf", width=1400, height=800)


# In[13]:


data = pd.read_csv("../data/DTLZ5.csv")
fig = auto_SCORE(
    data,
    dist_parameter=0.3,
    clustering_algorithm="GMM",
    clustering_score="BIC",
    solutions=True,
    # use_absolute_corr = True,
    # distance_formula = 2,
)[0]
fig


# In[14]:


fig.write_image("DTLZ5-SCORE.pdf", width=1400, height=1200)


# In[10]:


data = pd.read_csv("../data/Obj_arch_nds_1.csv")
fig = auto_SCORE(
    data,
    dist_parameter=0.3,
    clustering_algorithm="GMM",
    clustering_score="BIC",
    solutions=True,
    # use_absolute_corr = True,
    # distance_formula = 2,
)[0]
fig


# In[12]:


fig.write_image("GAA-SCORE-2.pdf", width=1400, height=1000)


# In[ ]:

