#!/usr/bin/env python
# coding: utf-8

# ## Pre-requirements and presentation functions

# ## https://simpletransformers.ai/docs/usage/#loading-a-local-save

# xlnet based cased - 50 - 78% on 2nd epoch

# In[1]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import pandas as pd
import numpy as np
import os

# figure plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "figures"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[2]:


def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.title(title)


# In[16]:


df_fpb = pd.read_csv("./Sentences_75Agree.txt", sep='@',encoding='latin-1', names=['Text','Rating'])


# In[17]:


df_fpb.head()


# In[18]:


len(df_fpb)


# In[19]:


df_fpb = sklearn.utils.shuffle(df_fpb, random_state=42)


# In[20]:


df_fpb.head()


# In[21]:


"""Changed the getlabel function in binaryprocessor class to have 3 labels, negative, neutral, positive"""
df_fpb['Rating'] = df_fpb['Rating'].replace('negative',0)
df_fpb['Rating'] = df_fpb['Rating'].replace('neutral',1)
df_fpb['Rating'] = df_fpb['Rating'].replace('positive',2)


# In[22]:


df_fpb


# In[33]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_fpb, test_size=0.2, random_state=42)


# In[24]:


df_train


# ## BERT

# In[25]:


from simpletransformers.classification import ClassificationModel


# In[26]:


# Create a ClassificationModel
cuda_available = torch.cuda.is_available()

model_group = 'roberta'
model_spec = 'roberta-large'

model = ClassificationModel(
    model_group, model_spec, num_labels=3, args={"reprocess_input_data": True, "overwrite_output_dir": False, "num_train_epochs":4, "evaluate_during_training_verbose":True, "evaluate_during_training":True, "manual_seed":42}, use_cuda=cuda_available
)


# In[27]:


# Train the model
model.train_model(df_train)


# In[28]:


# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(df_test)

df_model_outputs = pd.DataFrame(model_outputs, columns=['negative', 'neutral', 'positive'])

df_model_outputs.to_csv('output ' + model_spec)

# In[29]:


result


# In[30]:


model_outputs


# In[32]:


len(model_outputs)


# In[31]:


len(wrong_predictions)


# In[ ]:


df_train = df_train.reset_index()
df_test = df_test.reset_index()


# In[ ]:


df_test


# In[ ]:


# Uses the array model outputs to pick the location of the max one and thus the prediction
predictions = []
for i in model_outputs:
    predictions.append(np.argmax(i)) 


# In[ ]:


df_test['predictions']=predictions


# In[ ]:


df_test[['Rating','predictions']][0:8]


# In[ ]:


correct=[]
for index, row in df_test.iterrows():
    if(row['Rating'] == (row['predictions'])):
        correct.append('True')
    else:
        correct.append('False')


# In[ ]:


df_test['correct']=correct


# In[ ]:


from collections import Counter 

Counter(df_test['correct'])


# In[ ]:


df_test.loc[6].Text


# In[ ]:


pd.set_option('display.max_colwidth', None)


# In[ ]:


df_test[0:7]


# In[ ]:


df_test['correct']


# In[ ]:




