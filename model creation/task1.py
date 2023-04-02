#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Ajay Ugale
# #### Student ID: S3940207
# 
# Date: 02/10/2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# You should give a brief information of this assessment task here.
# 
# <span style="color: red"> Note that this is a sample notebook only. You will need to fill in the proper markdown and code blocks. You might also want to make necessary changes to the structure to meet your own needs. Note also that any generic comments written in this notebook are to be removed and replace with your own words.</span>

# ## Importing libraries 

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import re
import os 
from collections import Counter


# ### 1.1 Examining and loading data
# - xamine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# In[2]:


# Identify the categories of job advertisement

print("The categories of the job adverstisement are: ")
for branch in [branches for branches in os.listdir('./data') if branches!='.DS_Store']:
    print(branch)

# Job advertisemetn file count for each category
for branch in [branches for branches in os.listdir('./data') if branches!='.DS_Store']:
    jd = os.listdir(f'./data/{branch}')
    print(f"Job description file count for the category {branch} is {len(jd)}")
        


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# ...... Sections and code blocks on basic text pre-processing
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[3]:


global_text = []
for branch in [branches for branches in os.listdir('./data') if branches!='.DS_Store']:
    for jd in os.listdir(f'./data/{branch}'):
        with open(os.path.join('./data', branch, jd), "r", encoding='utf-8') as f:
            text = f.read()
            # Fetch the job description from the file
            text = text.split('Description: ', 1)[1]            
            # Regular expression for tokenization of each job description
            reg = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
            # Remove words that have length less than 2
            filtered = [word for word in re.findall(reg, text.lower()) if len(word)>=2]
            with open("stopwords_en.txt", "r") as f:
              stopwords = f.readlines()
            # Remove the stopwords 
            stopwords = [line.replace('\n', '') for line in stopwords]
            filtered = [word for word in filtered if word not in stopwords]
            # Remove words having frequency 1
            counts = Counter(filtered)
            filtered = [word for word in filtered if counts[word]!=1]
            global_text.extend(filtered)

global_count = Counter(global_text)
# Remove the top 50 most repeated words from the document
top_50_freq = [word for word, count in global_count.most_common()[:50]]
global_text = [word for word in global_text if word not in top_50_freq]


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[4]:


# Only include unique values with their index
global_text = np.array(global_text)
words = np.unique(global_text)

# Include index starting from 0 
output = []
for idx, word in enumerate(words):
    output.append(f"{word}:{idx}")

output = "\n".join(output)

# Save to disk
with open('vocab.txt', 'w') as f:
    f.write(output)


# ## Summary
# 
# Here, we explored the files to see the file structure. We preprocessed the job description by tokenizing with regular expression, lower case, removing words having length less than 2, removing the stopwords present, removing words having frequency 1 in each document and removing the top 50 frequent words from all the documents. Then we created a vocabulary and saved it.
