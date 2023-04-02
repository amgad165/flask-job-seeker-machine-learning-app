#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
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

from zeugma.embeddings import EmbeddingTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ...... Sections and code blocks on buidling different document feature represetations
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# <h4> Using bag of words approach for generating count vectors </h4>

# In[2]:


global_text = []
global_title = []
y = []

# Regular expression to tokenize the job description
def data_preprocessing(text):
    reg = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    filtered = [word for word in re.findall(reg, text.lower())]
    with open("stopwords_en.txt", "r") as f:
        stopwords = f.readlines()
        # Remove the stopwords 
        stopwords = [line.replace('\n', '') for line in stopwords]
        filtered = [word for word in filtered if word not in stopwords]

        # Find the word counts
        output = np.unique(np.array(filtered), return_counts=True)
        return output[0], output[1]


def convert_dict(words, counts):
    # Creaing a dictionary of vocabulary for easy data manipulation
    vocab_dict = {}
    output_dict = {}

    # Read the vocabulary text
    with open('vocab.txt') as f:
        vocab = f.readlines()
    vocab = [v.replace('\n', '') for v in vocab]

    for v in vocab:
        word, count = v.split(':')
        vocab_dict[word] = int(count)

    for w, c in zip(words, counts):
        if w in vocab_dict.keys():
            output_dict[vocab_dict[w]] = c
    # Sort the output dictionary according to key values
    output_dict = dict(sorted(output_dict.items(), key=lambda x:x[0]))
    return output_dict


for branch in [branches for branches in os.listdir('./data') if branches!='.DS_Store']:
    for jd in os.listdir(f'./data/{branch}'):
        y.append(branch)
        with open(os.path.join('./data', branch, jd), "r", encoding='utf-8') as f:
            text_full = f.read()
            # Fetch the job description from the file
            text = text_full.split('Description: ', 1)[1]
            # Fetch the title from the file
            text_title = text_full.split('\n')[0].split("Title: ")[1]
            
        words, counts = data_preprocessing(text)
        words_title, counts_title = data_preprocessing(text_title)
        
        output_dict = convert_dict(words, counts)
        output_dict_title = convert_dict(words_title, counts_title)
        
        global_title.append(output_dict_title)
        
        # Feteching the webindex
        webindex = re.findall(r"[0-9]+",text_full.split('\n')[1])
        count_vectors = ""
        count_vectors = f"{count_vectors}\n#{webindex[0]}"
        for k, v in output_dict.items():
            count_vectors = f"{count_vectors},{k}:{v}"
        # Append all the individual feature representative to global one
        global_text.append(count_vectors)


# <h4> Using TFIDF Approach for generating feature vectors </h4>

# In[14]:


#Using TFIDF for generating weighted feature representation

def data_preprocessing(text):
    reg = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    filtered = [word for word in re.findall(reg, text.lower())]
    with open("stopwords_en.txt", "r") as f:
        stopwords = f.readlines()
        # Remove the stopwords 
        stopwords = [line.replace('\n', '') for line in stopwords]
        filtered = [word for word in filtered if word not in stopwords]
        return filtered
    
X = []
X_title = []
y_tfidf = []
X_title_description = []

for branch in [branches for branches in os.listdir('./data') if branches!='.DS_Store']:
    for jd in os.listdir(f'./data/{branch}'):
        y_tfidf.append(branch)
        with open(os.path.join('./data', branch, jd), "r", encoding='utf-8') as f:
            text_full = f.read()
            # Fetch the job description from the file
            text = text_full.split('Description: ', 1)[1]

            # Fetch the title from the file
            text_title = text_full.split('\n')[0].split("Title: ")[1]
            
        filtered_text = data_preprocessing(text)
        X.append(" ".join(filtered_text))
        filtered_title = data_preprocessing(text_title)
        X_title.append(" ".join(filtered_title))
        filtered_title_description = filtered_text + filtered_title
        X_title_description.append(" ".join(filtered_title_description))

        
# Feature representation using TFIDF on description
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Feature representation using TFIDF on title
X_tfidf_title = vectorizer.fit_transform(X_title)

# Feature representation using TFIDF on title+description
X_tfidf_title_description = vectorizer.fit_transform(X_title_description)


# <h4>Using Glove word embeddings for generating unweighted feature representation </h4>

# In[4]:


glove = EmbeddingTransformer('glove')
X_train_glove = glove.transform(X)
X_train_title_glove = glove.transform(X_title)


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[5]:


# Save the bag of words output to the disk
with open('count_vectors.txt', 'w') as f:
    f.write("".join(global_text))


# ## Task 3. Job Advertisement Classification

# ...... Sections and code blocks on buidling classification models based on different document feature represetations. 
# Detailed comparsions and evaluations on different models to answer each question as per specification. 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# <h3> 01: Language model comparisons </h3>

# <h4> Using Bag of words approach</h4>

# In[6]:


# Open the vocabulary file to count the number of features
with open('vocab.txt') as f:
    n_features = len(f.readlines()) + 1
with open('count_vectors.txt') as f:
    count_vectors = f.read()

# Initialize pandas dataframe with features as columns and data points as rows
df = pd.DataFrame(columns=[np.arange(0, n_features-1)])

# Add all the data points in rows
for i, row in enumerate(count_vectors.split('\n')[1:]):
    row = row.split(',')[1:]
    for c in row: 
        df.loc[i, df[int(c.split(':')[0])]] = c.split(':')[1]
        
# Fill missing data with 0
df = df.fillna(0)

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# <h4> Using TFIDF approach </h4>

# In[7]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tfidf, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# <h4> Using Glove Word Embeddings Approach </h4>

# In[8]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train_glove, y_tfidf, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# <p> From the above results, we can see that the TFIDF generated feature vectors provided the best accuracy </p>

# <h3> Q2: Does more information provide higher accuracy? </h3>

# <h4> Using K-Fold Validation for choosing best language model </h4>

# In[9]:


# On Bag of words
scores = cross_val_score(lr, df, y, cv=5)
print("Bag of Words Validation Scores: ", scores)

# On TFIDF
scores = cross_val_score(lr, X_tfidf, y_tfidf, cv=5)
print("TFIDF Validation Scores: ", scores)

# On Glove Word Embeddings
scores = cross_val_score(lr, X_train_glove, y_tfidf, cv=5)
print("Glove Word Embeddings Validation Scores: ",scores)


# <p> As we can see that the TFIDF provides the best scores, so we'll do further analysis using the TFIDF weighted vectors </p>

# <h4> Using only Job Title</h4>

# In[16]:


print("*"*20, "Using Title", "*"*20)

# Splitting the dataset into train and test
X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(X_tfidf_title, y_tfidf, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# <h4> Using only Job Description</h4>

# In[17]:


print("*"*20, "Using Description", "*"*20)
# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tfidf, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# In[18]:


# Using Title & Description combined
print("*"*20, "Using Title & Description", "*"*20)

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_title_description, y_tfidf, test_size=0.2, random_state=77)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train+X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Classification Report using Logistic Regression Classifier")
print(classification_report(y_test, y_pred_lr))


# ## Summary
# In the above code blocks, for task 2 I have first generated the feature vectors of only the description using bag of words, TFIDF and Glvoe word embeddings, then I have applied Logistic Regression classfier on all three feature vectors and I found that the best performance (88%) was achieved using the TFIDF approach.
# 
# For Task 3, first I have used K-flod Cross Validation on all three types of feature vectors to check which was the best one for further analysis and again TFIDF gave the best validation scores of around 90%, so, I used TFIDF first over only the title feature vectors, then on the description feature vectors and then on both of them combined to check whether adding more information has any effect on the accuracy. From the above results we can see that I got the best accuracy (90%) from using only the title feature vectors,  both of the other appraoches max. accuracy was 88%, so, from these results I can deduce that increasing the amount of information doesn't mean that the model will be more accurate.
