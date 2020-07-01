#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


# copy data frame
df = black_friday.copy()


# In[4]:


# data frame info()
df.info()


# In[5]:


# percent missing values
(df.isnull().sum()/df.shape[0])*100


# In[6]:


# data frame first entries
df.head()


# In[7]:


# check column age object
df['Age'].value_counts()


# In[8]:


# column age percent
(df['Age'].value_counts()/df.shape[0])*100


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    shape = df.shape
    return shape
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    woman = df[(df['Gender']=='F') & (df['Age'] =='26-35')]
    return len(woman) 
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    user = df['User_ID'].nunique()
    return user
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[12]:


def q4():
    # Retorne aqui o resultado da questão 4.
    data_types = df.dtypes.nunique()
    return int(data_types)
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[13]:


def q5():
    # Retorne aqui o resultado da questão 5.
    reg = len(df[df.isnull().any(axis = 1)])
    percent = reg/df.shape[0]
    return float(percent)
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[14]:


def q6():
    # Retorne aqui o resultado da questão 6.
    max_null = df.isna().sum().max()
    return max_null
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


def q7():
    # Retorne aqui o resultado da questão 7.
    freq_value = int(df['Product_Category_3'].value_counts().idxmax())
    return freq_value
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[16]:


def q8():
    # Retorne aqui o resultado da questão 8.
    norm = (df.Purchase - df.Purchase.min())/(df.Purchase.max() - df.Purchase.min())
    return float(norm.mean())
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[17]:


def q9():
    # Retorne aqui o resultado da questão 9.
    z = (df['Purchase'] - df['Purchase'].mean())/df['Purchase'].std()
    result = z[(z >= -1) & (z <= 1)]
    return int(result.count())
q9() 


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[18]:


def q10():
    # Retorne aqui o resultado da questão 10.
    p = df[(df.Product_Category_2.isnull() & df.Product_Category_3.isnull()) ]
    p2 = df[df.Product_Category_2.isnull()]
    return  p2.Product_Category_2.equals(p.Product_Category_3)
q10()

