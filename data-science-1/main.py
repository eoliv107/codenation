#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.
df = dataframe.copy()


# In[5]:


# first entries
df.head()


# In[6]:


# data frame shape
df.shape


# In[7]:


# summary statistic
df.describe()


# In[8]:


# columns type
df.dtypes


# In[9]:


# data frame info
df.info()


# In[10]:


fig, axes = plt.subplots(ncols=len(df.columns), figsize=(10,5))
for col, ax in zip(df, axes):
    df[col].plot.hist(ax=ax, title=col)
plt.tight_layout()    
plt.show()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[11]:


def q1():
    q_normal = df.normal.quantile([0.25,0.5,0.75])
    q_binomial = df.binomial.quantile([0.25,0.5,0.75])
    result = (q_normal - q_binomial).values
    result = np.around(result, decimals=3)
    return tuple(result)
q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[12]:


interval1 = df.normal.mean() - df.normal.std()
interval2 = df.normal.mean() + df.normal.std()


# In[27]:


def q2():
    interval1 = df.normal.mean() - df.normal.std()
    interval2 = df.normal.mean() + df.normal.std()
    ecdf = ECDF(df.normal)
    ecdf1 = ecdf(interval1)
    ecdf2 = ecdf(interval2)
    results = float(round((ecdf2 - ecdf1),3))  
    return results
q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# In[14]:


# [𝑥¯−2𝑠,𝑥¯+2𝑠]
inter1 = df.normal.mean() - 2 * df.normal.std()
inter2 = df.normal.mean() + 2 * df.normal.std()
ecdf = ECDF(df.normal)
e1 = ecdf(inter1)
e2 = ecdf(inter2)
res1 = round((e2 - e1),3)
res1


# In[15]:


# [𝑥¯−3𝑠,𝑥¯+3𝑠] 
inter3 = df.normal.mean() - 3 * df.normal.std()
inter4 = df.normal.mean() + 3 * df.normal.std()
ecdf = ECDF(df.normal)
e3 = ecdf(inter3)
e4 = ecdf(inter4)
res2 = round((e4 - e3),3)
res2


# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[16]:


def q3():
    m_norm = df.normal.mean()
    v_norm = df.normal.std()**2
    m_binom = df.binomial.mean()
    v_binom = df.binomial.std()**2
    diference = (round(m_binom - m_norm,3),
                 round(v_binom - v_norm,3))
    return diference
q3()  


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[17]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[18]:


# data frame first entries
data = stars.copy()
data.head()


# In[19]:


# data frame shape
data.shape


# In[20]:


# summary statistics
data.describe()


# In[21]:


# data frame info
data.info()


# In[22]:


fig = plt.figure()
d = data.drop(['target'], axis=1)
ax = fig.gca()
d.hist(ax=ax)
plt.show();


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[23]:


# filter and standadization
s = data.mean_profile[data.target.eq(0)]
s_norm = (s - s.mean())/s.std()


# In[24]:


def q4():
    ecdf = ECDF(s_norm)
    results = ([round(ecdf(i), 3) for i in sct.norm.ppf([0.80, 0.90, 0.95])])
    return tuple(results)    
q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[25]:


def q5():
    x = s_norm.quantile([0.25,0.5,0.75])
    y = sct.norm.ppf((0.25,0.5,0.75))
    r = round((x - y),3)
    return tuple(r)
q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
