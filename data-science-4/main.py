#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()


# In[3]:


df = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

df.columns = new_column_names

df.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[5]:


# Substituindo ',' por '.' e mudando as colunas para o tipo float
df['Pop_density'] = df['Pop_density'].str.replace(',', '.').astype(float)
df['Coastline_ratio'] = df['Coastline_ratio'].str.replace(',', '.').astype(float)
df['Net_migration'] = df['Net_migration'].str.replace(',', '.').astype(float)
df['Infant_mortality'] = df['Infant_mortality'].str.replace(',', '.').astype(float)
df['Literacy'] = df['Literacy'].str.replace(',', '.').astype(float)
df['Phones_per_1000'] = df['Phones_per_1000'].str.replace(',', '.').astype(float)
df['Arable'] = df['Arable'].str.replace(',', '.').astype(float)
df['Crops'] = df['Crops'].str.replace(',', '.').astype(float)
df['Other'] = df['Other'].str.replace(',', '.').astype(float)
df['Climate'] = df['Climate'].str.replace(',', '.').astype(float)
df['Birthrate'] = df['Birthrate'].str.replace(',', '.').astype(float)
df['Deathrate'] = df['Deathrate'].str.replace(',', '.').astype(float)
df['Agriculture'] = df['Agriculture'].str.replace(',', '.').astype(float)
df['Industry'] = df['Industry'].str.replace(',', '.').astype(float)
df['Service'] = df['Service'].str.replace(',', '.').astype(float)

# Removendo os espaços no começo e final
df['Country'] = df['Country'].str.strip()
df['Region'] = df['Region'].str.strip()


# ## Inicia sua análise a partir daqui

# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.Region.value_counts(ascending=True).plot(kind='barh');


# In[10]:


df.isna().sum()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[11]:


def q1():
    regions = df.Region.unique()
    return sorted(regions)
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    x = df['Pop_density'].values
    x = x.reshape(-1,1)
    kbd = KBinsDiscretizer(n_bins=10, 
                           encode='ordinal', 
                           strategy='quantile')
    kbd.fit(x)
    perc90 = np.percentile(x, 90)
    country = 0
    for i in x:
        if i > perc90:
            country +=1
    
    return country
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[13]:


def q3():
    x = df.Climate.nunique(dropna=False)
    y = df.Region.nunique(dropna=False)
    one_hot = x + y
    return one_hot

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[14]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[15]:


# Arable index
df.columns.get_loc("Arable")


# Each element from the list above represents a column from a dataframe.
# Our column `Arable` has the index 11. But solving the problem we removed
# two elements from our list so that's why was used transform[0][9]  

# In[16]:


def q4():
    cols = df.select_dtypes(['int64', 'float64']).columns
    pipe = Pipeline(steps = [('imputer', 
                              SimpleImputer(strategy = 'median')),
                             ('scaler', 
                              StandardScaler())])
    pipe.fit(df[cols])
    transform = pipe.transform([test_country[2:]])
    arable = round(transform[0][9],3)
    return float (arable)

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[17]:


def q5():
    Q1 = df.Net_migration.quantile(0.25)
    Q3 = df.Net_migration.quantile(0.75)
    IQR = Q3 - Q1

    low = int((df.Net_migration < (Q1 - 1.5 * IQR)).sum())
    up_ = int((df.Net_migration > (Q3 + 1.5 * IQR)).sum())
    remove_ = bool(False)

    return tuple((low, up_, remove_))
    
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[18]:


categories = ['sci.electronics', 
              'comp.graphics', 
              'rec.motorcycles']

newsgroup = fetch_20newsgroups(subset="train",
                               shuffle=True,
                               categories=categories,  
                               random_state=42)


# In[19]:


def q6():
    vector = CountVectorizer()
    train_ = vector.fit_transform(newsgroup.data)
    counts = pd.DataFrame(train_.toarray(),
                          columns=vector.get_feature_names())
    quant_ = counts['phone'].sum()
    return int(quant_)

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[20]:


def q7():
    vector = TfidfVectorizer()
    train_ =  vector.fit_transform(newsgroup.data)
    counts = pd.DataFrame(train_.toarray(),
                          columns = vector.get_feature_names())
    quanti = round(counts['phone'].sum(),3)
    return float(quanti)

q7()

