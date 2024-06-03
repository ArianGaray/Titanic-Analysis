#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1> ANÁLISIS DE DATOS : TITANIC
# </center>

# **<h2>1. Librerías a usar**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# **<h2>2. Importación de la base de datos a analizar**

# In[70]:


archivo = r"C:\Users\arian\OneDrive\Escritorio\Python\Proyectos\Titanic\train.csv"
titanic = pd.read_csv(archivo,index_col ="PassengerId")
titanic.head(20)


# La base de datos se extrajo de la siguiente url: https://www.kaggle.com/competitions/titanic/data?select=train.csv

# **<h2>3. Exploración de datos**
# 
# Vamos a explorar los datos para entender mejor su estructura:

# In[20]:


titanic.info()


# In[24]:


titanic.describe().round(2)


# In[25]:


titanic.isnull().sum()


# Aquí se observa que en la columna **Cabin** se encuenta el mayor número de variables nulas que tiene la tabla, seguido de **Age**

# **<h2>4. Limpieza de datos**
# 
# Vamos a limpiar los datos:

# In[75]:


# Se reemplaza los valores NA de la columna AGE por la mediana 
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)


# In[76]:


# Se reemplaza los valores nuloes en la columna EMBARKED con el modo
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)


# In[77]:


# Aquí se concenta la mayor cantidad de valores nulos, en la columna CABIN,entonces se tomó la decisión de colocar Unknown.
titanic['Cabin'].fillna('Unknown', inplace=True)


# In[78]:


# Se verifica si la tabla tiene algún valor nulo
titanic.isnull().sum()


# In[80]:


titanic.head(30)


# **<h2>5. Análisis descriptivo**
# 

# Vamos usar la libería **SEABORN** para poder observar la cantidad de supervivientes que tuvo el Titanic:

# In[62]:


import seaborn as sns

cantidad = titanic['Survived'].value_counts().reset_index()
cantidad.columns = ['Survived', 'Count']
print(survived_counts)

sns.countplot(data=titanic, x='Survived')
plt.title('Distribución de Supervivientes')
plt.show()



#El 0 hace referencia a las personas muertas y el 1 a las vivas


# Ahora vamos a analizar dentro de las personas que han sobrevivido y no, la clase en la que ellos estaban.

# In[63]:


# Supervivencia por clase
clase = titanic.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
clase ['Pclass'] = clase ['Pclass'].map({1: 'Primera Clase', 2: 'Segunda Clase', 3: 'Tercera Clase'})
print(clase)

sns.countplot(data=titanic, x='Survived', hue='Pclass')
plt.title('Supervivencia por Clase')
plt.show()


# Aquí se analiza los sobrevivientes y a los fallecidos según su género:

# In[64]:


# Supervivencia por género
genero = titanic.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
genero['Sex'] = genero['Sex'].map({'male': 'Masculino', 'female': 'Femenino'})
print(genero)

sns.countplot(data=titanic, x='Survived', hue='Sex')
plt.title('Supervivencia por Género')
plt.show()


# En esta gráfica se observa las edades de las personas que estuvieron en el Titanic.

# In[65]:


sns.histplot(titanic['Age'], kde=True)
plt.title('Distribución de Edades')
plt.show()


# Ahora, si queremos ser más curiosos podemos analizar la tendendía de edad según los sobrevivientes y fallecidos que tuvo el Titanic

# In[66]:


plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='Age', hue='Survived', kde=True, bins=20, multiple='stack')
plt.title('Distribución de Edades por Sobrevivientes y Fallecidos')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.legend(labels=['Fallecidos', 'Sobrevivientes'])
plt.show()


# Y finalmente, si queremos hacer un análisis un poco más complejo pero a la vez útil, podemos realizar un gráfico de violín que me indica a las personas que han sobrevivido según su edad y su clase:

# In[68]:


plt.figure(figsize=(10, 6))
sns.violinplot(data=titanic, x='Pclass', y='Age', hue='Survived', split=True)
plt.title('Supervivencia por Edad y Clase')
plt.show()

