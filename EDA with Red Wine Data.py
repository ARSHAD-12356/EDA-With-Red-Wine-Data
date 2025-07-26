# %% [markdown]
# # EDA with Red Wine Data Project

# %% [markdown]
# # Dataset Information
# Additional Information
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks.  The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

# %% [markdown]
# # For more information, read [Cortez et al., 2009].
# Input variables (based on physicochemical tests):
#    1 - fixed acidity
#    
#    2 - volatile acidity
#    
#    3 - citric acid
#    
#    4 - residual sugar
#    
#    5 - chlorides
#    
#    6 - free sulfur dioxide
#    
#    7 - total sulfur dioxide
#    
#    8 - density
#    
#    9 - pH
#    
#    10 - sulphates
#    
#    11 - alcohol
#    
# Output variable (based on sensory data): 
# 
#    12 - quality (score between 0 and 10)

# %% [markdown]
# # i > Importing Library

# %%
import pandas as pd

# %% [markdown]
# # ii > Read the CSV File

# %%
df = pd.read_csv("winequality-red.csv")

# %%
df.head()

# %% [markdown]
# # Summary of the Dataset

# %%
df.info()

# %% [markdown]
# # Descriptive Summary of the Dataset

# %%
df.describe()

# %% [markdown]
# # Shape of the Dataset

# %%
df.shape

# %% [markdown]
# ## List down all the columns

# %%
df.columns

# %%
df['quality'].unique()

# %% [markdown]
# # Insights 1 : count the quality of each variant

# %%
## Conclusion --> Imbalanced Dataset
df['quality'].value_counts()

# %% [markdown]
# # Data Cleaning / Preprocessing

# %% [markdown]
# # Check the Missing Values

# %%
df.isnull().sum()

# %% [markdown]
# ## To Check Duplicate Records

# %%
df.duplicated()

# %%
df[df.duplicated()]

# %% [markdown]
# ## Remove the Duplicate Records

# %%
df.drop_duplicates(inplace=True)

# %%
df.shape

# %% [markdown]
# # Retrive Correlation of DataSet

# %%
df.corr()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.figure(figsize = (10,6))
sns.heatmap(df.corr() , annot = True)

# %%
df.quality.value_counts().plot(kind = 'bar')

# %%
sns.histplot(df['fixed acidity'])

# %%
for i in df.columns:
    sns.histplot(df[i], kde = True)

# %% [markdown]
# # Categorical Plot

# %%
sns.catplot(x = "quality", y = "alcohol" , data = df , kind = "box")

# %%
sns.scatterplot(x = 'alcohol' , y = 'pH' , hue = 'quality' , data = df)

# %%



