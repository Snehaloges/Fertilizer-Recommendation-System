#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("FRSC.csv")
df.head()


# In[3]:


df.drop('Unnamed: 9',axis=1,inplace=True)


# In[4]:


df.rename(columns ={'Organic Fertilizer':'Organic_Fertilizer'}, inplace = True)


# In[5]:


df.head(5)


# In[6]:


df.describe()


# In[7]:


num_df = df.select_dtypes(include=['float64','int64'])
corr =num_df.corr()
corr


# In[8]:


df.info()


# In[9]:


df["Soil Type"].unique()


# In[10]:


df["Crop Type"].unique()


# In[11]:


df["Temparature"].unique()


# In[12]:


df["Organic_Fertilizer"].unique()


# In[13]:


df["Fertilizer Name"].unique()


# In[14]:


df['Organic_Fertilizer'].value_counts()


# In[15]:


df['Soil Type'].value_counts()


# In[16]:


df['Crop Type'].value_counts()


# In[17]:


df.columns


# In[18]:


df.isnull().sum()


# In[19]:


df['Organic_Fertilizer'].unique()


# In[20]:


df.shape


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize =(16,8))
sns.countplot(x="Organic_Fertilizer",data =df)


# In[22]:


#Encoding Categorical data values
df.iloc[:,9].values


# In[24]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
df['Fertilizer Name'] = onehot_encoder.fit_transform(df[['Fertilizer Name']]).toarray()
df['Fertilizer Name'].values


# In[25]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder1= OneHotEncoder()
df['Soil Type'] = onehot_encoder.fit_transform(df[['Soil Type']]).toarray()
df['Soil Type'].values


# In[26]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder2 = OneHotEncoder()
df['Crop Type'] = onehot_encoder.fit_transform(df[['Crop Type']]).toarray()
df['Crop Type'].values


# In[27]:


df["Organic_Fertilizer"].dtype


# In[28]:


df.head()


# In[29]:


## Visualization
import seaborn as sns
sns.heatmap(corr, annot = True, cbar = True, cmap ='coolwarm')


# In[30]:


X = df.iloc[:,0:8].values
Y = df.iloc[:,9].values


# In[31]:


print(X)


# In[32]:


print(Y)


# In[33]:


#  column exact names
print(df.columns)



# In[34]:


df.head()


# ## Implementation of Training and Testing using Random Forest Algorithm


## Label Encoder to encode the organic fertilizer
from sklearn.preprocessing import LabelEncoder
fertilizer_list = ['compost and animal manure', 'poultry manure', 'Kelp and fish emulsion', 
                   'Green Manure', 'compost and greensand', 'vermicompost and banana peels', 
                   'Bone meal and wood Ash']
label_encoder = LabelEncoder()
label_encoder.fit(fertilizer_list)
df['Encoded_Organic_Fertilizer'] = label_encoder.transform(df['Organic_Fertilizer'])








# In[36]:

# Input Feature and Target Feature
X = df[['Temparature', 'Moisture', 'Humidity ','Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
Y = df['Encoded_Organic_Fertilizer']


# In[37]:


X.head()


# In[38]:


Y.head()


# In[39]:

#Splitting of dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2,random_state= 42,shuffle=True)


# In[40]:


X_train.shape


# In[41]:


from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)


# In[42]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[43]:


X_train


# In[44]:


#RandomClassifier Algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)


# In[45]:
#Prediction
pred = model.predict(X_test)


# In[46]:
pred


# In[47]:
#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,pred)


# In[48]:

#Evaluation Metrics
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,pred))


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,pred)
     )


# In[50]:


df.columns


# In[51]:


df.rename(columns ={'Soil Type':'Soil_Type'}, inplace = True)
df.rename(columns ={'Crop_Type':'Crop_type'}, inplace = True)


# In[53]:

Y.head()

# In[54]:



  # Function for Organic Fertilizer Prediction
def fertilizer_recommend(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorus):


    #  input for prediction
    input_data = np.array([[Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorus]])

    mx_features = mx.transform(input_data)
    sc_mx_features = scaler.transform(mx_features)

    # Predict using the model
    encoded_prediction = model.predict(sc_mx_features)
    fertilizer_name = label_encoder.inverse_transform(encoded_prediction)[0]
    

    print("Type of encoded_prediction:", type(encoded_prediction))
    print("Content of encoded_prediction:", fertilizer_name)

    return fertilizer_name




# In[55]:


Temparature=28	
Moisture=54		
Humidity =65	
Soil_Type=1.0	
Crop_Type=0.0
Nitrogen=39		
Potassium=0
Phosphorous=0	
predict = fertilizer_recommend(Temparature, Humidity , Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous)


# In[56]:

predict


#pip install joblib


# In[62]:


import joblib

# Save the  model
import joblib


joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(mx, 'minmaxscaler.pkl')
joblib.dump(scaler, 'standardscaler.pkl')
# Save the LabelEncoder for Organic Fertilizers
joblib.dump(label_encoder, 'fertilizer_label_encoder.pkl')
