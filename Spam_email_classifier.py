
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
df=pd.read_csv("mail_data.csv")

# %%
#df

# %%
data=df.where((pd.notnull(df)),'')

# %%


# %%


# %%
#data.head(5)

# %%


# %%
data.loc[data["Category"]=="spam","Category"]=0
data.loc[data["Category"]=="ham","Category"]=1


# %%
x=data["Message"]
y=data["Category"]

# %%
#x

# %%
#y

# %%


# %%
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=3)

# %%
#print(x.shape)
#print(X_train.shape)
#print(X_test.shape)

# %%
#print(y.shape)
#print(Y_train.shape)
#print(Y_test.shape)

# %%
feature_extraction=TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# %%
#X_train

# %%
#print(X_train_features)

# %%
model=LogisticRegression()

# %%
model.fit(X_train_features,Y_train)

# %%
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)

# %%
#print("Accuracy on training data:",accuracy_on_training_data)

# %%
prediction_on_testing_data=model.predict(X_test_features)
accuracy_on_testing_data=accuracy_score(Y_test,prediction_on_testing_data)

# %%
#print("Accuracy on testing data:",accuracy_on_testing_data)

# %%
input_content=input("Enter mail content:")
input_your_mail=[input_content]
input_data_features=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)
#print(prediction)

if (prediction[0]==1):
    print("Ham Mail")
else:
    print("Spam mail")