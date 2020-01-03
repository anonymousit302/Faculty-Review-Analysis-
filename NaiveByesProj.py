import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('F:/mini proj/Review-Analyzer/data/scrapped_data_rmp.csv')
df.head()

df.size

# Data Cleaning
# Checking for column name consistency
df.columns

df[df.Useful == 1].size

df_names = df
Xfeatures =df_names['Review']
# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
cv.get_feature_names()
from sklearn.model_selection import train_test_split
# Features 
X
# Labels
y = df_names.Useful
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# Accuracy of our Model
print("Accuracy of Training Model",clf.score(X_train,y_train)*100,"%")
# Accuracy of our Model
print("Accuracy of Testing Model",clf.score(X_test,y_test)*100,"%")

# Sample1 Prediction
sample_name = ["Good"]
vect = cv.transform(sample_name).toarray()
vect

clf.predict(vect)

sample_name1 = ["Best Teacher"]
vect1 = cv.transform(sample_name1).toarray()
clf.predict(vect1)

'''vector = cv.transform(["Best Teacher"]).toarray()
print(vector)
if clf.predict(vector) == 1:
    print("Positive")'''
# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    print(vector)
    if clf.predict(vector) == 1:
        print("Positive")
    else:
        print("Negative")
        
genderpredictor("Bad Teacher")

from sklearn.externals import joblib
NaiveBayesModel = open("naivebayesgendermodel.pkl","wb")
joblib.dump(clf,NaiveBayesModel)
NaiveBayesModel.close()


loaded_model = joblib.load("naivebayesgendermodel.pkl")
result = loaded_model.score(X_test, y_test)
print(result*100)