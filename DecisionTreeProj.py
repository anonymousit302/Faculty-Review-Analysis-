import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree



df = pd.read_csv('F:/mini proj/Review-Analyzer/data/scrapped_data_rmp.csv')
#df.head()

#df.size

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.159, random_state=42)
# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
# Accuracy of our Model
print("Accuracy of Training Model",clf.score(X_train,y_train)*100,"%")
# Accuracy of our Model
print("Accuracy of Testing Model",clf.score(X_test,y_test)*100,"%")

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


tree.plot_tree(clf.fit(X_train,y_train)) 





'''
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = Xfeatures,class_names=y)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('feedback.png')
Image(graph.create_png())'''


'''# Sample1 Prediction
sample_name = ["Good"]
vect = cv.transform(sample_name).toarray()
vect

clf.predict(vect)

sample_name1 = ["Best Teacher"]
vect1 = cv.transform(sample_name1).toarray()
clf.predict(vect1)


test_name = []
def genderpredictor(*a):
    for x in a:
        test_name.append(a)
    vector = cv.transform(test_name).toarray()
    print(vector)
    if clf.predict(vector) == 1:
        print("Positive")
    else:
        print("Negative")

review = ["Bad Teacher","Worst Proff","Good Teacher","Hard Class"]
genderpredictor(review)'''

'''from sklearn.externals import joblib
NaiveBayesModel = open("naivebayesgendermodel.pkl","wb")
joblib.dump(clf,NaiveBayesModel)
NaiveBayesModel.close()


loaded_model = joblib.load("naivebayesgendermodel.pkl")
result = loaded_model.score(X_test, y_test)
print(result*100)'''



