

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


dataset = pd.read_csv('F:/mini proj/Review-Analyzer/data/scrapped_data_rmp.csv')
X = dataset['Review']
Y = dataset['Useful']

dataset


stoplist = stopwords.words('english')


f_pre = []
for i in X:
  item = []
  #for j in i:
  #j.lower()
  word_tokens = word_tokenize(i)
  j = [w for w in word_tokens if not w in stoplist]
  item.append(j)
  f_pre.append(item)

"""**CountVectorizer**"""

cv = CountVectorizer(analyzer="word",
                     ngram_range=(1,1),
                     binary=True,
                     tokenizer = None,
                     preprocessor = None,
                     stop_words = 'english',
                     max_df = 0.99)

x = cv.fit_transform([str(i) for i in f_pre])

LE = LabelEncoder()
LE.fit(Y)
encoded_Y = LE.transform(Y)
y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(x,y,train_size=0.9)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_dim=4766,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

import matplotlib.pyplot as plt

history = model.fit(X_train,Y_train,epochs=6,batch_size=25,validation_split=0.2)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(X_test,Y_test)
print('Accuracy: %2f'%(accuracy*100),'%')


'''model.save('review.h5')

k = models.load_model('review.h5')

'''