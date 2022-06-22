import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.corpus import stopwords
import pickle



# Reading the train, test and validation data

df_train = pd.read_csv('RAW_DATA_DIR/train.csv')
#df_test = pd.read_csv('RAW_DATA_DIR/test.csv')
#df_val = pd.read_csv('RAW_DATA_DIR/validation.csv')

# defining x_train, y_train, x_test,x_val and y_val from cleaned dataframe

X_train, y_train = df_train['abstract'],df_train['category']
#X_val, y_val = df_val['abstract'],df_val['category']
#X_test= df_test['abstract']

print('DATA FRAME TO TRAIN TEST AND VAL DATA IS LOADED') 

# making number representation of y_train and y_val data,
# here we make y_train and y_val as same as y_train so that we can make good prediction 

label_encoder = LabelEncoder().fit(y_train)

#y_train = label_encoder.transform(y_train)
#y_val = label_encoder.transform(y_val)

print('DATA CATEGORY IS ENCODED') 

# powerful TFID which have more stop words of different language and with different tunned parameters

stopwords_list = stopwords.words('english') + stopwords.words('french') + stopwords.words('german')
tfidf = TfidfVectorizer(max_features=2000000, sublinear_tf=True, ngram_range=(1, 2), 
                        stop_words=stopwords_list)
tfidf_vectorizer = tfidf.fit(X_train)





print('SAVING TFID MODEL FOR FUTURE USE!!')

# saving tfid model for further use




with open('MODEL_DIR/tfid7046super6bestinput2000000.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
    
    
print('SAVED TFID MODEL!!') 
    
# saving y_label model for further use


print('SAVING LABELING MODEL FOR FUTURE USE!!')


with open('MODEL_DIR/labelencoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)    

print('SAVED LABEL ENCODING MODEL!!')     
    
print('DATA PREPARED!! RUN train.py')   