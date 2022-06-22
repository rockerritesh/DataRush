import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Reading the train, test and validation data

df_train = pd.read_csv('RAW_DATA_DIR/train.csv')
df_test = pd.read_csv('RAW_DATA_DIR/test.csv')
df_val = pd.read_csv('RAW_DATA_DIR/validation.csv')

# defining x_train, y_train, x_test,x_val and y_val from cleaned dataframe

X_train, y_train = df_train['abstract'],df_train['category']
X_val, y_val = df_val['abstract'],df_val['category']
X_test= df_test['abstract']



# loading tfid model

print('loading tfid model')

with open('MODEL_DIR/tfid7046super6bestinput2000000.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    

# transforming according to above trained loaded tfid_vectorizer

print('Transferming train and val data to vector for further processing')

X_train = tfidf_vectorizer.transform(X_train)
X_val = tfidf_vectorizer.transform(X_val)
X_test= tfidf_vectorizer.transform(X_test)    

print('loading y label model')

with open('MODEL_DIR/labelencoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print('Transferming y label to category number representation')

y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)


print('Training the model')

# definig svc model 


clf=LinearSVC(random_state=0)
clf.fit(X_train, y_train)    # training model on train data


clfval = clf.predict(X_val)   # predicting val data
print('F1 Score : {}'.format(f1_score(y_val, clfval, average='micro')))  # printing F1 score 

print('model is trained')

# saving model

print('saving model for future')



with open('MODEL_DIR/saved7046super6best.pkl', 'wb') as file:
    pickle.dump(clf, file)
    
    
print('model saved')    