# loading model

import pickle
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np

# loading model

print('loading model')

with open('MODEL_DIR/saved7046super6best.pkl', 'rb') as f:
    clf = pickle.load(f)
    
print('Model loaded')


df_test = pd.read_csv('RAW_DATA_DIR/test.csv')
X_test= df_test['abstract']

print('test data loaded')



# loading tfid model

print('loading tfid model')

with open('MODEL_DIR/tfid7046super6bestinput2000000.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    

# transforming according to above trained loaded tfid_vectorizer

print('Transferming X_test  to vector for further processing')

X_test= tfidf_vectorizer.transform(X_test) 

print('x_test is transferd to vector')

print('now its time to predict')


#predicting X_test data

testclf=clf.predict(X_test)

print('prediction completed')

print('Generating solution.csv')


# convert array into dataframe
DF = pd.DataFrame(testclf,columns=['category_num'])
id=np.array(df_test['id'])
DF.insert(0, 'id', id)  
# save the dataframe as a csv file
   
DF.to_csv(r'SUBMISSION_DIR\solution.csv', index = False)


print('output is generated')

print('THANKS!!')