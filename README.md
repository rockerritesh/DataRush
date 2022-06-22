### Notes:
1. No specific third party app are required to run our code.
2. No any special assumptions are made by our code.
3. Our code doesn't have any specific side effects as they are run in google colab.
4.We haven't saved any cleaned data or dataframes. We just used the temporary dataframes with cleaned data to train and test. So we don't have directories such as CLEAN_DATA_DIR, TRAIN_DATA_CLEAN_PATH and TEST_DATA_CLEAN_PATH.

The directory tree is as follows:

### Root

---prepare_data.py

---train.py

---predict.py

---RAW_DATA_DIR

    ---train.csv

    ---validation.csv

    ---labels.csv

    ---test.csv

---MODEL_DIR

    ---labelencoder.pkl

    ---saved7046super6best.pkl
    
    ---tfid7046super6bestinput2000000.pkl

---SUBMISSION_DIR

    ---solution.csv


NOTE: Google colab is best way to run our code.