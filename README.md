# TEAM DEEPLEARNER

Our Model have Three different codes file which are run in respective manner to get a perfect **solution.csv**.






## Compiling and Executing:

1. Run requirements.txt

    ```bash
    pip install -r requirements.txt
    ```


2. First prepare_data.py:
    
    Open prepare_data.py Give the appropriate path of the **train.csv**
   
   ```bash
   python prepare_data.py
   ```

3. Second train.py:
   
   Give the path of **train.csv and test.csv** data.
   
   ```bash
   python train.py
   ```

4. Third predict.py:
    
    Give the path of **test.csv** data.
   
   ```bash
   python predict.py
   ```
   
#### Output Will be generated i.e *solution.csv*. 


#### For New Data

---> Save the file same as test.csv dataframe(Most have same columm as test) and correct the  the path in predict.py. This will again produce the outout of same name i.e *solution.csv*.



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


### Notes:
1. No specific third party app are required to run our code.
2. No any special assumptions are made by our code.
3. Our code doesn't have any specific side effects as they are run in google colab.
4.We haven't saved any cleaned data or dataframes. We just used the temporary dataframes with cleaned data to train and test. So we don't have directories such as CLEAN_DATA_DIR, TRAIN_DATA_CLEAN_PATH and TEST_DATA_CLEAN_PATH.


NOTE: Google colab is best way to run our code.
