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
