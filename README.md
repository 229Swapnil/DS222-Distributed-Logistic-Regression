# DS222-Distributed-Logistic-Regression

This repository contains a comparative study implementing Logistic Regression in single machine and a distributed environment. The results are summarized in the `Report.pdf` and the implementation details are as discusstion below.

# Running the codes
The above codes are developed using `python 2.7` and requires the following libraries:
1. numpy==1.15.1
2. tensorflow==1.10.1
3. scipy==1.15

Before running the any code run: `python data_prep.py`

# In memory
This code can optionally take a input argument taking one of the three arguments: `constant`,`decay` and `increase` corresponding to the three strategies for varying learning rate.

Run the code: `python LogitR_memo.py -lr "decay"`
