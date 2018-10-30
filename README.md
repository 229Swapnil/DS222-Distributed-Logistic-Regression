# DS222-Distributed-Logistic-Regression

This repository contains a comparative study implementing Logistic Regression in single machine and a distributed environment. The results are summarized in the `Report.pdf` and the implementation details are as discusstion below.

# Running the codes
The above codes are developed in `python 2.7` and requires the following libraries:
1. numpy==1.15.1
2. tensorflow==1.10.1
3. scipy==1.15

Before running any code of the codes run: `python data_prep.py`

Optionally for ease of experimentation the preprocessed data is already present in the repository in stored in sparse format.

# In memory
This code can optionally take a input argument taking one of the three arguments: `constant`,`decay` and `increase` corresponding to the three strategies for varying learning rate.

Run the code: `python LogitR_memo.py -lr "decay"`

# Distributed Tensorflow
For all the three settings namely `bulk synchronous`, `stale-synchrnous` and `asynchronous` 2 parameter server nodes and 2 worker nodes are used. To run any of the above codes go to the respective nodes and run the following commands:

1. pc1: python LogitR_stsynchro.py --job_name="ps" --task_index=0
2. pc2: python LogitR_stsynchro.py --job_name="ps" --task_index=1
3. pc3: python LogitR_stsynchro.py --job_name="worker" --task_index=0
4. pc4: python LogitR_stsynchro.py --job_name="worker" --task_index=1

Inside the code the ip address of the above nodes need to be specified in the following section:

parameter_servers = ["10.24.1.218:2222","10.24.1.214:2223"]

workers = ["10.24.1.215:2224", "10.24.1.217:2225"]
