Codes for paper:
Wenhui Yu and Zheng Qin. 2020. Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters. In ICML.

This project is for our model LCFN and baselines.

* Environment:
  Python 3.6.8 :: Anaconda, Inc.
* Libraries:
  tensorflow 1.12.0
  numpy 1.16.4
  scipy 0.18.1
  pandas 0.18.1
  openpyxl 2.3.2
  xlrd 1.0.0
  xlutils 2.0.0

Please follow the steps below:
1. Running _hypergraph_embeddings.py in folder pretraining.
2. Running _main.py in folder pretraining (set EMB_DIM as 64, 42, and 32 in p_params.py).
We also provide downloading for these two steps:

https://drive.google.com/file/d/1pAVE41umi__v_EKa5Q_V-Gc62TD3AGzn/view?usp=sharing
https://pan.baidu.com/s/191KISpCFN6HvpI2AyytGGA

You can choose one of these two URLs for downloading and we recommend the first one. Downloaded and unzip dataset.zip and use it to replace the folder dataset in our project.
3. Running _main.py in our project (datasets, hyperparameters can be set in params.py).
4. Check the result in folder experiment_result.