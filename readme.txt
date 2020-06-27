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
1. Pretrain
    1.1 Run _hypergraph_embeddings.py in folder pretraining.
    1.2 Run _main.py in folder pretraining (set EMB_DIM as 64, 42, and 32 in p_params.py).
    We also provide downloading for pretraining:
        https://drive.google.com/file/d/12_109fqgUUUSQDeiwopqQ_fsQMCgTTVo/view?usp=sharing
        https://pan.baidu.com/s/1z-KZLF_soCUD2LwXB_0LVg (password: ss2h)
    You can choose one of these two URLs for downloading (we recommend the first one). Downloaded and unzip dataset.zip and use it to replace the folder dataset in our project.

2. Run _main.py in our project (datasets, hyperparameters can be set in params.py).

3. Check results in folder experiment_result. Collect results by result_collection.

******************************************************************************

* In the result_collection folder, we provide a tool for results collection. Please read the manual for details.

* In the supplementary_material folder, we provide the dropout version, the toy example, and our tuning results.

1. Dropout version: Experiments show that fine tuned with respect to the regularization coefficient, models fail to achieve further improvement with dropout, and many models even perform worse. Here we also provide the "dropout version" for more choices, however, we do not recommend this version.

2. Toy example: GFTandFFT.py is for the toy example shown in Figures 2 and 4 in the paper.

3. Our tuning results: For fair comparison, we conduct very comprehensive tuning strategy and we also release our tuning result for readers.

* In the dataset folder, we prodive Amazon and Movielens to conduct our experiments. For each dataset, train_data, validation_data, and test_data are three datasets after preprocessing. You can use our processed datasets, or construct them from the raw data by running amazon.py and movielens.py. For the raw data please find on:

1. Amazon 
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

2. Movielens
http://grouplens.org/datasetss/movielens/1m

******************************************************************************

Please cite our paper if you use our codes£º

@inproceedings{LCFN,
	title={Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters},
	author={Yu, Wenhui and Qin, Zheng},
	booktitle = {ICML},
	year={2020},
}
