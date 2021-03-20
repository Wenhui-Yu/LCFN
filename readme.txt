Codes for paper:
Wenhui Yu and Zheng Qin. 2020. Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters. In ICML.

Wenhui Yu, Zixin Zhang, Junfeng Ge, Wenwu Ou, and Zheng Qin. 2021. Less is More: Exploring Simple and Powerful Low-pass Graph Convolutional Network for Recommendation.

Wenhui Yu, Xiao Lin, Jinfei Liu, Junfeng Ge, Wenwu Ou, and Zheng Qin. 2021. Self-propagation Graph Neural Network for Recommendation. In TKDE.

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
    1.2 Run _graph_embeddings.py in folder pretraining.
    1.3 Run _main.py in folder pretraining (set EMB_DIM as 128, 64, 42, and 32 in p_params.py).
    We also provide downloading for pretraining:
        https://drive.google.com/file/d/12_109fqgUUUSQDeiwopqQ_fsQMCgTTVo/view?usp=sharing
        https://pan.baidu.com/s/1z-KZLF_soCUD2LwXB_0LVg (password: ss2h)
    You can choose one of these two URLs for downloading (we recommend the first one). Downloaded and unzip LCFN_dataset.zip and use it to replace the folder dataset in our project.

2. Run _main.py in our project (datasets, hyperparameters can be set in params.py).

3. Check results in folder experiment_result. Collect results by result_collection.

******************************************************************************

* In the result_collection folder, we provide a tool for results collection. Please read the manual for details.

* We also release our tuning results in folder supplementary_material.

* In the dataset folder, we prodive Amazon and Movielens to conduct our experiments. For each dataset, train_data, validation_data, and test_data are three datasets after preprocessing. You can use our processed datasets, or construct them from the raw data by running amazon.py and movielens.py. For the raw data please find on:

1. Amazon 
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

2. Movielens
http://grouplens.org/datasetss/movielens/1m

******************************************************************************

Please cite our paper if you use our codes:

@inproceedings{LCFN,
	title={Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters},
	author={Yu, Wenhui and Qin, Zheng},
	booktitle = {ICML},
	year={2020}
}

@article{SGNN,
	title={Self-propagation Graph Neural Network for Recommendation},
	author={Yu, Wenhui and Lin, Xiao and Liu, Jinfei and Ge, Junfeng and Ou, Wenwu},
	journal={TKDE},
	year={2021}
}
