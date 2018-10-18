### System requirements
The code were tesed on Linux and Mac OS systems.
The required software/packages are:
* python=3.6.5
* numpy=1.14.0 
* scipy=1.0.0 
* h5py=2.7.1 
* tensorflow=1.8.0 
* keras=2.1.6 
* scikit-learn=0.19.1 
* biopython=1.71 
* viennarna=2.4.5
* matplotlib
* DotMap
* GPyOpt
* pandas

It is worth noting that when the computing environment(e.g, the version of tensorflow or biopython) changes, the prediction results might change slightly, but the main conclusion won't be affected.

### Installation Guide
```bash
conda create -n crispr python=3.6.5 ipykernel matplotlib pandas numpy=1.14.0 scipy=1.0.0 h5py=2.7.1 tensorflow=1.8.0 keras=2.1.6 scikit-learn=0.19.1 biopython=1.71 viennarna=2.4.5
pip install GPyOpt
pip install DotMap
ipython kernel install --user --name crispr --display-name "Python3(crispr)"
```
Installation time depends on your own network environment.

### Demo
Demos were included in the  [Demo.ipynb](https://github.com/izhangcd/DeepHF/blob/master/Demo.ipynb) file. It contains prediction, metrics and model training demos.

### Files description
* [feature_util.py](https://github.com/izhangcd/DeepHF/blob/master/feature_util.py) contains the code for extracting position related features and biological features.

* [prediction_util.py](https://github.com/izhangcd/DeepHF/blob/master/prediction_util.py) contains the core code of prediction module for the website [www.DeepHF.com](http://www.deephf.com).

* [training_util.py](https://github.com/izhangcd/DeepHF/blob/master/training_util.py) provides the code for training model in your own computing environment. The optimized hyperparameters is only fit for the aformentioned software/package environment.

* [data/esp_seq_data_array.pkl](https://github.com/izhangcd/DeepHF/blob/master/data/esp_seq_data_array.pkl), features and experimental edit efficiency data for eSpCas9(1.1). It can be used to train the model.

* [data/hf_seq_data_array.pkl](https://github.com/izhangcd/DeepHF/blob/master/data/hf_seq_data_array.pkl) , features and experimental edit efficiency data for Cas9-HF1. It can be used to train the model.

* [models/esp_rnn_model.hd5](https://github.com/izhangcd/DeepHF/blob/master/models/esp_rnn_model.hd5),the final model file of eSpCas9(1.1) used in the DeepHF wibsite.

* [models/hf_rnn_model.hd5](https://github.com/izhangcd/DeepHF/blob/master/models/hf_rnn_model.hd5),the final model file of Cas9-HF1 used in the DeepHF wibsite.
