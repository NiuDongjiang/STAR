# Interaction-Aware Adaptive Network for Drug-Drug Interaction Prediction

# Abstract

Accurate prediction of protein-ligand binding affinity is critical for structure-based drug discovery but remains challenging due to the complex interplay of chemical and geometric information in biomolecular systems. Existing multi-modal and two-dimensional graph-based approaches often struggle with feature redundancy, insufficient modeling of three-dimensional structures, and limited capacity to capture fine-grained interface interactions. We propose STAR, a structure-based interface-aware network. STAR uses a ligand-centric graph representation and a dual geometric-sensitive adaptive gating mechanism to balance chemical and geometric information during message passing. An interface interaction attention module explicitly models the physical interactions at the binding interface. Comprehensive experiments conducted on multiple benchmark datasets demonstrate the superior predictive performance and generalization capability of STAR. Visualization of RSK2 kinase inhibitors illustrates the internal decision process of the model, confirming its utility for practical affinity prediction tasks.


# 1. Requirements

To reproduce **STAR**, you can create your environment by env.yaml:
```sh
    $ conda env create -f env.yaml
```

# 2. Usage

### 2.1. Data

Data for STAR can be unzipped from data.zip, and the raw data can be downloaded from [here](https://www.pdbbind-plus.org.cn/), processed by graph_constructer.py.

### 2.2. Weight 
Weights for STAR can be downloaded from [here](https://pan.baidu.com/s/1-PDKToc8Lf9xXTRgZGeDFQ?pwd=0000).

### 2.3. Useage 
For training:
```sh
    $ python train_s1.py or train_s2.py
```
# 3. Concat
Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at niudongjiang@qdu.edu.cn.
