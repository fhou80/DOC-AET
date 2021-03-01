DOC-AET: Improving Named Entity Linking through Anonymous Entity Mentions
========

A Python implementation of ACL 2021 submission: \
Improving Named Entity Linking through Anonymous Entity Mentions.\
Our code is based on the MulRel model [Le and Titov 2018](https://github.com/lephong/mulrel-nel).




### Installation

- Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8 

### Usage

The following instruction is for replicating the experiments reported in our manuscript. 


#### Data

Download data from [GoogleDrive](https://drive.google.com/file/d/17yyrEp5_ngWXB5nR3f70AUs_NBUhSihN)
and unzip to the main folder (i.e. your-path/mulrel-nel).

#### Evaluate

Download our pre-trained model from [GoogleDrive](https://drive.google.com/file/d/1mPcwsKXbE1T_TxPWQ1Mkvshvp58wB_FJ)

    python -u -m nel.main --mode eval --model_path model

#### Train

To train a 3-relation ment-norm model, from the main folder run 

    export PYTHONPATH=$PYTHONPATH:../
    python -u -m nel.main --mode train --n_rels 3 --mulrel_type ment-norm --model_path model
 
Using a GTX 1080 Ti GPU it will take about 1 hour. The output is a model saved in two files: 
`model.config` and `model.state_dict` . 


