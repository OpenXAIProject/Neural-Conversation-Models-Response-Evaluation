# Neural Conversation Models

Open-sources of neural conversation models

## Environment
We've tested the code in this environment

- Python 3.7.3
- PyTorch 1.1.0

## Implemented conversation models
- HRED ([paper](https://arxiv.org/abs/1507.04808))
- Speaker-Addressee model ([paper](https://www.aclweb.org/anthology/papers/P/P16/P16-1094/))

## How to run the code
Please see the slides in XAI Workshop in slides folder.

## Reference
- https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling
- https://github.com/jiweil/Neural-Dialogue-Generation


# Speaker Sensitive Response Evaluation Model

## Overview
This repo provides the implementation of Speaker Sensitive Response Evaluation Model (SSREM).

## Tested Environmnet
- Python 3.6.3
- Pytorch 1.3

## How to run
In 'src' folder, we make bash script file to train and evaluate SSREM.
All arguments for the bash files are passed into argparse in `configs.py`/

 - `Run_train.sh`: a bash script file to train SSREM

``
bash Run_train.sh 0 TC SSREM 5000 2000 1e-4
``

- `Run_eval1.sh`: a bash script file to identify the true and false responses

``
bash Run_eval1.sh 0 TC SSREM 5000 ../results/TC/SSREM/20191209_235959/2000.pkl 4
``



## Data
We use Twitter conversation corpus: https://www.aclweb.org/anthology/D19-1202/
Please contact to the authors of the paper to get the corpus.

## Discussion
Please let us know if you have any question or requests by issues or email. (First author of the paper page: https://nosy.github.io/)

## Reference
- https://github.com/NoSyu/VHUCM
- https://github.com/NoSyu/SSREM

<br /> 
<br />

# XAI Project 

**This work was supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence (의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : KAIST, Korea Univ., Yonsei Univ., UNIST, AITRICS  

+ Web Site : <http://openXai.org>
