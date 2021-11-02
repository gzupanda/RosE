# The model RosE

This repository contains the code of the main experiments presented in the paper:RosE: Towards the Expressiveness of Translation-based Model for Knowledge Graph Completion. 

In this paper, we introduce freedoms to improve the expressiveness of the translation-based model by rotating and scaling the translation equation (i.e., the approximate equation $\mathbf{h} + \mathbf{r}\approx \mathbf{t}$). Further research shows that the manner of the tranformations of it strongly related with the expressiveness of the model.

## Install 

Firstly, the code of the model $\textit{RosE}$ can be cloned from the repository in github:
```
git clone https://github.com/gzupanda/RosE.git
```

Secondly, the code depends on [downhill](https://github.com/lmjohns3/downhill), a theano-based Stochastic Gradient Descent implementation. The proposed model and baseline models are implemented in Python and the deep learning framework [Theano](http://www.uml.org.cn/ai/202104132.asp) on the basis of the model [ComplEx](https://github.com/ttrouill/complex). 

To run the code, the dependencies should also be installed:
```
pip install -r requirements.txt
```
Furthermore, the code is compatible with both Python 2 and 3.
## Run the experimental code

We provide two versions of the proposed model, they are RosE$_{1l}$ and  RosE$_{2l}$, respectively.
### 1. on datasets Kinships and UMLS
These tow datasets are the small datasets and used for evaluating the model in the early age. For the dataset Kinships:
```
python kinships_1L.py
python kinships_2L.py
```
And for the dataset UMLS:
```
python umls_1L.py
python umls_2L.py
```
### 2. on datasets FB15k and WN18
These two datasets can be considered as standard ones for new embedding-based knowledge graph completion model evaluting. For the dataset Freebase (FB15k):
```
python fb15k_1L.py
python fb15k_2L.py
```
And for the dataset Wordnet (WN18):
```
python wn18_1L.py
python wn18_2L.py
```
In the following table, the results can be contributed into two groups. The first group includes the models from TransE to GTrans-GD, they are the models transformed from the tranlation equation linearly. The second group contain the model RotatE, RatE and TimE, which are newly proposed models transformed from the translation equation non-linearly.
|Model|Stra.|FB15k|FB15k|FB15k|FB15k|WN18|WN18|WN18|WN18|
|:-----|:--------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|TransE |unif.|125|243|47.1|34.9|251|263|89.2|75.4|
|TransH |unif.|84|211|58.5|42.5|303|318|86.7|75.4|
|TransH |bern.|87|212|64.4|45.7|388|401|82.3|73.0|
|TransR |unif.|78|226|65.5|43.8|219|232|91.7|78.3|
|TransR |bern.|77|198|68.7|48.2|225|238|92.0|79.8|
|CTransR |unif.|82|233|66.3|44.0|230|243|92.3|78.9|
|CTransR  |bern.|75|199|70.2|48.4|218|231|92.3|79.4|
|TransD |unif.|67|211|74.2|49.4|229|242|92.5|79.2|
|TransD |bern.|91|194|77.3|53.4|212|224|92.2|79.6|
|TransPES|unif.|66|198|67.3|48.1|212|223|81.3|71.6|
|TransE-RS |unif.|62|$\mathbf{161}$|72.3|53.1|348|362|93.7|80.3|
|TransE-RS |bern.|63|$\mathbf{161}$|72.1|53.2|371|385|93.7|80.4|
|TransH-RS |unif.|64|$\underline{163}$|72.6|53.4|389|401|94.7|81.2|
|TransH-RS |bern.|77|178|75.0|53.6|357|371|94.5|80.3|
|TransGH |unif.|66|186|79.8|$\underline{54.0}$|$\underline{179}$|$\underline{191}$|94.8|$\underline{81.4}$|
|TransGH |bern.|64|186|80.1|$\mathbf{54.1}$|197|210|$\underline{95.3}$||\mathbf{81.6}||
|GTrans-DW |unif.|147|256|63.4|44.1|197|210|92.2|78.4|
|GTrans-DW |bern.|126|235|60.5|43.1|$\mathbf{166}$|$\mathbf{180}$|90.3|77.1|
|GTrans-SD |unif.|66|207|75.1|50.6|234|247|92.9|79.1|
|GTrans-SD |bern.|85|189|75.3|52.9|202|215|93.5|80.2|
|RotatE|unif.|$\underline{40}$|-|74.6|-|309|-|95.9|-|
|RatE |unif.|$\mathbf{24}$|-|$\mathbf{89.8}$|-|180|-|$\mathbf{96.2}$|-|
|TimE |unif.|45.9|-|$\underline{87.9}$|-|259|-|$\underline{96.1}$|-|
|RosE$_{1l}$|unif.|79|233|81.5|50.4|271|284|95.0|$\underline{81.4}$|
|RosE$_{2l}$|unif.|79|234|80.4|48.7|347|362|94.5|78.8|
### on the datasets FB15k237 and WN18RR
These two datasets consider to be challenging since they are proposed for test set leakage. For the dataset Freebase (FB15k237):
```
python fb15k237_1L.py
python fb15k237_2L.py
```
And for the dataset Wordnet (WN18RR):
```
python wn18rr_1L.py
python wn18rr_2L.py
```
To run on GPU (approx 5x faster at least), simply add the following theano flag before the python call:
```
THEANO_FLAGS='device=gpu' python fb15k_1L.py
```
## Run on your own data

Create a subfolder in the `datasets` folder, and put your data in three files `train.txt`, `valid.txt` and `test.txt`. Each line is a triple, in the format: 
```
subject_entity_id	relation_id	object_entity_id
```
separated with tabs. Then modify `fb15k_1L.py` (`fb15k_2L.py`) for example, by changing the `name` argument in the `build_data` function call to your data set folder name:
```
fb15kexp = build_data(name = 'your_dataset_folder_name',path = tools.cur_path + '/datasets/')
```
## License

This software comes under a non-commercial use license, please see the LICENSE file.
