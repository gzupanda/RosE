# wTransE: A Weighted Translation-based Model forKnowledge Graph Completion

This repository contains the code of the main experiments presented in the papers:

## Install 

First clone the repository:
```
git clone https://github.com/gzupanda/wTransE.git
```

The code depends on [downhill](https://github.com/lmjohns3/downhill), a theano-based Stochastic Gradient Descent implementation.

Install it, along with other dependencies with:
```
pip install -r requirements.txt
```

The code is compatible with Python 2 and 3.

## Run the experiments

We provide two versions of them, they are $w_1$TransE and  $w_w$TransE, respectively.
### The first group of datasets.
For the dataset Kinships:
```
python kinships_1L.py
python kinships_2L.py
```
And for the dataset UMLS:
```
python umls_1L.py
python umls_2L.py
```
### The second group of datasets.
For the dataset Freebase (FB15k):
```
python fb15k_1L.py
python fb15k_2L.py
```
And for the dataset Wordnet (WN18):
```
python wn18_1L.py
python wn18_2L.py
```
### The third group of datasets.
For the dataset Freebase (FB15k237):
```
python fb15k237_1L.py
python fb15k237_2L.py
```
And for the dataset Wordnet (WN18RR):
```
python wn18rr_1L.py
python wn18rr_2L.py
```

By default, it runs the ComplEx (Complex Embeddings) model, edit the files and uncomment the corresponding lines to run DistMult, TransE, RESCAL or CP models. The given hyper-parameters for each model are the best validated ones by the grid-search described in the paper.

To run on GPU (approx 5x faster), simply add the following theano flag before the python call:
```
THEANO_FLAGS='device=gpu' python fb15k_1L.py
```

## Run on your own data

Create a subfolder in the `datasets` folder, and put your data in three files `train.txt`, `valid.txt` and `test.txt`. Each line is a triple, in the format: 
```
subject_entity_id	relation_id	object_entity_id
```
separated with tabs. Then modify `fb15k_run.py` for example, by changing the `name` argument in the `build_data` function call to your data set folder name:
```
fb15kexp = build_data(name = 'your_dataset_folder_name',path = tools.cur_path + '/datasets/')
```


## Implement your own model


Models are defined as classes in `models.py`, that all inherit the class `Abstract_Model` defined in the same file. The `Abstract_Model` class handles all the common stuff (training functions, ...), and child classes (the actual models) just need to define their embeddings shape and initialization, and their scoring and loss function.

To properly understand the following, one must be comfortable with [Theano basics](http://deeplearning.net/software/theano/library/tensor/basic.html).

The `Abstract_Model` class contains the symbolic 1D tensor variables `self.rows`, `self.cols`, `self.tubes` and `self.ys` that will instantiate at runtime the corresponding: subject entity indexes, relation indexes, object entity indexes and truth values (1 or -1) respectively, of the triples of the current batch. It also contains the number of subject entities, relations and object entities of the dataset in `self.n`, `self.m`, `self.l` respectively, as well as the current embedding size in `self.k`.

Two functions must be overridden in the child classes to define a proper model: `get_init_params(self)` and `define_loss(self)`.

Let's have a look at the `DistMult_Model` class and its `get_init_params(self)` function:
```
def get_init_params(self):
	params = { 'e' : randn(max(self.n,self.l),self.k),
			   'r' : randn(self.m,self.k)}
	return params
```
This function both defines the embedding-matrix shapes (number of entities * rank for `e`, number of relations * rank for `r`), and their initial value (`randn` is `numpy.random.randn`), by returning a dictionnary where the key names correspond to the class attribute names. From this dict the mother class will create shared tensor variables initialized with the given values, and assigned to the corresponding attribute names (`self.e` and `self.r`).

Now the `define_loss(self)` function must define three Theano expressions: the scoring function, the loss, and the regularization.
Here is the `DistMult_Model` one:
```
def define_loss(self):
	self.pred_func = TT.sum(self.e[self.rows,:] * self.r[self.cols,:] * self.e[self.tubes,:], 1)

	self.loss = TT.sqr(self.ys - self.pred_func).mean()

	self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
					+ TT.sqr(self.r[self.cols,:]).mean() \
					+ TT.sqr(self.e[self.tubes,:]).mean()
```
The corresponding expressions must be written in their batched form, i.e. to compute the scores of multiple triples at once. For a given batch, the corresponding embeddings are retrieved with `self.e[self.rows,:]`, `self.r[self.cols,:]` and `self.e[self.tubes,:]`.

In the case of the DistMult model, the trilinear product between these embeddings is computed, here by doing first two element-wise multiplications and then a sum over the columns in the `self.pred_func` expression. The `self.pred_func` expression must yield a vector of the size of the batch (the size of `self.rows`, `self.cols`, ...).
The loss defined in `self.loss` is the squared-loss here (see the `DistMult_Logistic_Model` class for the logistic loss), and is averaged over the batch, as the `self.loss` expression must yield a scalar value.
The regularization defined here is the L2 regularization over the corresponding embeddings of the batch, and must also yield a scalar value.

That's all you need to implement your own tensor factorization model! All gradient computation is handled by Theano auto-differentiation, and all the training functions by the [downhill](https://github.com/lmjohns3/downhill) module and the `Abstract_Model` class.

## License

This software comes under a non-commercial use license, please see the LICENSE file.
