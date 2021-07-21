#import scipy.io

import efe
from efe.exp_generators import *
import efe.tools as tools

if __name__ =="__main__":

	#Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
	fb15kexp = build_data(name = 'UMLS',path = tools.cur_path + '/datasets/')


	#SGD hyper-parameters:
	params = Parameters(learning_rate = 0.2, 
						max_iter = 1000, 
						batch_size = int(len(fb15kexp.train.values) / 20),  #Make 100 batches
						neg_ratio = 10, 
						valid_scores_every = 1000,
						learning_rate_policy = 'adagrad',
						contiguous_sampling = False,
						)

	#Here each model is identified by its name, i.e. the string of its class name in models.py
	#Parameters given here are the best ones for each model, validated from the grid-search described in the paper
	#all_params = { "Complex_Logistic_Model" : params } ; emb_size = 2; lmbda =0.01
	#all_params = { "DistMult_Logistic_Model" : params } ; emb_size = 200; lmbda =0.01
	#all_params = { "CP_Logistic_Model" : params } ; emb_size = 150; lmbda =0.03
	#all_params = { "Rescal_Logistic_Model" : params } ; emb_size = 150; lmbda =0.3
	all_params = { "wTransE_1L_Model" : params } ; emb_size = 200; lmbda = 1 ;params.miuA = 0.5;params.miuB = 0.5;params.lambda_A =1;params.lambda_B=1



	tools.logger.info("Max iter: " + str(params.max_iter))
	tools.logger.info("Generated negatives ratio: " + str(params.neg_ratio))
	tools.logger.info("Batch size: " + str(params.batch_size))
	tools.logger.info( "Learning rate: " + str(params.learning_rate))
	tools.logger.info("emb_size: " + str(emb_size))
	tools.logger.info("miuA: " + str(params.miuA))
	tools.logger.info("miuB: " + str(params.miuB))
	tools.logger.info("lambda_A: " + str(params.lambda_A))
	tools.logger.info("lambda_B: " + str(params.lambda_B))


	#Then call a local grid search, here only with one value of rank and regularization
	fb15kexp.grid_search_on_all_models(all_params, embedding_size_grid = [emb_size], lmbda_grid = [lmbda], nb_runs = 1)


	#Print best averaged metrics:
	fb15kexp.print_best_MRR_and_hits()



	#Save ComplEx embeddings (last trained model, not best on grid search if multiple embedding sizes and lambdas)
	#e1 = fb15kexp.models["Complex_Logistic_Model"][0].e1.get_value(borrow=True)
	#e2 = fb15kexp.models["Complex_Logistic_Model"][0].e2.get_value(borrow=True)
	#r1 = fb15kexp.models["Complex_Logistic_Model"][0].r1.get_value(borrow=True)
	#r2 = fb15kexp.models["Complex_Logistic_Model"][0].r2.get_value(borrow=True)
	#scipy.io.savemat('complex_embeddings.mat', \
	#		{'entities_real' : e1, 'relations_real' : r1, 'entities_imag' : e2, 'relations_imag' : r2  })2  })
