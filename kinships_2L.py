#import scipy.io

import efe
from efe.exp_generators import *
import efe.tools as tools

if __name__ =="__main__":

	#Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
	fb15kexp = build_data(name = 'Kinship',path = tools.cur_path + '/datasets/')
	#SGD hyper-parameters:
	params = Parameters(learning_rate = 0.1, 
						max_iter = 1000, 
						batch_size = int(len(fb15kexp.train.values) / 100),  #Make 100 batches
						neg_ratio = 1, 
						valid_scores_every = 1000,
						learning_rate_policy = 'adagrad',
						contiguous_sampling = False )

	all_params = { "wTransE_2L_Model" : params } ; emb_size = 200; lmbda =5;params.miuA = 0.5;params.miuB = 0.5; params.lambda_A =0.002;params.lambda_B=0.004

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
