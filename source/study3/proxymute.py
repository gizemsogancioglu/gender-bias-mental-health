import collections
import copy
import os

import numpy as np
import pandas as pd
# from interpret.blackbox import shap
import shap
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from source.study2.bias_mitigation import get_predictions, classifier
from source.study2.data_prep import fold_cv, create_fold_i
from source.study2.embeddings import set_length
from source.study2.eval import measure_ratio, evaluate
from sklearn.inspection import permutation_importance

# Encode labels
label_encoder = LabelEncoder()


def create_random(X_test, filename):
	# Create an array with numbers from 0 to 200
	random_array = np.arange((X_test.shape[1]))
	np.random.shuffle(random_array)
	print(len(random_array))
	df = pd.DataFrame({'Feature': [f'Feature_{i + 1}' for i in range(X_test.shape[1])],
	                   'index': random_array, })
	df.to_csv("../explanations/random_{file}_{emb}.csv".format(file=filename, emb=embedding))
	return


def pcc_scores(X, filename):
	index = 3
	gender_binary = [0 if val == 'F' else 1 for val in y[index]['GENDER']]
	
	correlation_scores = []
	for i in range(X[index].shape[1]):
		feature = X[index][:, i]
		corr, _ = pearsonr(feature, gender_binary)
		correlation_scores.append(corr)
	df = pd.DataFrame(
		{'Feature': [f'Feature_{i + 1}' for i in range(X[0].shape[1])], 'Pearson_Correlation': correlation_scores})
	df['Absolute_Correlation'] = df['Pearson_Correlation'].abs()
	df = df.sort_values(by='Absolute_Correlation', ascending=False)
	pd.DataFrame(df).to_csv(f"../explanations/pcc_{filename}_{embedding}.csv")


def get_kernel_expl(filename, model, test_features):
	# Define a wrapper for the predict function to return numerical values
	def predict_numerical(X):
		predictions = model.predict(X)
		return np.where(predictions == 'M', 1, 0)
	
	expl = shap.KernelExplainer(model.predict, test_features.values)
	shap_values = expl.shap_values(test_features.values)
	important_feat = pd.DataFrame(abs(shap_values).mean(0)).reset_index(drop=True)
	df = pd.DataFrame({'Feature': [f'Feature_{i + 1}' for i in range(test_features.shape[1])],
	                   'score': important_feat})
	df['Absolute_Score'] = df['score'].abs()
	df = df.sort_values(by='Absolute_Score', ascending=False)
	df.to_csv("../explanations/shap_{file}_{emb}.csv".format(file=filename, emb=embedding))


# Function to compute permutation feature importance
def permutation_feature_importance(model, X_test, y_test, filename, metric=f1_score, average='macro'):
	# Custom scoring: using F1 score
	f1_scorer = make_scorer(f1_score, average=average)
	result_f1 = permutation_importance(model, X_test, label_encoder.transform(y_test), n_repeats=20, random_state=42,
	                                   n_jobs=-1, scoring=f1_scorer)
	
	# Get feature importance with F1 scoring
	importance_f1 = result_f1.importances_mean
	std_f1 = result_f1.importances_std
	
	df = pd.DataFrame({'Feature': [f'Feature_{i + 1}' for i in range(X_test.shape[1])],
	                   'score': importance_f1,
	                   'std': std_f1})
	df['Absolute_Score'] = df['score'].abs()
	df = df.sort_values(by='Absolute_Score', ascending=False)
	df.to_csv("../explanations/pfi_{file}_{emb}.csv".format(file=filename, emb=embedding))
	return np.array(importance_f1)


def remove_given_indices(df, index_arr):
	df_new = df.T.reset_index(drop=True)
	df_new = (df_new.loc[~df_new.index.isin(index_arr), :])
	return (df_new.T)


def nullfy_given_indices(df, test_df, index_arr, strategy='mean'):
	df_new = copy.deepcopy(df)
	test_df_new = copy.deepcopy(test_df)
	mean_df = df_new.mean().T.to_frame()
	mean_df['feat_name'] = mean_df.index
	mean_df = mean_df.reset_index(drop=True)
	for data in [df_new, test_df_new]:
		for index in index_arr:
			feature = mean_df.loc[index]['feat_name']
			data[feature] = mean_df.loc[index][0]
	return df_new, test_df_new


def iterative_analysis(fold_i, model, X, X_test, y_test, split, expl, y, method='mute'):
	filename = str(fold_i)
	
	df_expl = pd.read_csv(f"../explanations/{expl}_{filename}_{embedding}.csv")
	df_expl['feat'] = X_test.columns
	df_expl['index'] = df_expl['Unnamed: 0']
	
	if os.path.isfile(
			f"../results/iterative_{method}_{expl}_{embedding}_{split}_{fold_i}.csv".format(fold_i=str(fold_i))):
		print("Iterative analysis already exists...")
	return
	
	clf = 'DEPRESSION_majority'
	df_expl = pd.read_csv(f"../explanations/{expl}_{filename}_{embedding}.csv")
	df_expl['feat'] = X_test.columns
	
	df_expl['index'] = df_expl['Unnamed: 0']
	
	params = model.get_params()
	pipeline = Pipeline(steps=[('standardscaler', StandardScaler()),
	                           ('clf', SVC(random_state=0, probability=True, class_weight='balanced'))])
	pipeline.set_params(**params)
	
	for arr in [df_expl['index']]:
		arr_sub = collections.defaultdict(list)
		for i in range(0, len(X_test.columns) - 1, 1):
			scores_arr = collections.defaultdict(list)
			del_val = list(arr[0:i])
			if method == 'mute':
				train_data, test_data = nullfy_given_indices(pd.DataFrame(X[0]).reset_index(drop=True), X_test, del_val)
			elif method == 'roar':
				train_data = remove_given_indices(pd.DataFrame(X[0]).reset_index(drop=True), del_val)
				test_data = remove_given_indices(X_test, del_val)
				
				pipeline.set_params(**params)
				model = pipeline.fit(train_data, pd.DataFrame(y[0]).reset_index(drop=True)[clf])
			
			test_preds = get_predictions(model, test_data, y_test)
			scores_arr = evaluate(test_preds, scores_arr, int(len(test_preds) / 2), clf, embedding, fold_i, dataset)
			for score in scores_arr.keys():
				arr_sub[score].append(scores_arr[score][0])
		
		df = pd.DataFrame().from_dict(arr_sub, orient='index').transpose()
		print(f"SAVING ITERATIVE ANALYSIS FILE TO {del_}")
		
		df.to_csv(
			f"../results/iterative_{method}_{expl}_{embedding}_{split}_{fold_i}.csv".format(fold_i=str(fold_i)))
	return


def model_selection(fold, embedding, method, expl):
	# choose the model that gives the lowest score for lambda * mae on validation set.
	df = collections.defaultdict(list)
	for split in ['val', 'test']:
		df[split] = pd.read_csv(
			f"../results/iterative_{method}_{expl}_{embedding}_{split}_{fold}.csv")
		df[split]['ID'] = df[split].index
	func_methods = ['FUNC_' + str(lambda_val) for lambda_val in [0, 0.25, 0.5, 0.75, 1]]
	dict_m = collections.defaultdict(list)
	
	for measure in ['TPR', 'FPR', 'F1']:
		for data in [df['test'], df['val']]:
			data[f'{measure}R'] = [measure_ratio(row[f'{measure}-F'], row[f'{measure}-M']) for index, row in
			                       data.iterrows()]
	
	df['val'] = df['val'][0:10]
	for m in ['mismatch_ratio', 'TPRR', 'FPRR', 'F1R'] + func_methods:
		metric = m
		if m in ['TPRR', 'FPRR', 'F1R']:
			df['val'] = df['val'].sort_values(by=[m], ascending=False)
		
		elif m in ['mismatch_ratio']:
			df['val'] = df['val'].sort_values(by=[m], ascending=True)
		
		elif m.startswith('FUNC_'):
			lambda_val = float(m.split('_')[1])
			df['val']['fairness_score'] = [(abs(row['TPRR'])) for index, row in
			                               df['val'].iterrows()]
			
			df['val']['threshold_func'] = [
				(lambda_val * (float(row["fairness_score"]))) + ((1 - lambda_val) * (float(row["F1"]))) for
				index, row in df['val'].iterrows()]
			df['val'] = df['val'].sort_values(by=['threshold_func'], ascending=False)
			metric = 'threshold_func'
		
		print(df['val'][df['val'][metric] == df['val'].iloc[0][metric]])
		index = \
			df['val'][df['val'][metric] == df['val'].iloc[0][metric]].sort_values(by=['ID'], ascending=True).iloc[0][
				'ID']
		
		metrics = ['mismatch_ratio']
		# ["TPRR", "FPRR", "mismatch_ratio", 'F1R']
		df['test']['ID'] = df['test'].index
		sub = df['test'][df['test']['ID'] == index]
		dict_m[m] = sub.values[0]
	# mm, f1, func0, func25, func5, func75 = \
	#   df['test'][df['test']['ID'] == index][["mismatch_ratio", 'F1', 'FUNC_0', 'FUNC_0.25', 'FUNC_0.5', 'FUNC_0.75']].values[0]
	# dict_m[m] = [mm, f1, func0, func25, func5, func75]
	
	df_all = pd.DataFrame.from_dict(dict_m)
	df_all.index = df['test'].columns
	df_all.to_csv("../results/ours_mimic_{method}_{fold}.csv".format(method=method, fold=fold))
	return


def get_explanations(X, y, fold_i, expl='shap'):
	filename = str(fold_i)
	if os.path.isfile(f"../explanations/{expl}_{filename}_{embedding}.csv"):
		print("Explanations already exist...")
		return
	model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
	                   pd.DataFrame(X[3]).reset_index(drop=True),
	                   pd.DataFrame(y[0]).reset_index(drop=True),
	                   pd.DataFrame(y[3]).reset_index(drop=True), weights=None, clf='GENDER')
	
	if expl == 'shap':
		get_kernel_expl(str(fold_i), model,
		                pd.concat(
			                [pd.DataFrame(X[0]).reset_index(drop=True), pd.DataFrame(X[3]).reset_index(drop=True)]))
	elif expl == 'pfi':
		permutation_feature_importance(model, X[3], y[3]['GENDER'], filename)
	elif expl == 'pcc':
		pcc_scores(X, filename)
	else:
		create_random(X[3], filename)


def ours(X, y, fold_i, expl, method):
	####### Step1: get explanations using the given explanation method (expl)############################
	get_explanations(X, y, fold_i, expl)
	
	model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
	                   pd.DataFrame(X[3]).reset_index(drop=True),
	                   pd.DataFrame(y[0]).reset_index(drop=True),
	                   pd.DataFrame(y[3]).reset_index(drop=True), weights=None, clf='DEPRESSION_majority')
	
	####### Step 2: mute the features cumulatively and compute fairness and performance in each step. ###
	for test_data, test_y, split in [pd.DataFrame(X[4]).reset_index(drop=True), y[4], 'val'], [
		pd.DataFrame(X[1]).reset_index(drop=True), y[1], 'test']:
		iterative_analysis(fold_i, model, X, test_data, test_y, split, expl, y, method)
	
	###### Step 3: select the model that gives optimum scores ###########################################
	model_selection(fold_i, embedding, method, expl)
	
	return


if __name__ == "__main__":
	print("*********** Bias analysis EXPERIMENTS ********")
	clf = 'DEPRESSION_majority'
	attr = 'GENDER'
	embeddings = ['biowordvec']  # or w2vec_news
	
	###### WE ASSUME THAT FEATURES ARE ALREADY EXTRACTED #####################
	embedding_length = set_length()
	dataset = 'MIMIC'
	label = clf
	
	measure = 'f1'
	config = 5
	folds_index = fold_cv(config=config)
	explanation = 'shap'  # or pfi, pcc, random
	method = 'mute'  # or roar
	for embedding in embeddings:
		folds = create_fold_i(embedding, 'orig', clf, folds_index)
		for fold_i in range(0, config):
			X, y = [features[[str(a) for a in range(embedding_length[embedding])]].values for features in
			        folds[str(fold_i)]], [y[[clf, 'GENDER', 'TEXT']] for y in folds[str(fold_i)]]
			
			ours(X, y, fold_i, explanation, method)
