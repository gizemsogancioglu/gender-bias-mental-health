import collections
import copy

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from source.study2.bias_mitigation import get_predictions

# Encode labels
label_encoder = LabelEncoder()


def create_random(X_test, filename):
	# Create an array with numbers from 0 to 200
	random_array = np.arange((X_test.shape[1]))
	
	np.random.shuffle(random_array)
	print(len(random_array))
	df = pd.DataFrame({'Feature': [f'Feature_{i + 1}' for i in range(X_test.shape[1])],
	                   'index': random_array, })
	
	df.to_csv("../results/random_{file}_{emb}.csv".format(file=filename, emb=embedding))
	return


def pcc_scores(X):
    index = 3
    gender_binary = [0 if val == 'F' else 1 for val in y[index]['GENDER']]

    correlation_scores = []
    for i in range(X[index].shape[1]):
        feature = X[index][:, i]
        corr, _ = pearsonr(feature, gender_binary)
        correlation_scores.append(corr)
    df = pd.DataFrame({'Feature': [f'Feature_{i+1}' for i in range(X[0].shape[1])],'Pearson_Correlation': correlation_scores })
    df['Absolute_Correlation'] = df['Pearson_Correlation'].abs()
    df = df.sort_values(by='Absolute_Correlation', ascending=False)
    filename = str(fold_i)
    pd.DataFrame(df).to_csv(f"../results/pcc_{filename}_{embedding}.csv")

def get_kernel_expl(filename, model, test_features):
    # Define a wrapper for the predict function to return numerical values
    def predict_numerical(X):
        predictions = model.predict(X)
        return np.where(predictions == 'M', 1, 0)

    expl = shap.KernelExplainer(model.predict, test_features.values)
    shap_values = expl.shap_values(test_features.values)
    important_feat = pd.DataFrame(abs(shap_values).mean(0)).reset_index(drop=True)
    important_feat.to_csv("../results/train_val_shap_{file}_{emb}.csv".format(file=filename, emb=embedding))

    #df = pd.DataFrame({'Feature': [f'Feature_{i+1}' for i in range(test_features.shape[1])],
#                                                      'score': important_feat})
    #df['Absolute_Score'] = df['score'].abs()
    #df = df.sort_values(by='Absolute_Score', ascending=False)
    #df.to_csv("../results/train_val_shap_{file}_{emb}.csv".format(file=filename, emb=embedding))

# Function to compute permutation feature importance
def permutation_feature_importance(model, X_test, y_test, filename, metric=f1_score, average='macro'):
    from sklearn.inspection import permutation_importance
    # Custom scoring: using F1 score
    f1_scorer = make_scorer(f1_score, average=average)
    result_f1 = permutation_importance(model, X_test, label_encoder.transform(y_test), n_repeats=20, random_state=42, n_jobs=-1, scoring=f1_scorer)

    # Get feature importance with F1 scoring
    importance_f1 = result_f1.importances_mean
    std_f1 = result_f1.importances_std

    #baseline = metric(y_test, model.predict(X_test), average=average)
    #importances = []
    #for col in range(X_test.shape[1]):
     #   X_permuted = X_test.copy()
      #  np.random.shuffle(X_permuted[:, col])
       # permuted_accuracy = metric(y_test, model.predict(X_permuted), average=average)
        #importance = baseline - permuted_accuracy
        #importances.append(importance)

    df = pd.DataFrame({'Feature': [f'Feature_{i+1}' for i in range(X_test.shape[1])],
                                                                  'score': importance_f1,
                                                                  'std': std_f1 })
    df['Absolute_Score'] = df['score'].abs()
    df = df.sort_values(by='Absolute_Score', ascending=False)
    df.to_csv("../results/pfi_{file}_{emb}.csv".format(file=filename, emb=embedding))

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


def iterative_analysis(fold_i, model, X, X_test, y_test, split, y):
	clf = 'DEPRESSION_majority'
	# clf = 'GENDER'
	file = str(fold_i)
	arr_sub = collections.defaultdict(list)
	
	# print(f"READING FILE ../results/sex_relevant_feat_{file}.csv")
	# df_pcc = pd.read_csv("../results/pcc_{file}_{emb}.csv".format(file=file, emb=embedding))
	df_shap = pd.read_csv("../results/average_shap_{emb}.csv".format(file=file, emb=embedding))
	df_pfi = pd.read_csv("../results/average_pfi_{emb}.csv".format(file=file, emb=embedding))
	# df_random = pd.read_csv("../results/random_{file}_{emb}.csv".format(file=file, emb=embedding))
	# for df in [df_pcc]:
	for df in [df_shap, df_pfi]:
		df['feat'] = X_test.columns
	# df['index'] = df['Unnamed: 0']
	# df = df.sort_values(by=['0'], ascending=False)
	
	params = model.get_params()
	pipeline = Pipeline(steps=[('standardscaler', StandardScaler()),
	                           ('clf', SVC(random_state=0, probability=True, class_weight='balanced'))])
	pipeline.set_params(**params)
	
	# for del_, arr in [["shap", df_shap['index']], ['pcc', df_pcc['index']], ['pfi', df_pfi['index']]]:
	# for del_, arr in [['random', df_random['index']]]:
	for del_, arr in [['shap', df_shap['index']], ['pfi', df_pfi['index']]]:
		
		arr_sub = collections.defaultdict(list)
		
		# for i in range(int(len(arr) - 1)):
		# print("len", len(X_test.columns)-1)
		for i in range(0, len(X_test.columns) - 1, 1):
			scores_arr = collections.defaultdict(list)
			del_val = list(arr[0:i])
			train_data, test_data = nullfy_given_indices(pd.DataFrame(X[0]).reset_index(drop=True), X_test, del_val)
			train_data, val_data = nullfy_given_indices(pd.DataFrame(X[0]).reset_index(drop=True),
			                                            pd.DataFrame(X[3]).reset_index(drop=True), del_val)
			
			# train_data = remove_given_indices(pd.DataFrame(X[0]).reset_index(drop=True), del_val)
			# test_data = remove_given_indices(X_test, del_val)
			# val_data = remove_given_indices(pd.DataFrame(X[3]).reset_index(drop=True), del_val)
			
			# ROAR:
			# pipeline.set_params(**params)
			# model = pipeline.fit(train_data, pd.DataFrame(y[0]).reset_index(drop=True)[clf])
			
			# arr_sub['F1'].append(f1_score((y_test[clf]), model.predict(test_data), average='macro'))
			test_preds = get_predictions(model, test_data, y_test)
			scores_arr = eval(test_preds, scores_arr, int(len(test_preds) / 2), clf, embedding, fold_i, dataset)
			for score in scores_arr.keys():
				arr_sub[score].append(scores_arr[score][0])
		
		df = pd.DataFrame().from_dict(arr_sub, orient='index').transpose()
		print(f"SAVING ITERATIVE ANALYSIS FILE TO {del_}")
		# old_df = pd.read_csv(f"../results/GENDER_ROAR_iterative_{del_}_{embedding}_{split}_{fold_i}.csv".format(del_=del_, embedding=embedding, split=split, fold_i=str(fold_i)))
		
		# df.to_csv(f"../results/iterative_{del_}_{embedding}_{split}_{fold_i}.csv".format(del_=del_, embedding=embedding, split=split, fold_i=str(fold_i)))
		df.to_csv(
			f"../results/avg_iterative_{del_}_{embedding}_{split}_{fold_i}.csv".format(del_=del_, embedding=embedding,
			                                                                           split=split, fold_i=str(fold_i)))
	return


def ours_(fold, method=''):
	# choose the model that gives the lowest score for lambda * mae on validation set.
	df = collections.defaultdict(list)
	for split in ['val', 'test']:
		df[split] = pd.read_csv(
			"../results/bias_iterative_analysis_" + method + "_" + split + f"_{fold}.csv")
		df[split]['ID'] = df[split].index
	func_methods = ['FUNC_' + str(lambda_val) for lambda_val in [0, 0.25, 0.5, 0.75, 1]]
	dict_m = collections.defaultdict(list)
	
	for measure in ['TPR', 'FPR', 'F1']:
		for data in [df['test'], df['val']]:
			data[f'{measure}R'] = [measure_ratio(row[f'{measure}-F'], row[f'{measure}-M']) for index, row in
			                       data.iterrows()]
	
	# df['val'] = df['test']
	# + func_methods
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


def proxyMute(X, y, fold_i):
	# model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
	#                       pd.DataFrame(X[3]).reset_index(drop=True),
	#                      pd.DataFrame(y[0]).reset_index(drop=True),
	#                     pd.DataFrame(y[3]).reset_index(drop=True), weights=None, clf='GENDER')
	
	# get_kernel_expl(str(fold_i), model, pd.concat([pd.DataFrame(X[0]).reset_index(drop=True), pd.DataFrame(X[3]).reset_index(drop=True)]))
	filename = str(fold_i)
	# data = pd.read_csv("../results/train_val_shap_{file}_{emb}.csv".format(file=filename, emb=embedding))
	# print(data.columns)
	# df = pd.DataFrame({'Feature': [f'Feature_{i+1}' for i in range(X[0].shape[1])],
	# 'score': data['0']})
	# df['Absolute_Score'] = df['score'].abs()
	# df = df.sort_values(by='Absolute_Score', ascending=False)
	# df.to_csv("../results/ordered_train_val_shap_{file}_{emb}.csv".format(file=filename, emb=embedding))
	
	model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
	                   pd.DataFrame(X[3]).reset_index(drop=True),
	                   pd.DataFrame(y[0]).reset_index(drop=True),
	                   pd.DataFrame(y[3]).reset_index(drop=True), weights=None, clf='DEPRESSION_majority')
	
	# filename = str(fold_i)
	# importances = permutation_feature_importance(model, X[3], y[3]['GENDER'], filename)
	
	# create_random(X[3], filename)
	for test_data, test_y, split in [pd.DataFrame(X[4]).reset_index(drop=True), y[4], 'val'], [
		pd.DataFrame(X[1]).reset_index(drop=True), y[1], 'test']:
		iterative_analysis(fold_i, model, X, test_data, test_y, split, y)
	
	# save_res('proxymute', 'f1', 'mimic')
	
	# for embedding in ['w2vec_news', 'biowordvec']:
	#   for fold_i in range(1):
	#      ours_(fold_i, method=embedding)
	
	return

