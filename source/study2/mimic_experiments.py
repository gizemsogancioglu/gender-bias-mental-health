from sklearn.metrics import f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bias_mitigation import post_processing, get_gender_based_predictions, get_gender_based_classifier, \
	pre_processing, classifier, get_predictions
from data_prep import convert_dataset, create_fold_i, fold_cv
from embeddings import set_length
from proxymute import proxyMute

def experiment(method_name, folds, config, embedding, clf):
	label_encoder = LabelEncoder()
	for fold_i in range(0, config):
		X, y = [features[[str(a) for a in range(embedding_length[embedding])]].values for features in
		        folds[str(fold_i)]], [y[[clf, 'GENDER', 'TEXT']] for y in folds[str(fold_i)]]
		column_names = [str(a) for a in range(embedding_length[embedding])] + ['GENDER', clf]
		train_data = folds[str(fold_i)][0][column_names]
		aif360_train = convert_dataset(train_data, clf)
		label_encoder.fit(y[0][clf])
		if method_name == 'preprocessing':
			reweighed_train, labels, weights = pre_processing(aif360_train)
		else:
			weights = None
		
		if method_name in ['orig', 'neutr', 'augmented', 'preprocessing']:
			model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
			                   pd.DataFrame(X[3]).reset_index(drop=True),
			                   pd.DataFrame(y[0]).reset_index(drop=True),
			                   pd.DataFrame(y[3]).reset_index(drop=True), weights)
			
			
			print(f1_score(label_encoder.transform(y[1][clf]), model.predict(X[1]), average='macro'))
			if method_name == 'neutr':
				data = get_predictions(model, X[2], y[2])
			else:
				data = get_predictions(model, X[1], y[1])
			
			val_data = get_predictions(model, X[4], y[4])
		
		elif method_name == 'gender_specific':
			model1, model2 = get_gender_based_classifier(pd.DataFrame(X[0]).reset_index(drop=True),
			                                             pd.DataFrame(X[3]).reset_index(drop=True),
			                                             pd.DataFrame(y[0]).reset_index(drop=True),
			                                             pd.DataFrame(y[3]).reset_index(drop=True), weights=None)
			data = get_gender_based_predictions(model1, model2, pd.DataFrame(X[1]).reset_index(drop=True),
			                                    pd.DataFrame(y[1]).reset_index(drop=True))
			val_data = get_gender_based_predictions(model1, model2, pd.DataFrame(X[4]).reset_index(drop=True),
			                                        pd.DataFrame(y[4]).reset_index(drop=True))
		
		elif method_name in ['postprocessing', 'inprocessing']:
			if method == 'postprocessing':
				val_data = pd.read_csv(
					"../../preds/MIMIC/val_predictions_MIMIC_orig_fold{fold_i}_{emb}.csv".format(measure=measure,
					                                                                                    fold_i=fold_i,
					                                                                                    emb=embedding))
				test_data = pd.read_csv(
					"../../preds/MIMIC/predictions_MIMIC_orig_fold{fold_i}_{emb}.csv".format(measure=measure,
					                                                                                fold_i=fold_i,
					                                                                                emb=embedding))
			else:
				val_data = folds[str(fold_i)][4][column_names]
				test_data = folds[str(fold_i)][1][column_names]
			aif360_val = convert_dataset(val_data, clf)
			aif360_test = convert_dataset(test_data, clf)
			if method == 'postprocessing':
				val_data, data = post_processing(aif360_val, aif360_test, val_data, test_data)
			
		elif method_name == 'proxymute':
			proxyMute(X, y, fold_i)
			data = pd.DataFrame()
			val_data = pd.DataFrame()
		
		else:
			data = pd.DataFrame()
			val_data = pd.DataFrame()
		
		data.to_csv(f"../../preds/MIMIC/predictions_MIMIC_{method_name}_fold{fold_i}_{embedding}.csv")
		
		val_data.to_csv(f"../../preds/MIMIC/val_predictions_MIMIC_{method_name}_fold{fold_i}_{embedding}.csv")


    
    #return


if __name__ == "__main__":
	print("*********** Bias analysis EXPERIMENTS ********")
	clf = 'DEPRESSION_majority'
	attr = 'GENDER'
	mental_arr = ['DEPRESSION_majority']
	# embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']
	embeddings = ['w2vec_news']
	
	data = pd.read_csv("../data/mimic_orig.csv", index_col=None)
	# create_MIMIC(data)
	# for type in ['neutr']:
	# extract_all_feat(type)

    ###### WE ASSUME THAT FEATURES ARE ALREADY EXTRACTED #####################
	embedding_length = set_length()
	dataset = 'MIMIC'
	label = clf
	
	measure = 'f1'
	# config = 50
	config = 10
	folds_index = fold_cv(config=config)
	print(len(folds_index['DEPRESSION_majority']['0'][0]))
	print(len(folds_index['DEPRESSION_majority']['0'][1]))
	print(len(folds_index['DEPRESSION_majority']['0'][2]))
	# methods = {'orig': ['orig', 'postprocessing', 'gender_specific', 'preprocessing'], 'neutr': ['neutr'], 'augmented': ['augmented']}
	methods = {'orig': ['orig']}
	for clf in mental_arr:
		for data_key, values in methods.items():
			for method in values:
				for embedding in embeddings:
					print(f"method {method} will be evaluated using {data_key}.")
					folds = create_fold_i(embedding, data_key, clf, folds_index)
					experiment(method, folds, config, embedding, clf)
