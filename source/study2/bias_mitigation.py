import pandas as pd
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

privileged_groups = [{'GENDER': 0}]
unprivileged_groups = [{'GENDER': 1}]

svm_params = [{
	'clf__C': [0.01, 0.1, 1, 10],
	'clf__kernel': ['linear']
}]
label_encoder = LabelEncoder()
attr = 'GENDER'
clf = 'DEPRESSION_majority'


def classifier(X_train, X_val, y_train, y_val, weights, clf='DEPRESSION_majority'):
	X_combined = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(X_val).reset_index(drop=True)])
	y_combined = pd.concat([y_train.reset_index(drop=True), y_val.reset_index(drop=True)])
	
	test_fold = [-1] * len(X_train) + [0] * len(X_val)
	
	ps = PredefinedSplit(test_fold=test_fold)
	
	pipeline = Pipeline(steps=[('standardscaler', StandardScaler()),
	                           ('clf', SVC(random_state=0, probability=True, class_weight='balanced'))])
	grid = GridSearchCV(pipeline, svm_params, scoring='f1_macro', verbose=1, cv=ps, n_jobs=-1, refit=False)
	
	# Encode labels
	# label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y_combined[clf])  # F -> 0, M -> 1
	
	grid.fit(X_combined, y_encoded, clf__sample_weight=weights)
	best_params = grid.best_params_
	pipeline.set_params(**best_params)
	model = pipeline.fit(X_train, label_encoder.transform(y_train[clf]), clf__sample_weight=weights)
	# print(grid.best_score_)
	print("VALL: ", f1_score(label_encoder.transform(y_val[clf]), model.predict(X_val), average='macro'))
	print("TRAIN: ", f1_score(label_encoder.transform(y_train[clf]), model.predict(X_train), average='macro'))
	print(best_params)
	return model

def get_predictions(model, X, y):
	y_preds = model.predict(X)
	y_prob = model.predict_proba(X)[:, 1]
	
	y_all = pd.DataFrame(y[[attr, clf]], columns=[attr, clf]).reset_index(drop=True)
	
	data = pd.concat([pd.DataFrame(y_preds, columns=['preds']).reset_index(drop=True),
	                  pd.DataFrame(y_prob, columns=['probability']).reset_index(drop=True),
	                  y_all], axis=1)
	return data

def get_gender_based_classifier(X_train, X_val, y_train, y_val, weights=None):
	female_data = []
	male_data = []
	female_y = []
	male_y = []
	
	for X, y in [[X_train, y_train], [X_val, y_val]]:
		# Extract the feature subsets for each gender
		X_female = X.loc[y[y['GENDER'] == 'F'].index]
		X_male = X.loc[y[y['GENDER'] == 'M'].index]
		
		female_data.append(X_female)
		male_data.append(X_male)
		
		female_y.append((y[y['GENDER'] == 'F']))
		male_y.append(y[y['GENDER'] == 'M'])
	
	model_female = classifier(female_data[0], female_data[1], female_y[0], female_y[1], weights)
	model_male = classifier(male_data[0], male_data[1], male_y[0], male_y[1], weights)
	return model_female, model_male


def get_gender_based_predictions(model1, model2, X_test, y_test):
	# Split the examples based on gender
	female_indices = y_test[y_test[attr] == 'F'].index
	male_indices = y_test[y_test[attr] == 'M'].index
	
	# Extract the feature subsets for each gender
	X_female = X_test.loc[female_indices]
	X_male = X_test.loc[male_indices]
	
	# Get predictions for each gender
	predictions_female = model1.predict(X_female) if not X_female.empty else []
	predictions_male = model2.predict(X_male) if not X_male.empty else []
	# Combine the predictions
	predictions = pd.Series(index=X_test.index, dtype='float64')
	predictions.loc[female_indices] = predictions_female
	predictions.loc[male_indices] = predictions_male
	
	y_prob_female = model1.predict_proba(X_female)[:, 1]
	y_prob_male = model2.predict_proba(X_male)[:, 1]
	predictions_prob = pd.Series(index=X_test.index, dtype='float64')
	predictions_prob.loc[female_indices] = y_prob_female
	predictions_prob.loc[male_indices] = y_prob_male
	
	y_all = pd.DataFrame(y_test[[attr, clf]], columns=[attr, clf]).reset_index(drop=True)
	
	data = pd.concat([pd.DataFrame(predictions, columns=['preds']).reset_index(drop=True),
	                  pd.DataFrame(predictions_prob, columns=['probability']).reset_index(drop=True),
	                  y_all], axis=1)
	
	return data


def pre_processing(aif360_train):
	RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
	RW.fit(aif360_train)
	dataset = RW.transform(aif360_train)
	label_names = dataset.label_names  # or a list of label column names if available
	sensitive_attribute_names = dataset.protected_attribute_names  # or a list of sensitive attribute column names if available
	
	# Exclude label and sensitive attribute names from the feature names list
	feature_names = [name for name in dataset.feature_names if name not in label_names + sensitive_attribute_names]
	features_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
	features_df = features_df[feature_names]
	# features_df = pd.DataFrame(dataset.features, columns=[str(a) for a in range(embedding_length[embedding])])
	return features_df, dataset.labels.ravel(), dataset.instance_weights


def post_processing(val_aif360, test_aif360, val_data, test_data):
	ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups,
	                                 metric_name='Average odds difference'
	                                 )
	aif360_test_pred = test_aif360.copy(deepcopy=True)
	aif360_test_pred.scores = test_data['probability'].values.reshape(-1, 1)
	
	aif360_val_pred = val_aif360.copy(deepcopy=True)
	aif360_val_pred.scores = val_data['probability'].values.reshape(-1, 1)
	
	ROC = ROC.fit(val_aif360, aif360_val_pred)
	
	print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
	print("Optimal ROC margin = %.4f" % ROC.ROC_margin)
	
	# Transform the validation set
	dataset_transf_val_pred = ROC.predict(aif360_val_pred)
	dataset_transf_test_pred = ROC.predict(aif360_test_pred)
	
	test_data = pd.concat([pd.DataFrame(dataset_transf_test_pred.labels, columns=['preds']).reset_index(drop=True),
	                       pd.DataFrame(aif360_test_pred.scores, columns=['probability']).reset_index(drop=True),
	                       test_data[['GENDER', clf]]], axis=1)
	
	val_data = pd.concat([pd.DataFrame(dataset_transf_val_pred.labels, columns=['preds']).reset_index(drop=True),
	                      pd.DataFrame(aif360_val_pred.scores, columns=['probability']).reset_index(drop=True),
	                      val_data[['GENDER', clf]]], axis=1)
	
	for data in [val_data, test_data]:
		data['GENDER'] = data['GENDER'].replace(0, 'F')
		data['GENDER'] = data['GENDER'].replace(1, 'M')
	
	return val_data, test_data


