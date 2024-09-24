import collections
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.model_selection import StratifiedKFold

from source.common.text_processing import gender_swapping, swap_gender

clf = 'DEPRESSION_majority'
attr = 'GENDER'

def create_MIMIC(data):
	for val, str_ in [[True, 'neutr'], [False, 'swapped']]:
		tmp = data.copy(deep=True)
		(tmp['TEXT']) = [gender_swapping(row['TEXT'], row['GENDER'], neutralize=val) for index, row in data.iterrows()]
		if str_ == 'swapped':
			(tmp['GENDER']) = [swap_gender(row['GENDER']) for index, row in data.iterrows()]
		tmp.to_csv("../mimic_{str}.csv".format(str=str_), index=False)


def create_fold_i(name, type, clf, fold_i):
	neg_class = 'None-mental'
	folds = collections.defaultdict(list)
	orig_data = pd.read_csv("../features/mimic_{name}_orig.csv".format(name=name))
	swapped_data = pd.read_csv("../features/mimic_{name}_swapped.csv".format(name=name))
	neutr_data = pd.read_csv("../features/mimic_{name}_neutr.csv".format(name=name))
	
	if type != "augmented":
		data = pd.read_csv("../features/mimic_{name}_{type}.csv".format(name=name, type=type))
		subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)
	
	subset_orig = orig_data[(orig_data[clf] == 1) | (orig_data[neg_class] == 1)].reset_index(drop=True)
	subset_swapped = swapped_data[(swapped_data[clf] == 1) | (swapped_data[neg_class] == 1)].reset_index(drop=True)
	subset_neutr = neutr_data[(neutr_data[clf] == 1) | (neutr_data[neg_class] == 1)].reset_index(drop=True)
	
	for i in range(len(fold_i[clf])):
		if type == "augmented":
			train_df = pd.concat([subset_orig.loc[fold_i[clf][str(i)][0]], subset_swapped.loc[fold_i[clf][str(i)][0]]])
			val_df = pd.concat([subset_orig.loc[fold_i[clf][str(i)][1]]])
		else:
			train_df = subset.loc[fold_i[clf][str(i)][0]]
			val_df = subset.loc[fold_i[clf][str(i)][1]]
		
		folds[str(i)].append(train_df)
		
		test_df = subset_orig.loc[fold_i[clf][str(i)][2]]
		augmented_test_df = pd.concat([test_df, subset_swapped.loc[fold_i[clf][str(i)][2]]])
		folds[str(i)].append(augmented_test_df)
		
		test_df = subset_neutr.loc[fold_i[clf][str(i)][2]]
		swapped_test = test_df.copy(deep=True)
		swapped_test['GENDER'] = [swap_gender(row['GENDER']) for index, row in swapped_test.iterrows()]
		
		augmented_blind_df = pd.concat([test_df, swapped_test])
		
		folds[str(i)].append(augmented_blind_df)
		folds[str(i)].append(val_df)
		
		augmented_val_df = pd.concat([val_df, subset_swapped.loc[fold_i[clf][str(i)][1]]])
		if type == 'neutr':
			swapped_val = val_df.copy(deep=True)
			swapped_val['GENDER'] = [swap_gender(row['GENDER']) for index, row in val_df.iterrows()]
			folds[str(i)].append(pd.concat([val_df, swapped_val]))
		else:
			folds[str(i)].append(augmented_val_df)
	return folds


def convert_dataset(train_data, clf):
	for data in [train_data]:
		data['GENDER'] = data['GENDER'].replace('F', 0)
		data['GENDER'] = data['GENDER'].replace('M', 1)
	
	train_aif360 = BinaryLabelDataset(df=train_data, label_names=[clf], protected_attribute_names=['GENDER'])
	# fav label:1, unfav: 0
	return train_aif360


def fold_cv(validation_split=0.1, config=10):
	folds = collections.defaultdict(list)
	
	data = pd.read_csv("../data/mimic_orig.csv")
	
	neg_class = 'None-mental'
	n_splits = 10
	
	for clf in ['DEPRESSION_majority']:
		i = 0
		folds[clf] = collections.defaultdict(list)
		subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)
		
		indices_with_depression = subset[subset['TEXT'].str.contains("depression", case=False, na=False)].index
		all_indices = np.arange(len(subset))
		remaining_indices = np.setdiff1d(all_indices, indices_with_depression)
		
		sub_dep = subset.iloc[indices_with_depression]
		sub_rem = subset.iloc[remaining_indices]
		
		for subset in [sub_dep, sub_rem]:
			i = 0
			for random_state in range(config):
				stratify_label = subset[clf].astype(str) + subset[attr].astype(str)
				# Initialize a stratified 10-fold cross-validator
				cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
				# print(stratify_label.index)
				
				for trainval_idx, test_idx in cv.split(subset, stratify_label):
					# trainval_data = [stratify_label[i] for i in trainval_idx]
					orig_trainval_idx = subset.iloc[trainval_idx].index
					orig_test_idx = subset.iloc[test_idx].index
					
					trainval_data = stratify_label.iloc[trainval_idx]
					cv_validation = StratifiedKFold(n_splits=int(1 / validation_split))
					
					# Use the first split as the validation set
					for train_index, validation_index in cv_validation.split(np.zeros(len(trainval_data)),
					                                                         trainval_data):
						# Adjust indices to original data size
						orig_train_idx = orig_trainval_idx[train_index]
						orig_val_idx = orig_trainval_idx[validation_index]
						break  # Only need the first split
					
					if str(i) in folds[clf]:
						
						folds[clf][str(i)][0] = (np.concatenate([orig_train_idx, folds[clf][str(i)][0]]))
						folds[clf][str(i)][1] = (np.concatenate([orig_val_idx, folds[clf][str(i)][1]]))
						folds[clf][str(i)][2] = (np.concatenate([orig_test_idx, folds[clf][str(i)][2]]))
					
					else:
						folds[clf][str(i)] = [orig_train_idx, orig_val_idx, orig_test_idx]
					
					i += 1
	
	return folds
