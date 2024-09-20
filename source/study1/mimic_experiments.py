import collections
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from source.study2.text_processing import gender_swapping, swap_gender
import numpy as np
from sklearn.pipeline Pipeline
import pandas as pd

clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
bert_model = 'bert-base-uncased'
w2vec_model = 'word2vec-google-news-300'
pubmed_bert = 'BiomedNLP-PubMedBERT-base-uncased-abstract'

svm_params = [{
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__kernel': ['linear']
}]

C_values = [0.01, 0.1, 1, 10]

def classifier(X_train, X_val, y_train, y_val):
    X_combined = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(X_val).reset_index(drop=True)])
    y_combined = pd.concat([y_train.reset_index(drop=True), y_val.reset_index(drop=True)])

    test_fold = [-1] * len(X_train) + [0] * len(X_val)

    ps = PredefinedSplit(test_fold=test_fold)

    pipeline = Pipeline(steps=[('standardscaler', StandardScaler()),
                               ('clf', SVC(random_state=0, probability=True, class_weight='balanced'))])
    grid = GridSearchCV(pipeline, svm_params, scoring='f1_macro', verbose=1, cv=ps, n_jobs=-1, refit=False)

    grid.fit(X_combined, y_combined[clf])
    best_params = grid.best_params_
    pipeline.set_params(**best_params)

    model = pipeline.fit(X_train, y_train[clf], clf__sample_weight=weights)
    print(grid.best_score_)
    print(f1_score(y_val[clf], model.predict(X_val), average='macro'))
    print(best_params)
    return model


def create_MIMIC(data):
    for val, str_ in [[True, 'neutr'], [False, 'swapped']]:
        tmp = data.copy(deep=True)
        (tmp['TEXT']) = [gender_swapping(row['TEXT'], row['GENDER'], neutralize=val) for index, row in data.iterrows()]
        if str_ == 'swapped':
            (tmp['GENDER']) = [swap_gender(row['GENDER']) for index, row in data.iterrows()]
        tmp.to_csv("../mimic_{str}.csv".format(str=str_), index=False)

def folds_index(pos, neg, pos2, neg2, config):
    folds = collections.defaultdict(list)
    neg_class = 'None-mental'
    #neg_class = 'NONE'
    #neg_class = 'None-dep'
    data = pd.read_csv("../mimic_orig.csv")
    np.random.seed(0)
    for clf in mental_arr:
        subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)
        folds[clf] = collections.defaultdict(list)
        depression_text_data = subset[subset['TEXT'].str.contains("depression", case=False, na=False)]
        remaining_data = subset[~subset.index.isin(depression_text_data.index)]
        for i in range(config):
            train_rows = []
            test_rows = []
            val_rows = []

            for d, g in [[1, 'F'], [1, 'M'], [0, 'F'], [0, 'M']]:
                depression_sub1 = remaining_data[(remaining_data[clf] == d) & (remaining_data['GENDER'] == g)].copy(deep=True)
                depression_sub2 = depression_text_data[(depression_text_data[clf] == d) & (depression_text_data['GENDER'] == g)].copy(deep=True)
                depression_all = subset[(subset[clf] == d) & (subset['GENDER'] == g)].copy(deep=True)
                if (d == 1):
                    train_no = pos["train"+clf][0] if g == 'F' else pos["train"+clf][1]
                    train_no2 = pos2["train"+clf][0] if g == 'F' else pos2["train"+clf][1]

                    val_no = pos["val"+clf][0] if g == 'F' else pos["val"+clf][1]
                    val_no2 = pos2["val"+clf][0] if g == 'F' else pos2["val"+clf][1]

                else:

                    train_no = neg["train"+clf][0] if g == 'F' else neg["train"+clf][1]
                    train_no2 = neg2["train"+clf][0] if g == 'F' else neg2["train"+clf][1]

                    val_no = neg["val"+clf][0] if g == 'F' else neg["val"+clf][1]
                    val_no2 = neg2["val"+clf][0] if g == 'F' else neg2["val"+clf][1]

                rows2 = np.random.choice(depression_sub2.index.values, train_no2, replace=False)    
                rows = np.random.choice(depression_sub1.index.values, train_no, replace=False)

                print("depression {d}, gender {g}, number of examples {e}".format(d=d, g=g, e=len(rows)))
                train_rows = np.concatenate([np.array(train_rows), rows, rows2]).astype(int)
                    #depression_sub = depression_sub[~depression_sub.index.isin(rows)]
                
                rows_val1 = np.random.choice(depression_sub1[~depression_sub1.index.isin(rows)].index.values, val_no, replace=False)
                print(len(depression_sub2), train_no2, val_no2)
                rows_val2 = np.random.choice(depression_sub2[~depression_sub2.index.isin(rows2)].index.values, val_no2, replace=False)
                val_rows = np.concatenate([np.array(val_rows), rows_val1, rows_val2]).astype(int)
                test_rows = np.concatenate(
                    [np.array(test_rows), depression_all[~depression_all.index.isin(np.concatenate([rows,rows2, rows_val1, rows_val2]))].index.values]).astype(int)
                print("LENG val", len(val_rows))
                print("LENG test", len(test_rows))

            folds[clf][str(i)].append(train_rows)
            folds[clf][str(i)].append(val_rows)
            folds[clf][str(i)].append(test_rows)

    return folds

def create_fold_i(name, type, clf, fold_i):
    neg_class = 'None-mental'
    #neg_class = 'NONE'
    #neg_class = 'None-dep'
    folds = collections.defaultdict(list)
    orig_data = pd.read_csv("../mimic_{name}_orig.csv".format(name=name))
    swapped_data = pd.read_csv("../mimic_{name}_swapped.csv".format(name=name))
    neutr_data = pd.read_csv("../mimic_{name}_neutr.csv".format(name=name))

    #print("Reading ../mimic_{name}_{type}.csv".format(name=name, type=type))
    if type != "augmented":
        data = pd.read_csv("../mimic_{name}_{type}.csv".format(name=name, type=type))
        subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)

    subset_orig = orig_data[(orig_data[clf] == 1) | (orig_data[neg_class] == 1)].reset_index(drop=True)
    subset_swapped = swapped_data[(swapped_data[clf] == 1) | (swapped_data[neg_class] == 1)].reset_index(drop=True)
    subset_neutr = neutr_data[(neutr_data[clf] == 1) | (neutr_data[neg_class] == 1)].reset_index(drop=True)

    #print("DEBUG: ", len(fold_i[clf]))
    for i in range(len(fold_i[clf])):
     #   print("DEBUG1: ", i)
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


def set_numbers():
    pos_numbers = collections.defaultdict(list)
    neg_numbers = collections.defaultdict(list)
    clf = 'DEPRESSION_majority'
    for split, pos, neg in [['train', [15,10], [130, 120]], ['val', [5, 10], [17, 58]]]:
        #[['train', [90,90], [140, 140]], ['val', [14, 22], [22, 69]]]:

    # 22, 69
    #for clf, pos_train, neg_train, pos_val, neg_val in [['DEPRESSION_majority', [90, 90], [140, 140], [22, 14], [69, 22]],
                         # ['ALCOHOL.ABUSE_majority', [10, 50], [50, 50]],
                         # ['OTHER.SUBSTANCE.ABUSE_majority', [10, 50], [50, 50]],
                         # ['SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS_majority', [55, 55], [55, 55]], ]:
        pos_numbers[split+clf] = pos
        neg_numbers[split+clf] = neg
        

    return pos_numbers, neg_numbers

def set_numbers2():
    pos_numbers = collections.defaultdict(list)
    neg_numbers = collections.defaultdict(list)
    clf = 'DEPRESSION_majority'
    for split, pos, neg in [['train', [75,80], [10, 20]], ['val', [10, 13], [5, 11]]]:
    # 22, 69
    #for clf, pos_train, neg_train, pos_val, neg_val in [['DEPRESSION_majority', [90, 90], [140, 140], [22, 14], [69, 22]],
                         # ['ALCOHOL.ABUSE_majority', [10, 50], [50, 50]],
                         # ['OTHER.SUBSTANCE.ABUSE_majority', [10, 50], [50, 50]],
                         # ['SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS_majority', [55, 55], [55, 55]], ]:
        pos_numbers[split+clf] = pos
        neg_numbers[split+clf] = neg


    return pos_numbers, neg_numbers

def get_predictions(model, X, y, index):
    y_preds = model.predict(X[index])
    y_prob = model.predict_proba(X[index])[:, 1]

    y_all = pd.DataFrame(y[index][[attr, clf]], columns=[attr, clf]).reset_index(drop=True)

    data = pd.concat([pd.DataFrame(y_preds, columns=['preds']).reset_index(drop=True),
                      pd.DataFrame(y_prob, columns=['probability']).reset_index(drop=True),
                      y_all], axis=1)
    return data

def stratified_split(indices, stratify_labels, test_size=0.2, random_state=0):
        # Split indices into training and temp (validation + test) sets
    train_indices, temp_indices = train_test_split(indices, test_size=test_size*1.5, stratify=stratify_labels, random_state=random_state)
                
    # Further split temp into actual validation and test sets
    stratify_temp = stratify_labels[temp_indices].reset_index(drop=True)
    val_indices, test_indices = train_test_split(temp_indices, test_size=1/3, stratify=stratify_temp, random_state=random_state)
    

    return train_indices, val_indices, test_indices

def fold_cv(validation_split=0.2, config=10):
    folds = collections.defaultdict(list)

    data = pd.read_csv("../mimic_orig.csv")
    neg_class = 'None-mental'
    n_splits = 10
    validation_split = 0.1
    for clf in mental_arr:
        i = 0
        folds[clf] = collections.defaultdict(list)
        subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)

        indices_with_depression = subset[subset['TEXT'].str.contains("depression", case=False, na=False)].index
        all_indices = np.arange(len(subset))
        remaining_indices = np.setdiff1d(all_indices, indices_with_depression)


        sub_dep = subset.iloc[indices_with_depression]
        sub_rem = subset.iloc[remaining_indices]

        for subset in [sub_dep, sub_rem]: 
            i=0
            for random_state in range(config):
                stratify_label = subset[clf].astype(str) + subset[attr].astype(str)
                # Initialize a stratified 10-fold cross-validator
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                #print(stratify_label.index)

                for trainval_idx, test_idx in cv.split(subset, stratify_label): 
                    #trainval_data = [stratify_label[i] for i in trainval_idx]
                    orig_trainval_idx = subset.iloc[trainval_idx].index
                    orig_test_idx = subset.iloc[test_idx].index

                    trainval_data = stratify_label.iloc[trainval_idx]
                    cv_validation = StratifiedKFold(n_splits=int(1/validation_split))
                                                     
                    # Use the first split as the validation set
                    for train_index, validation_index in cv_validation.split(np.zeros(len(trainval_data)), trainval_data):
                    # Adjust indices to original data size
                        orig_train_idx = orig_trainval_idx[train_index]
                        orig_val_idx = orig_trainval_idx[validation_index]
                        break  # Only need the first split
                        

                    if folds[clf][str(i)]:

                        folds[clf][str(i)][0] = (np.concatenate([orig_train_idx, folds[clf][str(i)][0]]))
                        folds[clf][str(i)][1] = (np.concatenate([orig_val_idx, folds[clf][str(i)][1]]))
                        folds[clf][str(i)][2] = (np.concatenate([orig_test_idx, folds[clf][str(i)][2]]))

                    else:
                        folds[clf][str(i)].append(orig_train_idx)
                        folds[clf][str(i)].append(orig_val_idx)
                        folds[clf][str(i)].append(orig_test_idx)

                    i += 1

    return folds

def kfold(config):
    folds = collections.defaultdict(list)

    data = pd.read_csv("../mimic_orig.csv")
    neg_class = 'None-mental'
    for clf in mental_arr:
        folds[clf] = collections.defaultdict(list)
        
        subset = data[(data[clf] == 1) | (data[neg_class] == 1)].reset_index(drop=True)
        for split in range(config):
            indices_with_depression = subset[subset['TEXT'].str.contains("depression", case=False, na=False)].index
            all_indices = np.arange(len(subset))
            remaining_indices = np.setdiff1d(all_indices, indices_with_depression)
            
            # Assume df has 'gender' and 'class' columns for stratification
            stratify_with_depression = subset.loc[indices_with_depression, ['GENDER', clf]].astype(str).agg('_'.join, axis=1)
            stratify_remaining = subset.loc[remaining_indices, ['GENDER', clf]].astype(str).agg('_'.join, axis=1)
            # Step 1: Split indices_with_depression
            train_dep, val_dep, test_dep = stratified_split(indices_with_depression, stratify_with_depression, random_state=split)

            # Step 2: Split remaining_indices
            train_rem, val_rem, test_rem = stratified_split(remaining_indices, stratify_remaining, random_state=split)

            # Step 3: Merge corresponding sets
            train_final = np.concatenate([train_dep, train_rem])
            val_final = np.concatenate([val_dep, val_rem])
            test_final = np.concatenate([test_dep, test_rem])

            #print(train_indices)     
        
            folds[clf][str(split)].append(train_final)
            folds[clf][str(split)].append(val_final)
            folds[clf][str(split)].append(test_final)
        
    return folds

def experiment(method_name, folds, config, embedding, clf):

    for fold_i in range(config): 
        X, y = [features[[str(a) for a in range(embedding_length[embedding])]].values for features in folds[str(fold_i)]], [y[[clf, 'GENDER', 'TEXT']] for y in folds[str(fold_i)]] 
        #print("size of the training X: {x} and Y: {yi}".format(x=X[0].shape, yi=(y[0].shape)))
        #print("size of the test X: {x} and Y: {yi}".format(x=X[1].shape, yi=(y[1].shape)))  
        i = int(len(y[1]) / 2) 
        column_names = [str(a) for a in range(embedding_length[embedding])] + ['GENDER', clf] 
        train_data = folds[str(fold_i)][0][column_names] 

        if method_name in ['orig', 'neutr',  'augmented']:
            model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
                            pd.DataFrame(X[3]).reset_index(drop=True),
                            pd.DataFrame(y[0]).reset_index(drop=True),
                            pd.DataFrame(y[3]).reset_index(drop=True))


            if method_name == 'neutr':
                data = get_predictions(model, X, y, 2)
            else:
                data = get_predictions(model, X, y, 1)

            val_data = get_predictions(model, X, y, 4)

        data.to_csv("../preds/MIMIC/{measure}/predictions_MIMIC_{i}_fold{fold_i}_{emb}.csv".format(i=method_name,measure=measure,fold_i=fold_i,emb=embedding))
        val_data.to_csv("../preds/MIMIC/{measure}/val_predictions_MIMIC_{i}_fold{fold_i}_{emb}.csv".format(i=method_name,measure=measure,fold_i=fold_i,emb=embedding))
    
    return

              
if __name__ == "__main__":
    print("*********** Bias analysis EXPERIMENTS ********")
    clf = 'DEPRESSION_majority'
    attr = 'GENDER'

    mental_arr = ['DEPRESSION_majority']
    embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']

    data = pd.read_csv("../mimic_orig.csv", index_col=None)

    #create_MIMIC(data)
    #for type in ['neutr']:
          #extract_all_feat(type)

    embedding_length = collections.defaultdict(list)
    embedding_length['w2vec_news'] = 300
    embedding_length['biowordvec'] = 200
    embedding_length['bert'] = 768
    embedding_length['clinical_bert'] = 768
    dataset= 'MIMIC'
    label = clf
    
    #pos, neg = set_numbers()
    #pos2, neg2 = set_numbers2()
    
    measure = 'f1'
    config = 50
    
    #folds_index = folds_index(pos, neg, pos2, neg2, config)
    folds_index = fold_cv(config=10)
    
    #methods = {'orig': ['orig', 'postprocessing', 'gender_specific', 'preprocessing'], 'neutr': ['neutr'], 'augmented': ['augmented']}
    methods = {'orig': ['preprocessing']}
    for clf in mental_arr:
        for data_key, values in methods.items():
                for method in values:
                    for embedding in embeddings:
                        print(f"method {method} will be evaluated using {data_key}.")
                        folds = create_fold_i(embedding, data_key, clf, folds_index)
                        experiment(method, folds, config, embedding, clf)
                   
                

