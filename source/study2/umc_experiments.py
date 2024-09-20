import collections
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import recall_score
import nltk
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.metrics import classification_report, make_scorer, mean_absolute_error, f1_score, confusion_matrix, \
    roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import bias_analysis as bias
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gensim.downloader
import gensim.downloader as api
from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd
import sent2vec
import tensorflow.compat.v1 as tf
import re
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from tqdm import tqdm
from eval import eval
tf.disable_eager_execution()

privileged_groups = [{'GENDER': 1}]
unprivileged_groups = [{'GENDER': 0}]


def post_processing(val_aif360, test_aif360, val_data, test_data):
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     metric_name='Average odds difference'
                                     )
    aif360_test_pred = test_aif360.copy(deepcopy=True)
    print(test_data['probability'].values.reshape(-1,1))
    aif360_test_pred.scores = test_data['probability'].values.reshape(-1,1)

    aif360_val_pred = val_aif360.copy(deepcopy=True)
    aif360_val_pred.scores = val_data['probability'].values.reshape(-1,1)

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


    return val_data, test_data



def pre_processing(aif360_train):
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                           privileged_groups=privileged_groups)
    RW.fit(aif360_train)
    dataset = RW.transform(aif360_train)
    label_names = dataset.label_names  # or a list of label column names if available
    sensitive_attribute_names = dataset.protected_attribute_names  # or a list of sensitive attribute column names if available

    # Exclude label and sensitive attribute names from the feature names list
    feature_names = [name for name in dataset.feature_names if name not in label_names + sensitive_attribute_names]
    features_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    features_df = features_df[feature_names]

    print(features_df.shape)
    #features_df = pd.DataFrame(dataset.features, columns=[str(a) for a in range(embedding_length[embedding])])
    return features_df, dataset.labels.ravel(), dataset.instance_weights

def convert_dataset(train_data, clf):

    train_aif360 = BinaryLabelDataset(df=train_data, label_names=[clf], protected_attribute_names=['GENDER'])
    

    # fav label:1, unfav: 0
    return  train_aif360


def in_processing(train_aif360, test_aif360):
    tf.reset_default_graph()
    sess = tf.Session()
    model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                 unprivileged_groups=unprivileged_groups,
                                 # scope_name='plain_classifier',
                                 scope_name='debiased_classifier',
                                 # debias=False,
                                 debias=True,
                                 sess=sess,
                                 batch_size=1,
                                 num_epochs=100)

    model.fit(train_aif360)
    test_preds = model.predict(test_aif360)

    classified_metric_debiasing_test = ClassificationMetric(test_aif360, test_preds,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())

    return test_preds.labels

## FEMALE: 0, MALE: 1
import collections
pos = [140,140]
neg = [1500,1500]

val_pos = [12,60]
val_neg = [265,160]

def folds_index(pos, neg, config):
    folds = collections.defaultdict(list)
    data = pd.read_csv("../UMC_w2vec_orig.csv")
    np.random.seed(0)
    clf = 'outcome'
    folds[clf] = collections.defaultdict(list)

    for i in range(config):
        train_rows = []
        test_rows = []
        val_rows = []
        for label, gender in [[1, 0], [1, 1], [0, 0], [0, 1]]:
            sub = data[(data[clf] == label) & (data['GENDER'] == gender)].copy(deep=True)
            if (label == 1):
                no = pos[0] if gender == 0 else pos[1]
                val_no = val_pos[0] if gender == 0 else val_pos[1]

            else:
                no = neg[0] if gender == 0 else neg[1]
                val_no = val_neg[0] if gender == 0 else val_neg[1]


            rows = np.random.choice(sub.index.values, no, replace=False)
            #print("depression {d}, gender {g}, number of examples {e}".format(d=label, g=gender, e=len(rows)))
            train_rows = np.concatenate([np.array(train_rows), rows]).astype(int)
            rows2 = np.random.choice(sub[~sub.index.isin(rows)].index.values, val_no,
                                     replace=False)
            val_rows = np.concatenate([np.array(val_rows), rows2]).astype(int)
            test_rows = np.concatenate(
                [np.array(test_rows),
                 sub[~sub.index.isin(np.concatenate([rows, rows2]))].index.values]).astype(int)


        folds[clf][str(i)].append(train_rows)
        folds[clf][str(i)].append(val_rows)
        folds[clf][str(i)].append(test_rows)

    return folds



svm_params = [{
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__kernel': ['rbf'],

}]

C_values = [0.01, 0.1, 1, 10]
from sklearn.preprocessing import MinMaxScaler
def classifier(X_train, X_val, y_train, y_val, weights, personalized=False):

    X_combined = pd.concat([X_train, pd.DataFrame(X_val)], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    test_fold = [-1] * len(X_train) + [0] * len(X_val)

    # Create the PredefinedSplit
    ps = PredefinedSplit(test_fold=test_fold)

    pipeline = Pipeline(steps=[('standardscaler', StandardScaler()), ("clf", SVC(random_state=0, probability=True, class_weight='balanced'))])
    #MinMaxScaler
    grid = GridSearchCV(pipeline, svm_params, scoring='f1_macro', verbose=1, cv=ps, n_jobs=-1, refit=False)
    
    #roc_auc
    grid.fit(X_combined, y_combined[clf], clf__sample_weight= weights)
    best_params = grid.best_params_
    pipeline.set_params(**best_params)
    model = pipeline.fit(X_train, y_train[clf], clf__sample_weight=weights)
    print(best_params)
    print(grid.best_score_)
    return model

def get_predictions(model, X, y, index):
    y_preds = model.predict(X[index])
    y_prob = model.predict_proba(X[index])[:, 1]

    y_all = pd.DataFrame(y[index][[attr, clf]], columns=[attr, clf]).reset_index(drop=True)

    data = pd.concat([pd.DataFrame(y_preds, columns=['preds']).reset_index(drop=True),
                              pd.DataFrame(y_prob, columns=['probability']).reset_index(drop=True),
                              y_all], axis=1)
    return data

# Commented out IPython magic to ensure Python compatibility.
# %pip install lime

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator
from gensim.models.fasttext import load_facebook_model

def get_vector(text, vectors):
    split_text = text.split(' ')

    vector_list = []
    for i, word in enumerate(split_text) :
        if (word.lower() not in vectors.wv.key_to_index) :
            #
            #  or (not word.isalpha())
            continue
        vector_list.append(vectors.wv[word.lower()])
    if len(vector_list) == 0:
        return None
    return np.average(np.asarray(vector_list), axis=0)

#word_vectors = load_facebook_model('../cc.nl.300.bin')
# class our_extractor(BaseEstimator):
#
#     # model = gensim.models.KeyedVectors.load_word2vec_format(
#     #      "../resources/bio_embedding_extrinsic", binary=True)
#     def __init__(self):
#         print("... called... ")
#
#     def fit(X):
#         return X
#
#     def transform(X):
#         #print("Data in X: \n", X)
#         if X:
#             df = pd.DataFrame([get_vector(row, word_vectors) for row in X])
#             return df
#         else:
#             return pd.DataFrame()
#

def explain(pipeline, index, X, method, clf):
    class_names = ['female', 'male']
    #print("index", index_num)
    #idx = y[1].index[index_num]
    explainer = LimeTextExplainer(class_names=class_names)

    text1 = X['TEXT'].iloc[index]
    print(text1)

    exp1 = explainer.explain_instance(text1
    , pipeline.predict_proba, num_features=50, labels=(1,))

    exp1.save_to_file('lime1_{index}_{method}_{clf}.html'.format(index=index, method=method, clf=clf))

from sklearn.metrics import classification_report, make_scorer, mean_absolute_error, f1_score, confusion_matrix

embedding_length = collections.defaultdict(list)
embedding_length['w2vec'] = 300
embedding_length['biow2vec'] = 100
embedding_length['bert'] = 768
embedding_length['clinical_bert'] = 768

def swap_gender(gender):
    if gender == 0:
        return 1
    else:
        return 0


def create_fold_i(name, type, clf, fold_i):

    folds = collections.defaultdict(list)
    orig_data = pd.read_csv("../UMC_{emb}_orig.csv".format(emb=name)).drop('Unnamed: 0', axis=1, errors='ignore')
    swapped_data = pd.read_csv("../UMC_{emb}_swapped.csv".format(emb=name)).drop('Unnamed: 0', axis=1, errors='ignore')
    neutr_data = pd.read_csv("../UMC_{emb}_neutr.csv".format(emb=name)).drop('Unnamed: 0', axis=1, errors='ignore')

    if (type != "augmented"):
        data = pd.read_csv("../UMC_{emb}_{type}.csv".format(type=type, emb=name)).drop('Unnamed: 0', axis=1, errors='ignore')

    for i in range(len(fold_i[clf])):
        if type == "augmented":
            train_df = pd.concat([orig_data.loc[fold_i[clf][str(i)][0]], swapped_data.loc[fold_i[clf][str(i)][0]]])
            val_df = orig_data.loc[fold_i[clf][str(i)][1]]

        else:
            train_df = data.loc[fold_i[clf][str(i)][0]]
            val_df = data.loc[fold_i[clf][str(i)][1]]

        folds[str(i)].append(train_df)

        test_df = orig_data.loc[fold_i[clf][str(i)][2]]
        augmented_test_df = pd.concat([test_df, swapped_data.loc[fold_i[clf][str(i)][2]]])

        folds[str(i)].append(augmented_test_df)

        test_df = neutr_data.loc[fold_i[clf][str(i)][2]]
        swapped_test = test_df.copy(deep=True)
        swapped_test['GENDER'] = [swap_gender(row['GENDER']) for index, row in swapped_test.iterrows()]

        augmented_blind_df = pd.concat([test_df, swapped_test])

        folds[str(i)].append(augmented_blind_df)
        folds[str(i)].append(val_df)

        augmented_val_df = pd.concat([val_df, swapped_data.loc[fold_i[clf][str(i)][1]]])
        if type == 'neutr':
            swapped_val = val_df.copy(deep=True)
            swapped_val['GENDER'] = [swap_gender(row['GENDER']) for index, row in swapped_val.iterrows()]
            folds[str(i)].append(pd.concat([val_df, swapped_val]))
        else:
            folds[str(i)].append(augmented_val_df)

    return folds



def get_personalized_predictions(model, X, y, index):
    y_preds = model.predict(X[index])
    y_prob = model.predict_proba(X[index])[:, 1]

    y_all = pd.DataFrame(y[index][[attr, clf]], columns=[attr, clf]).reset_index(drop=True)

    data = pd.concat([pd.DataFrame(y_preds, columns=['preds']).reset_index(drop=True),
                              pd.DataFrame(y_prob, columns=['probability']).reset_index(drop=True),
                              y_all], axis=1)
    return data


def get_gender_based_classifier(X_train, X_val, y_train, y_val, weights):
    female_data = []
    male_data = []
    female_y = []
    male_y = []


    for X, y in [[X_train, y_train], [X_val, y_val]]:
        # Extract the feature subsets for each gender
        X_female = X.loc[y[y['GENDER'] == 0].index]
        X_male = X.loc[y[y['GENDER'] == 1].index]

        female_data.append(X_female)
        male_data.append(X_male)

        female_y.append((y[y['GENDER'] == 0]))
        male_y.append(y[y['GENDER'] == 1])

    model_female = classifier(female_data[0], female_data[1], female_y[0], female_y[1], weights)
    model_male = classifier(male_data[0], male_data[1], male_y[0], male_y[1], weights)

    return model_female, model_male

def get_gender_based_predictions(model1, model2, X_test, y_test):
    # Split the examples based on gender
    female_indices = y_test[y_test[attr] == 0].index
    male_indices = y_test[y_test[attr] == 1].index

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

from sklearn.model_selection import StratifiedKFold
def fold_cv(validation_split=0.1, config=5):
    validation_split = 0.1
    n_splits = 10
    #config = 2
    folds = collections.defaultdict(list)
    subset = pd.read_csv("../UMC_w2vec_orig.csv")
    folds[clf] = collections.defaultdict(list)
    i = 0
    for random_state in range(config):
           
        stratify_label = subset[clf].astype(str) + subset[attr].astype(str)
        # Initialize a stratified 10-fold cross-validator
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for trainval_idx, test_idx in cv.split(subset, stratify_label):
            trainval_data = stratify_label[trainval_idx]
            cv_validation = StratifiedKFold(n_splits=int(1/validation_split))
                # Use the first split as the validation set
            for train_index, validation_index in cv_validation.split(np.zeros(len(trainval_data)), trainval_data):
                # Adjust indices to original data size
                train_index = trainval_idx[train_index]
                validation_index = trainval_idx[validation_index]
                break
            # Subsample each class in the training set to have `min_count` instances
            indices_to_keep = np.array([], dtype=int)
            train = subset.iloc[train_index]
            for class_value in np.unique(train[clf]):
                class_indices = train[train[clf] == class_value].index
                _, counts = np.unique(train.loc[class_indices, attr], return_counts=True)
                min_count = np.min(counts)
                
                female_indices = np.intersect1d(np.where((subset[clf] == class_value) & (subset[attr] == 0))[0], train_index)
                male_indices = np.intersect1d(np.where((subset[clf] == class_value) & (subset[attr] == 1))[0], train_index)
                class_subsample_female = np.random.choice(female_indices, min_count, replace=False)
                class_subsample_male = np.random.choice(male_indices, min_count, replace=False)
                
                indices_to_keep = np.concatenate([indices_to_keep, class_subsample_female, class_subsample_male])
                                                                
            #folds[clf][str(i)].append(train_index)
            folds[clf][str(i)].append(indices_to_keep)
            folds[clf][str(i)].append(validation_index)
            folds[clf][str(i)].append(test_idx)
            i += 1
    return folds

def experiment(method_name, folds, config, embedding, clf):
    for fold_i in range(config):
        X, y = [features[[str(a) for a in range(embedding_length[embedding])]].values for features in folds[str(fold_i)]], [y[[clf, 'GENDER']] for y in folds[str(fold_i)]]
        i = int(len(y[1]) / 2)
        if method_name == 'preprocessing':
            train_data = folds[str(fold_i)][0]
            aif360_train = convert_dataset(train_data, clf)
            reweighed_train, labels, weights = pre_processing(aif360_train)
        else: weights = None
        
        if method_name in ['orig', 'neutr',  'augmented', 'preprocessing', 'balanced']:

            model = classifier(pd.DataFrame(X[0]).reset_index(drop=True),
                                    pd.DataFrame(X[3]).reset_index(drop=True),
                                  pd.DataFrame(y[0]).reset_index(drop=True),
                                  pd.DataFrame(y[3]).reset_index(drop=True), weights)


            if method == 'neutr':
                data = get_predictions(model, X, y, 2)


            else:
                data = get_predictions(model, X, y, 1)

            val_data = get_predictions(model, X, y, 4)
        elif method_name == 'gender_specific':
            model1, model2 = get_gender_based_classifier(pd.DataFrame(X[0]).reset_index(drop=True),
                       pd.DataFrame(X[3]).reset_index(drop=True),
                       pd.DataFrame(y[0]).reset_index(drop=True),
                       pd.DataFrame(y[3]).reset_index(drop=True), weights)
            data = get_gender_based_predictions(model1, model2, pd.DataFrame(X[1]).reset_index(drop=True), pd.DataFrame(y[1]).reset_index(drop=True))
            val_data = get_gender_based_predictions(model1, model2, pd.DataFrame(X[4]).reset_index(drop=True), pd.DataFrame(y[4]).reset_index(drop=True))
        
        elif method_name == 'postprocessing':
            test_data = pd.read_csv("../preds/UMC/{measure}/predictions_UMC_orig_fold{fold_i}_{emb}.csv".format(measure=measure, fold_i=fold_i, emb=embedding))
            val_data = pd.read_csv("../preds/UMC/{measure}/val_predictions_UMC_orig_fold{fold_i}_{emb}.csv".format(measure=measure, fold_i=fold_i, emb=embedding))
            aif360_val = convert_dataset(val_data, clf)
            aif360_test = convert_dataset(test_data, clf)
            val_data, data = post_processing(aif360_val, aif360_test, val_data, test_data)


        data.to_csv("../preds/UMC/{measure}/predictions_UMC_{i}_fold{fold_i}_{emb}.csv".format(i=method_name, measure=measure, fold_i=fold_i,emb=embedding))
        val_data.to_csv("../preds/UMC/{measure}/val_predictions_UMC_{i}_fold{fold_i}_{emb}.csv".format(i=method_name, fold_i=fold_i, measure=measure, emb=embedding))
        scores_arr = collections.defaultdict(list)
        print(eval(data, scores_arr, int(len(data)/2) , clf, embedding, fold_i, dataset="UMC"))
 
    return

clf = 'outcome'
#method_arr = ['orig', 'neutr', 'augmented']
#method_arr = ['orig']
embedding_arr = ['w2vec', 'biow2vec', 'bert', 'clinical_bert']
#embedding_arr = ['biow2vec']
#methods = {'orig': ['orig', 'postprocessing', 'gender_specific', 'preprocessing'], 'neutr': ['neutr'], 'augmented': ['augmented']}

methods = {'orig': ['balanced']}
attr = 'GENDER'
clf = 'outcome'
measure = 'f1'
if __name__ == "__main__":
    print("*********** Bias analysis EXPERIMENTS ********")
    config = 20
    for data_key, values in methods.items():
        for method in values:
            folds_ind = fold_cv(config=config)
            #folds_ind = folds_index(pos, neg, config)
            #print("folds_index", (folds_ind[str(0)]))
            for embedding in embedding_arr:
                folds = create_fold_i(embedding, data_key, clf, folds_ind)
                experiment(method, folds, config, embedding, clf)    


