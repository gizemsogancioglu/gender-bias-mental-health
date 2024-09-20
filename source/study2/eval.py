import collections
import pandas as pd
from scipy.stats import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def eval(data, scores_arr, i, clf, embedding, fold_i, dataset):
    scores_arr = compute_mismatch(data, i, scores_arr, dataset)
    scores_arr = compute_predictive_parity(data, scores_arr, i, clf, dataset)
    scores_arr['total_num'].append(i)
    scores_arr['embedding'].append(embedding)
    scores_arr['fold_num'].append(fold_i)

    return scores_arr

def compute_predictive_parity(data, scores_arr, i, clf, dataset):
    data_test = data[0:i].reset_index().copy(deep=True)
    male = data_test[data_test[attr] == gender_dict[dataset][1]]
    female = data_test[data_test[attr] == gender_dict[dataset][0]]
    
    data_fm = data_test[((data_test[attr] == gender_dict[dataset][0]) & (data_test[clf] == 1)) |
                        ((data_test[attr] == gender_dict[dataset][1]) & (data_test[clf] == 0))]

    data_mf = data_test[((data_test[attr] == gender_dict[dataset][0]) & (data_test[clf] == 0)) |
                        ((data_test[attr] == gender_dict[dataset][1]) & (data_test[clf] == 1))]

    # positive female and negative male
    # positive male and negative female
    s = ''
    scores_arr[s + "AUC-fm"].append(roc_auc_score(data_fm[clf], data_fm['probability']))
    scores_arr[s + "AUC-mf"].append(roc_auc_score(data_mf[clf], data_mf['probability']))

    for group, t in [[male, 'M'], [female, 'F']]:
        tn, fp, fn, tp = (confusion_matrix(group[clf], group['preds']).ravel())
        scores_arr[s + 'FNR-' + t].append((fn / (fn + tp)))
        scores_arr[s + 'FPR-' + t].append(fp / (fp + tn))
        scores_arr[s + 'TPR-' + t].append(tp / (tp + fn))
        scores_arr[s + 'precision-' + t].append(tp / (tp + fp))

        scores_arr[s + 'F1-' + t].append(f1_score(group[clf], group['preds'], average='macro'))
        scores_arr[s + 'F1-pos-' + t].append(f1_score(group[clf], group['preds'], pos_label=1))
        scores_arr[s + 'F1-neg-' + t].append(f1_score(group[clf], group['preds'], pos_label=0))

        scores_arr[s + 'AUC-' + t].append(roc_auc_score(group[clf], group['probability']))
        if t == 'M':
            gender = gender_dict[dataset][1]
        else: 
            gender = gender_dict[dataset][0]
        scores_arr[s + 'Pos-' + t].append(len(data_test[(data_test[attr] == gender) & (data_test[clf] == 1)]))
        scores_arr[s + 'Neg-' + t].append(len(data_test[(data_test[attr] == gender) & (data_test[clf] == 0)]))
        

    tn, fp, fn, tp = (confusion_matrix(data_test[clf], data_test['preds']).ravel())

    scores_arr[s + 'TPR'].append(tp/ (tp+fn))
    scores_arr[s + 'precision'].append(tp/ (tp+fp))

    scores_arr[s + 'F1'].append(f1_score(data_test[clf], data_test['preds'], average='macro'))
    scores_arr[s + 'AUC'].append(roc_auc_score(data_test[clf], data_test['probability']))

    return scores_arr


def compute_mismatch(data, i, scores_arr, dataset):
    data_sub = data[0:i].reset_index()
    data_sub2 = data[i:(i * 2)].reset_index()
    values = ((data_sub['preds'] != data_sub2['preds']).value_counts())

    df1 = data_sub[(data_sub['preds'] != data_sub2['preds'])]
    df2 = data_sub2[(data_sub['preds'] != data_sub2['preds'])]
    bias_direction = 'F'
    sum1_F = (df1[df1['GENDER'] == gender_dict[dataset][0]]['preds'].sum()) + (df2[df2['GENDER'] == gender_dict[dataset][0]]['preds'].sum())
    sum1_M = (df1[df1['GENDER'] == gender_dict[dataset][1]]['preds'].sum()) + (df2[df2['GENDER'] == gender_dict[dataset][1]]['preds'].sum())
    if sum1_M > sum1_F:
        bias_direction = 'M'

    if len(values) == 2:
        mismatch = values[1]
    else:
        mismatch = 0

    scores_arr['mismatch'].append(mismatch)
    scores_arr['mismatch_direction'].append(bias_direction)
    scores_arr['mismatch_F'].append(sum1_F)
    scores_arr['mismatch_M'].append(sum1_M)
    scores_arr['mismatch_ratio'].append((mismatch/i))

    return scores_arr

def update(val):
    print(val)
    if val > 1:
        gender = 'F'
        val = round(1 / val, 2)

    else:
        gender = 'M'

    value = str(val) + "_" + gender


    return value
import numpy as np
def measure_ratio(val1, val2):
    if max([val1, val2]) != 0:
        return round(min([val1, val2]) / max([val1, val2]), 3)
    else:
        return np.nan
from scipy.stats import ttest_rel
def save_res(method, measure_file, dataset):
    for split in ['val_', '']:
        data = pd.read_csv(f"../results/{dataset}/{measure_file}/{split}experiments_{dataset}_{method}.csv")
        data = data.groupby(['embedding', 'fold_num']).tail(1)
        orig_data = pd.read_csv(f"../results/{dataset}/{measure_file}/{split}experiments_{dataset}_orig.csv")
        columns = ['TPR-F', 'TPR-M',
                       'AUC-F', 'AUC-M',
                       'AUC-fm', 'AUC-mf',
                       'FPR-F', 'FPR-M',
                       'F1-F', 'F1-M',
                       'F1-pos-F', 'F1-pos-M',
                       'F1-neg-F', 'F1-neg-M',
                       'AUC',
                       'F1',
                       'mismatch_ratio',
                       'embedding'
                       ]


        for group in [data, orig_data]:
            for measure in ['TPR', 'FPR', 'AUC', 'F1', 'F1-pos', 'F1-neg']:
                group[f'{measure}R'] = [measure_ratio(row[f'{measure}-F'], row[f'{measure}-M']) for index, row in group.iterrows()]
    
                measure = 'AUC'
            group[f'{measure}minmax'] = [((min([row[f'{measure}-F'], row[f'{measure}-M'],

                                                   row[f'{measure}-fm'], row[f'{measure}-mf']]) /
                                               max(row[f'{measure}-F'], row[f'{measure}-M'],
                                                   row[f'{measure}-fm'], row[f'{measure}-mf'])))  for
                                 index, row in group.iterrows()]
        #data[columns].to_csv("../results/{dataset}/{measure}/{split}experiments_{dataset}_{i}_extended.csv".format(dataset=dataset, split=split, measure=measure_file, i=method))
        #subset= subset.replace(0, np.nan, inplace=True)
        #subset = data.groupby('embedding').mean()
        subset = data.groupby('embedding').mean()
        subset2 = orig_data.groupby('embedding').mean()


        for group in [subset, subset2]:
            for measure in ['TPR', 'FPR', 'F1', 'F1-pos', 'F1-neg']:
                group[f'avg_{measure}R'] = [measure_ratio(row[f'{measure}-F'], row[f'{measure}-M']) for index, row in group.iterrows()]


        for measure in ['TPR', 'FPR', 'AUC', 'F1', 'F1-pos', 'F1-neg']:
            subset[f'{measure}R'] = subset[f'{measure}R'].astype(str)
            for val in subset.index:
                group1 = data[data['embedding'] == val][measure + '-F'].values
                group2 = data[data['embedding'] == val][measure + '-M'].values
                __, pval = stats.ttest_ind(group1, group2, equal_var=False)
                
                #subset.loc[val][f"{measure}-pval"] = round(pval, 3)
                if pval < 0.05:
                    subset.loc[val, f'{measure}R'] += '*'

        if (method != 'orig'): 
            for measure in ['TPRR', 'FPRR', 'F1R', 'F1', 'mismatch_ratio', 'TPR-F', 'TPR-M', 'FPR-F', 'FPR-M', 'F1-F', 'F1-M']:
                subset[f'{measure}'] = subset[f'{measure}'].astype(str)
                for val in subset.index:
                    group_orig = orig_data[orig_data['embedding'] == val][measure].values
                    new_group = data[data['embedding'] == val][measure].values
                    __, pval = ttest_rel(group_orig, new_group)
                    if pval < 0.05:
                        subset.loc[val, f'{measure}'] += '+'
        
        final_columns = [ 'mismatch_ratio','avg_TPRR', 'avg_FPRR', 'avg_F1R','avg_F1-posR', 'avg_F1-negR', 'TPRR', 'FPRR', 'F1R','TPR-F', 'TPR-M', 'FPR-F', 'FPR-M', 'F1-F', 'F1-M', 'F1-posR', 'F1-negR', 'AUCR', 'AUCminmax', 'F1', 'AUC']
    
        subset[final_columns].to_csv("../results/{dataset}/{measure}/{split}experiments_{dataset}_{i}_summary.csv".format(dataset=dataset, measure=measure_file, i=method, split=split))

    return


#method_arr = ['orig', 'neutr', 'augmented', 'preprocessing', 'gender_specific', 'postprocessing']
method_arr = ['orig', 'neutr', 'augmented']
attr = 'GENDER'
dataset = 'MIMIC'
#dataset = 'UMC'
gender_dict = {"UMC": [0, 1], "MIMIC": ['F', 'M']}
#MIMIC: F, M
if __name__ == "__main__":
    measure = 'f1'
    if dataset == 'MIMIC':
        clf = 'DEPRESSION_majority'
        #embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']
        embeddings = [ 'w2vec_news', 'biowordvec']
        config = 10
    else:
        clf = 'outcome'
        embeddings = ['w2vec', 'biow2vec', 'bert', 'clinical_bert']
        #embeddings = ['biow2vec']
        config = 20

    print("*********** Bias analysis EXPERIMENTS ********")
    for method in method_arr:
        scores_arr = collections.defaultdict(list)
        val_scores_arr = collections.defaultdict(list)
        method_name = method
        for embedding in embeddings:
            for fold_i in range(config):
                test_preds =  pd.read_csv(
                        "../preds/{dataset}/{measure}/predictions_{dataset}_{i}_fold{fold_i}_{emb}.csv".format(dataset=dataset, i=method, fold_i=fold_i,
                                                                                         emb=embedding, measure=measure))

                val_preds = pd.read_csv(
                    "../preds/{dataset}/{measure}/val_predictions_{dataset}_{i}_fold{fold_i}_{emb}.csv".format(dataset=dataset, i=method, fold_i=fold_i,
                                                                                 emb=embedding, measure=measure))
                #print(test_preds)
            
                scores_arr = eval(test_preds, scores_arr, int(len(test_preds)/2), clf, embedding, fold_i, dataset)
                val_scores_arr = eval(val_preds, val_scores_arr, int(len(val_preds) / 2), clf, embedding, fold_i, dataset)

        pd.DataFrame(val_scores_arr).to_csv("../results/{dataset}/{measure}/val_experiments_{dataset}_{i}.csv".format(dataset=dataset, measure=measure, i=method))
        pd.DataFrame(scores_arr).to_csv("../results/{dataset}/{measure}/experiments_{dataset}_{i}.csv".format(dataset=dataset, measure=measure, i=method))


    for method in method_arr:
        save_res(method, measure, dataset)
