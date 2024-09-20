import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
import collections
import scipy.stats as stats
from sklearn.metrics import recall_score, f1_score, confusion_matrix, \
    roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from eval import eval, save_res

from scipy.stats import shapiro

def bootstrap_test_model_comparison(model1_scores, model2_scores, n_bootstrap=10000):
    observed_diff = np.mean(model2_scores) - np.mean(model1_scores)
    combined_scores = np.concatenate([model1_scores, model2_scores])
    count_extreme = 0
                    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(combined_scores, size=len(combined_scores), replace=True)
        bootstrap_diff = np.mean(bootstrap_sample[len(model1_scores):]) - np.mean(bootstrap_sample[:len(model1_scores)])
                                                    
        if bootstrap_diff >= observed_diff:
            count_extreme += 1
                                                                                        
    p_value = count_extreme / n_bootstrap
    return p_value

def compare_df_values_multi(df_list, df_names, column_name, embedding, fold):
    # Assuming all dataframes are of the same length and have aligned indices
        # Extract the value from each dataframe for the current row and column
    values = [df[df['embedding'] == embedding].mean()[column_name] for df in df_list]
    max_value = max(values)
    max_index = values.index(max_value)
    #pint(f"Row : {df_names[max_index]} has the higher value of {max_value}")

    #t_stat, p_value = stats.ttest_rel(df_list[0][column_name], df_list[1][column_name])
    #if df_names[max_index] == 'neutr':
        #pred_orig = pd.read_csv("../preds/predictions_MIMIC_DEPRESSION_majority_orig_fold{num}_{emb}.csv".format(num=fold,emb=embedding))
        #pred_neutr = pd.read_csv("../preds/predictions_MIMIC_DEPRESSION_majority_neutr_fold{num}_{emb}.csv".format(num=fold,emb=embedding))
        #p_value = bootstrap_test_model_comparison(pred_orig, pred_neutr, n_bootstrap=100)
        #p_value = bootstrap_test_model_comparison(df_list[0][column_name], df_list[1][column_name], n_bootstrap=10000)
    
    # Calculate the differences
    differences = df_list[0][column_name] - df_list[1][column_name]

    # Perform the Shapiro-Wilk test for normality
    stat, p = shapiro(differences)

    # Interpret the results
    #alpha = 0.05
    #if p > alpha:
     #   print('The difference between the models seems to be normally distributed (fail to reject H0)')
    #else:
     #   print('The difference between the models does not appear to be normally distributed (reject H0)')
   
    __, p_value = ttest_rel(df_list[0][column_name], df_list[max_index][column_name])
    #p_value, tstat = two_sample_paired_sign_test(df_list[0][column_name], df_list[max_index][column_name])
    #print("p-val", p_value)
    #p_value = p_value/2
    #print("p-val from paired t-test: ", p_value, tstat)
    if p_value > 0.05:
        #print("not significant")
        return "orig"
    else:
        #print("WOW")
        return df_names[max_index]

    return  df_names[max_index]

def two_sample_paired_sign_test(S1, S2, alternative='greater'):
        
    S1 = np.array(S1)
    S2 = np.array(S2)
    m = len(S1)
    assert m > 0
    assert len(S2) == m
    assert alternative in ('greater', 'less')
    test_stat = np.greater_equal(S1 - S2, 0.0).sum()
    p_value = stats.binom_test(test_stat, n=len(S1), p=0.5, alternative=alternative)

    return p_value, test_stat

def participatory_method(config, dataset, measure):
    #method_arr = ['orig', 'neutr', 'augmented', 'gender_specific']
    method_arr = ['orig', 'neutr', 'augmented', 'gender_specific']
    scores_arr = collections.defaultdict(list)
    for embedding in embeddings:
        df_list = []
        for method in method_arr:
            data = pd.read_csv('../results/{dataset}/{measure}/val_experiments_{dataset}_{m}.csv'.format(dataset=dataset, m=method, measure=measure.lower()))
            #data = data[data['embedding'] == embedding].mean()
            df_list.append(data)
        #print(data)
        # df_list.append(data[(data['embedding'] == embedding) & (data['fold_num'] == fold)])
        for fold in range(config):
            preds = pd.read_csv(
                "../preds/{dataset}/{measure}/predictions_{dataset}_orig_fold{num}_{emb}.csv".format(dataset=dataset, num=fold,measure=measure.lower(),
                                                                                                    emb=embedding))
            predictions = pd.Series(index=preds.index, dtype='float64')
            predictions_prob = pd.Series(index=preds.index, dtype='float64')

            for gender in ['F', 'M']:
                best_model = compare_df_values_multi(df_list, method_arr, measure + '-' + gender, embedding, fold)
                if fold == 0: 
                    print("best model {best} for {emb} and {gender} for fold {fold}".format(best=best_model, emb=embedding,
                                                                                        gender=gender, fold=fold))

                best_preds = pd.read_csv(
                    "../preds/{dataset}/{measure}/predictions_{dataset}_{best}_fold{num}_{emb}.csv".format(dataset=dataset, clf=clf, best=best_model,
                                                                                            num=fold, measure=measure.lower(), emb=embedding))
                
                if dataset == 'UMC': 
                    gender = 0 if gender == 'F' else 1
                
                indices = best_preds[best_preds[attr] == gender].index

                new_predictions = best_preds.loc[indices]['preds']
                new_probs = best_preds.loc[indices]['probability']
                predictions.loc[indices] = new_predictions
                predictions_prob.loc[indices] = new_probs
        

            data = pd.concat([pd.DataFrame(predictions, columns=['preds']).reset_index(drop=True),
                              pd.DataFrame(predictions_prob, columns=['probability']).reset_index(drop=True),
                              best_preds[[clf, 'GENDER']].reset_index(drop=True)], axis=1)

            data.to_csv(
                "../preds/{dataset}/{measure}/predictions_{dataset}_{best}_fold{num}_{emb}.csv".format(dataset=dataset, clf=clf, best='participatory',
                                                                                        num=fold, emb=embedding, measure=measure.lower()))
if __name__ == "__main__":

    #dataset = 'MIMIC'
    dataset = 'UMC'

    gender_dict = {"UMC": [0, 1], "MIMIC": ['F','M']}
    attr = 'GENDER'

    if dataset == 'MIMIC':
        config = 50
        clf = 'DEPRESSION_majority'
        label  = 'DEPRESSION_majority'
        embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']
        #embeddings = ['w2vec_news']
        measure = 'F1'

    else: 
        config = 20
        clf = 'outcome'
        label= 'outcome'
        #embeddings = ['biow2vec']
        embeddings = ['w2vec', 'biow2vec', 'bert', 'clinical_bert']
        measure = 'F1'
    participatory_method(config, dataset, measure)

    print("CLF", clf)
    print("*********** Bias analysis EXPERIMENTS ********")
    for method in ['participatory']:
        scores_arr = collections.defaultdict(list)
        val_scores_arr = collections.defaultdict(list)
        method_name = method
        for embedding in embeddings:
            for fold_i in range(config):
                test_preds =  pd.read_csv(
                        "../preds/{dataset}/{measure}/predictions_{dataset}_{i}_fold{fold_i}_{emb}.csv".format(dataset=dataset, i=method, fold_i=fold_i,
                                                                                         emb=embedding, measure=measure.lower()))

                val_preds =  pd.read_csv(
                         "../preds/{dataset}/{measure}/predictions_{dataset}_{i}_fold{fold_i}_{emb}.csv".format(dataset=dataset, i=method, fold_i=fold_i, 
                             emb=embedding, measure=measure.lower()))



                scores_arr = eval(test_preds, scores_arr, int(len(test_preds)/2) , clf, embedding, fold_i, dataset)
                val_scores_arr = eval(val_preds, val_scores_arr, int(len(val_preds) / 2), clf, embedding, fold_i, dataset)

        pd.DataFrame(val_scores_arr).to_csv("../results/{dataset}/{measure}/val_experiments_{dataset}_{i}.csv".format(dataset=dataset, measure=measure.lower(), i=method))

        pd.DataFrame(scores_arr).to_csv("../results/{dataset}/{measure}/experiments_{dataset}_{i}.csv".format(dataset=dataset, i=method, measure=measure.lower()))
    
    save_res('participatory', measure.lower(), dataset)

