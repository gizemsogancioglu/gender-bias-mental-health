import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
import collections
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from eval import save_res, evaluate

def compare_df_values_multi(df_list, df_names, column_name, embedding, fold):
    values = [df[df['embedding'] == embedding].mean()[column_name] for df in df_list]
    max_value = max(values)
    max_index = values.index(max_value)
    print(f"Row : {df_names[max_index]} has the higher value of {max_value}")

    __, p_value = ttest_rel(df_list[0][column_name], df_list[max_index][column_name])
    if p_value > 0.05:
        #print("not significant")
        return "orig"
    else:
        #print("significant")
        return df_names[max_index]

    return  df_names[max_index]

def participatory_method(config, dataset, measure):
    method_arr = ['orig', 'neutr', 'augmented', 'gender_specific']
    scores_arr = collections.defaultdict(list)
    for embedding in embeddings:
        df_list = []
        for method in method_arr:
            data = pd.read_csv(f'../results/val_experiments_{dataset}_{method}.csv')
            df_list.append(data)
        for fold in range(config):
            preds = pd.read_csv(
                f"../preds/predictions_{dataset}_orig_fold{fold}_{embedding}.csv")
            predictions = pd.Series(index=preds.index, dtype='float64')
            predictions_prob = pd.Series(index=preds.index, dtype='float64')

            for gender in ['F', 'M']:
                print(embedding)
                best_model = compare_df_values_multi(df_list, method_arr, measure + '-' + gender, embedding, fold)
                if fold == 0: 
                    print("best model {best} for {emb} and {gender} for fold {fold}".format(best=best_model, emb=embedding,
                                                                                        gender=gender, fold=fold))

                best_preds = pd.read_csv(
                    f"../preds/predictions_{dataset}_{best_model}_fold{fold}_{embedding}.csv")
                
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
                f"../preds/predictions_{dataset}_participatory_fold{fold}_{embedding}.csv")
if __name__ == "__main__":
    dataset = 'MIMIC'
    #dataset = 'UMC'
    gender_dict = {"UMC": [0, 1], "MIMIC": ['F','M']}
    attr = 'GENDER'

    if dataset == 'MIMIC':
        config = 5
        clf = 'DEPRESSION_majority'
        label  = 'DEPRESSION_majority'
        #embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']
        embeddings = ['biowordvec']
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
                        f"../preds/predictions_{dataset}_{method}_fold{fold_i}_{embedding}.csv")

                val_preds =  pd.read_csv(
                    f"../preds/predictions_{dataset}_{method}_fold{fold_i}_{embedding}.csv")

                scores_arr = evaluate(test_preds, scores_arr, int(len(test_preds)/2) , clf, embedding, fold_i, dataset)
                val_scores_arr = evaluate(val_preds, val_scores_arr, int(len(val_preds) / 2), clf, embedding, fold_i, dataset)

        pd.DataFrame(val_scores_arr).to_csv(f"../results/val_experiments_{dataset}_{method}.csv")

        pd.DataFrame(scores_arr).to_csv(f"../results/experiments_{dataset}_{method}.csv")
    
    save_res('participatory', measure.lower(), dataset)

