import collections
import gensim.downloader as api

import gensim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModel

from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn.metrics import confusion_matrix

from source.common.embeddings import get_vector, get_word_vector, extract_w2vec
from source.common.text_processing import gender_swapping, clean

svm_params = [{
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['rbf', 'sigmoid', 'linear']
}]
clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
bert_model = 'bert-base-uncased'
w2vec_model = 'word2vec-google-news-300'
def swap_gender(gender):
    if gender == 'F':
        return 'M'
    else:
        return 'F'

def read_vec(filename, method, clf, i):
    X = []
    y = []
    label = clf if clf != 'SCHIZOPHRENIA' else 'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS'

    if method == 'orig':
        X.append(pd.read_csv("../features/mimic_train_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)))
        y.append(pd.read_csv("../data/train_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']])
    elif method == 'augmented':
        X.append(pd.concat([pd.read_csv("../features/mimic_train_{filename}_swapped_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)),
                            pd.read_csv("../features/mimic_train_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i))]))
        y.append(pd.concat([pd.read_csv("../data/train_mimic_swapped_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']],
                            pd.read_csv("../data/train_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']]]))
    else:
        X.append(pd.read_csv("../features/mimic_train_{filename}_{method}_{clf}_{i}.csv".format(filename=filename, method=method, clf=clf, i=i)))
        y.append(pd.read_csv("../data/train_mimic_{method}_{clf}_{i}.csv".format(method=method, clf=clf, i=i))[[label, 'GENDER', 'TEXT']])

    X.append(pd.concat([pd.read_csv(f"../features/mimic_test_{filename}_swapped_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)),
                         pd.read_csv(f"../features/mimic_test_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i))]))

    y.append(pd.concat([pd.read_csv("../data/test_mimic_swapped_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']],
                        pd.read_csv(f"../data/test_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']]]))
    return X, y

def classifier(X, y, label, attr, model, sample_weight=None):
    pipeline = Pipeline(steps=[("preprocesser", StandardScaler()), ("clf", SVC(random_state=0, probability=True))])
    grid = GridSearchCV(
            pipeline, svm_params, scoring='f1_macro', verbose=1,
            cv=3)

    y[0][label] = y[0][label].astype('int')
    y[1][label] = y[1][label].astype('int')

    grid.fit(X[0], y[0][label])
    print(y[0][label].values)
    y_preds = grid.predict(X[1])
    y_preds_prob = grid.predict_proba(X[1])
    if clf == 'clf':
        print("Evaluation of model: \n", classification_report(y[1][label], y_preds))
        print(
            pd.DataFrame(classification_report(y[1][label], y_preds, output_dict=True)).round(2).transpose().to_latex())
    return (y_preds, grid.best_estimator_, y_preds_prob)

def extract_w2vec(type, clinical=True):
    if clinical:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../resources/bio_embedding_extrinsic", binary=True)
    else:
        model = api.load("word2vec-google-news-300")

    name = 'biowordvec' if clinical else "w2vec_news"

    print("reading w2vec file")
    merged = pd.read_csv(f"../mimic_{type}.csv".format(type=type), index_col=None)
    #add clean()
    merged['clean_text'] = [clean(row['TEXT']) for index, row in merged.iterrows()]
    vector_df = pd.DataFrame([get_vector(row, model) for row in merged['clean_text']])
    final_df = pd.concat([merged, vector_df], axis=1)
    print("writing file ..")
    final_df.to_csv('../mimic_{name}_{type}.csv'.format(name=name, type=type), index=False)

def extract_bert(clf, type, clinical=True):
    if clinical:
        (bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(clinical_model_name),
                                            AutoModel.from_pretrained(clinical_model_name)]
    else:
        bert_model = 'bert-base-uncased'
        (bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(bert_model),
                                        AutoModel.from_pretrained(bert_model)]

    name = 'clinical_bert' if clinical else 'bert'
    for split in ['train', 'test']:
        for i in range(10):
            merged = pd.read_csv("../data/" + split + "_mimic_{type}{clf}_{i}.csv".format(type=type, clf=clf, i=i), index_col=None)
            merged['text'] = [clean(row['TEXT']) for index, row in merged.iterrows()]
            df = pd.DataFrame([get_word_vector([bert_model, bert_tokenizer], row, 1) for row in merged['text']])
            df.to_csv('../features/mimic_{split}_{name}_{type}{clf}_{i}.csv'.format(split=split, type=type, clf=clf, name=name, i=i), index=False)

# Step 1: create swapped and neutralized samples.
def create_MIMIC(clf):
    for split in ['train', 'test']:
        for i in range(10):
            merged = pd.read_csv("../data/" + split + "_mimic_{clf}_{i}.csv".format(clf=clf, i=i), index_col=None)
            for val, str_ in [[True, 'neutr'], [False, 'swapped']]:
                tmp = merged.copy(deep=True)
                (tmp['TEXT']) = [gender_swapping(row['TEXT'], row['GENDER'], neutralize=val) for index, row in merged.iterrows()]
                (tmp['GENDER']) = [swap_gender(row['GENDER']) for index, row in merged.iterrows()]
                tmp.to_csv("../data/{split}_mimic_{str}_{clf}_{i}.csv".format(split=split, str=str_, clf=clf, i=i), index=False)

# Step 2: extract features for original, swapped and neutralized test samples.
def extract_all_feat(clf, type):
    print("extraction: ")
    extract_w2vec(clf, type, clinical=True)
    extract_w2vec(clf, type, clinical=False)
    extract_bert(clf, type, clinical=True)
    extract_bert(clf, type, clinical=False)
    return

def name_corr(clf):
    if clf.startswith("DEPRESSION"):
        label = 'DEPRESSION'
    elif clf.startswith("SCHIZOPHRENIA"):
        label = "SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS"
    elif clf.startswith("ALCOHOL"):
        label = "ALCOHOL.ABUSE"
    elif clf.startswith("OTHER.SUBSTANCE.ABUSE"):
        label = "OTHER.SUBSTANCE.ABUSE"
    return label


if __name__ == "__main__":
    print("*********** Bias analysis EXPERIMENTS ********")

    attr = 'GENDER'
    mental_arr = ['SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS_majority', 'ALCOHOL.ABUSE_majority', 'OTHER.SUBSTANCE.ABUSE_majority', 'DEPRESSION_majority']
    embeddings = ['w2vec_news', 'biowordvec', 'bert', 'clinical_bert']
    method_arr = ['orig', 'swapped', 'augmented', 'neutr']

    #### STEP 1: create neutr and swapped datasets.
    #for data in mental_arr:
     #  create_MIMIC(data)
    
    #for clf in mental_arr:
     #   for type in ['neutr_']:
      #      print("Extraction started for {clf} and {type}".format(clf=clf, type=type))
       #     extract_all_feat(clf, type)


    # biowordvec = gensim.models.KeyedVectors.load_word2vec_format(
    #     "../resources/bio_embedding_extrinsic", binary=True)

    df_results = pd.DataFrame()
    #index_arr = False
    for clf_ in ['svm']:
        for clf in mental_arr:
            for range_i in range(10):
                embedding_arr = []
                scores_arr = collections.defaultdict(list)
                for embedding in embeddings:
                    for method in method_arr:
                        embedding_arr.append(embedding)
                        print("model and method", embedding, method)
                        X, y = read_vec(embedding, method, clf, range_i)
                        print("size of the training X: {x} and Y: {yi}".format(x=X[0].shape, yi=(y[0].shape)))
                        print("size of the test X: {x} and Y: {yi}".format(x=X[1].shape, yi=(y[1].shape)))

                        label = clf
                        indices = []

                        y_preds, classify, y_preds_prob = classifier(X, y, label=label, attr=attr, model=clf_)
                        pd.concat([pd.DataFrame(y_preds, columns=['pred']),  pd.DataFrame(y_preds_prob[:, 1], columns=['prob'])], axis=1).to_csv('preds_{embedding}_{clf}_{clf_}.csv'.format(embedding=embedding, clf=clf, clf_=clf_))

                        y_all = pd.DataFrame(y[1][[attr, label]], columns=[attr, label]).reset_index(drop=True)
                        data = (pd.concat(
                            [pd.DataFrame(y_preds, columns=['preds']), pd.DataFrame(y_preds_prob[:, 1], columns=['prob']),
                            y_all], axis=1))
                        i = int(len(y_preds)/2)
                        data_sub = data[0:i].reset_index()
                        data_sub2 = data[i:(i * 2)].reset_index()
                        values = ((data_sub['preds'] != data_sub2['preds']).value_counts())
                        if len(values) == 2:
                            print(values[1])
                            mismatch = values[1]
                        else:
                            print("mismatch yok")
                            mismatch = 0

                        male = data[data[attr] == 'M']
                        female = data[data[attr] == 'F']

                        male_1 = data[(data[attr] == 'M') & (data[label] == 1)]
                        male_0 = data[(data[attr] == 'M') & (data[label] == 0)]

                        male_1 = data[(data[attr] == 'M') & (data[label] == 1)]
                        male_0 = data[(data[attr] == 'M') & (data[label] == 0)]

                        female_1 = data[(data[attr] == 'F') & (data[label] == 1)]
                        female_0 = data[(data[attr] == 'F') & (data[label] == 0)]

                        print(male['prob'])

                        print("male prob: {m}, female prob: {f}".format(m=male['prob'].mean(), f=female['prob'].mean()))
                        scores_arr['male_prob'].append(male['prob'].mean())
                        scores_arr['female_prob'].append(female['prob'].mean())
                        scores_arr['male1_prob'].append(male_1['prob'].mean())
                        scores_arr['female1_prob'].append(female_1['prob'].mean())
                        scores_arr['male0_prob'].append(male_0['prob'].mean())
                        scores_arr['female0_prob'].append(female_0['prob'].mean())

                #
                # df_results['male_prob'] = list(map(lambda x: round(x, 3), scores_arr['male_prob']))
                # df_results['female_prob'] = list(map(lambda x: round(x, 3), scores_arr['female_prob']))
                #
                # df_results['male1_prob'] = list(map(lambda x: round(x, 3), scores_arr['male1_prob']))
                # df_results['female1_prob'] = list(map(lambda x: round(x, 3), scores_arr['female1_prob']))
                #
                # df_results['male0_prob'] = list(map(lambda x: round(x, 3), scores_arr['male0_prob']))
                # df_results['female0_prob'] = list(map(lambda x: round(x, 3), scores_arr['female0_prob']))
                # df_results['mismatch'] = scores_arr['mismatch']
                # df_results['total_number'] = scores_arr['total_number']


                        for group, t in [[male, 'M'], [female, 'F']]:
                            tn, fp, fn, tp  = (confusion_matrix(group[label], group['preds']).ravel())
                            print("{group}: False positive rate: {score}".format(group=t, score=fp/ (fp + tn)))
                            print("{group}: True positive rate {score}".format(group=t, score=tp/ (tp+fn)))
                            print("{group}: False negative rate {score}".format(group=t, score=fn / (fn + tp)))
                            scores_arr['FNR-'+t].append((fn / (fn + tp)))
                            scores_arr['FPR-'+t].append(fp / (fp + tn))
                            scores_arr['TPR-' + t].append(tp / (tp + fn))
                        scores_arr['F1'].append(round(f1_score(y[1][label], y_preds, average='macro'), 2))
                        scores_arr['mismatch'].append(mismatch)
                        scores_arr['total_num'].append(i)

                df_results['embedding'] = embedding_arr
                for mi in ['FNR', 'FPR', 'TPR']:
                    df_results[mi + '-F'] = list(map(lambda x: round(x, 2), scores_arr[mi+'-F']))
                    df_results[mi + '-M'] = list(map(lambda x: round(x, 2), scores_arr[mi+'-M']))
                    df_results[mi + '-R'] = [df_results[mi+'-F'][i]/ df_results[mi+'-M'][i] for i in range(len(df_results[mi+'-F']))]
                    df_results[mi + 'R'] = list(map(lambda x: round(x, 2) if x <1 else round(1/x, 2), df_results[mi+'-R']))
                df_results['F1'] = scores_arr['F1']
                df_results['mismatch'] = scores_arr['mismatch']
                df_results['total_num'] = scores_arr['total_num']
                #df_results['mismatch_ratio'] = list(map(lambda x:round( x['mismatch']/x['total_num'], 2), scores_arr))

                print(df_results.shape)
                print(df_results)
                df_results.index= len(embeddings) * method_arr
                df_results.to_csv("../t-test/mimic_experiments_{clf}_{clf_}_{i}.csv".format(clf=clf, clf_=clf_, i=range_i))
