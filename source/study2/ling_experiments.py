import collections
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, make_scorer, mean_absolute_error, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import bias_analysis as bias
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gensim.downloader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
import gensim.downloader as api
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd
from xgboost import XGBClassifier

svr_params = {'clf__kernel': ('linear', 'rbf', 'poly'), 'clf__C': [0.0000001, 0.000001, 0.00001, 0.0001,
                                                         0.001, 0.01, 0.1, 1, 10, 100, 1000,
                                                         0.0000003, 0.000003, 0.00003, 0.0003,
                                                         0.003, 0.03, 0.3, 3, 30, 300, 3000]}

mlp_params = [{
    'clf__alpha': np.logspace(-1, 1, 5),
    'clf__hidden_layer_sizes': [(100,), (150, 150)]
#(200, 200), (300, 300)
}]

rf_params = [{
    'max_depth': [1, 3, 5, 10, 20],
   # 'oob_score': [True],
}]

ridge_params = [{
    'alpha': [0.1, 0.01, 1, 10, 100, 1000],
}]

svm_params = [{
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['rbf', 'sigmoid', 'linear']
}]

clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
bert_model = 'bert-base-uncased'
w2vec_model = 'word2vec-google-news-300'
pubmed_bert = 'BiomedNLP-PubMedBERT-base-uncased-abstract'
import re
def gender_swapping(text, gender, neutralize=False):

    if neutralize:
        text = re.sub(r"\bher\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+F\b", "Sex: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwoman\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfemale\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bF\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhis\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+M\b", "Sex: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bman\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmale\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bM\b", "", text, flags=re.IGNORECASE)

    if gender == 'F':
        text = re.sub(r"\bher\b", "his", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+F\b", "Sex: M", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bshe\b", "he", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bwoman\b", "man", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfemale\b", "male", text, flags=re.IGNORECASE)
        text = re.sub(r"\bF\b", "M", text, flags=re.IGNORECASE)

        gender = 'M'
    else:
        text = re.sub(r"\bhis\b", "her", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "her", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+M\b", "Sex: F", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\b", "she", text, flags=re.IGNORECASE)
        text = re.sub(r"\bman\b", "woman", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmale\b", "female", text, flags=re.IGNORECASE)
        text = re.sub(r"\bM\b", "F", text, flags=re.IGNORECASE)

        gender = 'F'
    return text

def resampling(data, label, attr='Gender'):
    ros = RandomOverSampler(random_state=42)
    dfs = []
    dfb_1 = data[(data[label] == 0)]
    dfb_2 = data[(data[label] == 1)]
    for df in [dfb_1, dfb_2]:
        dfb_res, attr_res = ros.fit_resample(df, df[attr].tolist())
        dfs.append(pd.DataFrame(dfb_res, columns=df.keys().tolist()))
    df_res = pd.concat([dfi for dfi in dfs], ignore_index=True)
    return df_res


def get_word_vector(model, text, pos, oneWord=False):
    clinical_model = model[0]
    tokenizer = model[1]
    text_arr = text.split("\n\n")
    vector_list = []
    for text in text_arr:
        text = text.replace("\n", " ")
        split_arr = text.split(" ")
        n = int(round(len(split_arr)/512, 0))
        n = (n+15)
        text_sub_arr = np.array_split(split_arr, n)
        for text_sub in text_sub_arr:
            inputs = tokenizer(" ".join(text_sub), return_tensors="pt", padding=True)
            outputs = clinical_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            if oneWord == True:
                return last_hidden_states[0][pos].detach().numpy()
            else:
                i = 1
                while i < len(last_hidden_states[0]):
                    vector_list.append(last_hidden_states[0][i].detach().numpy())
                    i += 1
    return np.average(np.asarray(vector_list), axis=0)


def get_vector(text, vectors):
    split_text = text.split(' ')
    vector_list = []
    for i, word in enumerate(split_text) :
        if (word.lower() not in vectors) :
            #
            #  or (not word.isalpha())
            continue
        vector_list.append(vectors[word.lower()])
    if len(vector_list) == 0:
        return None
    return np.average(np.asarray(vector_list), axis=0)

# def extract_pubmed_bert():
#     bert_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
#     (bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(bert_model), AutoModel.from_pretrained(bert_model)]


def read_vec(filename, method, clf, i):
    X = []
    y = []
    label = clf if clf != 'SCHIZOPHRENIA' else 'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS'
    if label in  ['DEPRESSION_clean', 'DEPRESSION_v0']:
        label = 'DEPRESSION'
    if method == 'orig':
        X.append(pd.read_csv("../LING/features/mimic_train_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)))
        y.append(pd.read_csv("../LING/data/train_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']])
    elif method == 'augmented':
        X.append(pd.concat([pd.read_csv("../LING/features/mimic_train_{filename}_swapped_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)),
                            pd.read_csv("../LING/features/mimic_train_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i))]))
        y.append(pd.concat([pd.read_csv("../LING/data/train_mimic_swapped_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']],
                            pd.read_csv("../LING/data/train_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']]]))
    else:
        X.append(pd.read_csv("../LING/features/mimic_train_{filename}_{method}_{clf}_{i}.csv".format(filename=filename, method=method, clf=clf, i=i)))
        y.append(pd.read_csv("../LING/data/train_mimic_{method}_{clf}_{i}.csv".format(method=method, clf=clf, i=i))[[label, 'GENDER', 'TEXT']])

    X.append(pd.concat([pd.read_csv(f"../LING/features/mimic_test_{filename}_swapped_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i)),
                         pd.read_csv(f"../LING/features/mimic_test_{filename}_{clf}_{i}.csv".format(filename=filename, clf=clf, i=i))]))

    y.append(pd.concat([pd.read_csv("../LING/data/test_mimic_swapped_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']],
                        pd.read_csv(f"../LING/data/test_mimic_{clf}_{i}.csv".format(clf=clf, i=i))[[label, 'GENDER', 'TEXT']]]))

    return X, y

class our_extractor(BaseEstimator):
    model = api.load("word2vec-google-news-300")
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #      "../resources/bio_embedding_extrinsic", binary=True)
    def __init__(self):
        print("... called... ")

    def fit(X):
        return X

    def transform(X):
        #print("Data in X: \n", X)
        if X:
            df = pd.DataFrame([get_vector(row, model) for row in X])
            return df
        else:
            return pd.DataFrame()


def swap_gender(gender):
    if gender == 'F':
        return 'M'
    else:
        return 'F'

def read_MIMIC():
    for split in ['train', 'test']:
        merged = pd.read_csv("../LING/" + split + "_mimic.csv", index_col=None)
        (merged['TEXT']) = [gender_swapping(row['TEXT'], row['GENDER'], neutralize=True) for index, row in merged.iterrows()]
        (merged['GENDER']) = [swap_gender(row['GENDER']) for index, row in merged.iterrows()]

        merged.to_csv("../LING/" + split + "_mimic_neutr.csv", index=False)

def classifier(X, y, label, attr, model, sample_weight=None):
    if model == 'svm':
        pipeline = Pipeline(steps=[("preprocesser", StandardScaler()), ("clf", SVC(random_state=0, probability=True))])
        grid = GridSearchCV(
                pipeline, svm_params, scoring='f1_macro', verbose=1,
                cv=3)
    elif model == 'rf':
        grid = GridSearchCV(
                RandomForestClassifier(random_state=0, class_weight='balanced'), rf_params, scoring='f1_macro',
                verbose=1,
                cv=3)
    elif model == 'xgboost':
        grid = GridSearchCV(
                XGBClassifier(random_state=0), rf_params, scoring='f1_macro',
                verbose=1,
                cv=3)
    elif model == 'mlp':
        pipeline = Pipeline(steps=[("preprocesser", StandardScaler()), ("clf",  MLPClassifier(random_state=0))])

        grid = GridSearchCV(
                pipeline, mlp_params, scoring='f1_macro',
                verbose=1,
                cv=3)
    # sc = StandardScaler()
    # X[0] = pd.DataFrame(sc.fit_transform(X[0]))
    # X[1] = pd.DataFrame(sc.transform(X[1]))
    print("clf: ", label, y[0].columns)
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
        (eod, sp) = bias.compute_bias_scores(y[1][label], y_preds,
                                             y[1][attr])
        print("EOD {eod} score and SP {SP} score: ".format(eod=eod, SP=sp))


       # print("CCC score: ", round(ccc_score(pd.DataFrame(y[1]).reset_index(drop=True)['PHQ_8Total'], pd.DataFrame(y_preds)[8]), 2))
      #  y_preds_test = grid.predict(X[2])
    return (y_preds, grid.best_estimator_, y_preds_prob)


from sklearn.metrics import confusion_matrix
model = gensim.downloader.load('word2vec-google-news-300')

# (clinical_tokenizer, clinical_model) = [AutoTokenizer.from_pretrained(clinical_model_name),
#                                         AutoModel.from_pretrained(clinical_model_name)]


# def f(text):
#     word_vec = get_word_vector([clinical_model, clinical_tokenizer], text, 1)
#     return classify.predict_prob(word_vec)
#
# import shap
# def run_SHAP(model, X):
#     # build an explainer using a token masker
#     explainer = shap.Explainer(f, clinical_tokenizer)
#     # explain the model's predictions on IMDB reviews
#     #imdb_train = nlp.load_dataset("imdb")["train"]
#     shap_values = explainer(X[1][:10], fixed_context=1)
#     print(shap_values)

# .... benchmark methods ....
def resampling(data, label, attr='gender'):
    ros = RandomOverSampler(random_state=42)
    dfs = []
    bins = np.arange(0, 1.1, 0.1)
    for b in bins[1:]:
        dfb = data[(data[label] < b) & (data[label] >= b - 0.1)]
        if len(dfb[attr].unique()) == 1:
            continue
        dfb_res, attr_res = ros.fit_resample(dfb, dfb[attr].tolist())
        dfs.append(pd.DataFrame(dfb_res, columns=dfb.keys().tolist()))
    df_res = pd.concat([dfi for dfi in dfs], ignore_index=True)
    return df_res


def qualitative_analysis(label):
    i = int(len(y_preds)/2)
    data_sub = data[0:i].reset_index()
    data_sub2 = data[i:(i*2)].reset_index()
    print((data_sub['preds']  == data_sub2['preds']).value_counts())
    index_arr = (data_sub.index[data_sub['preds'] != data_sub2['preds']].tolist())
    #print(data_sub2.iloc[index_arr]['prob'])
    print(data_sub2.iloc[index_arr]['preds'])
    #print(data_sub2.iloc[index_arr]['GENDER'])
    print(y[1].iloc[index_arr][label])
    return index_arr
    #print(data_sub.preds[data_sub['preds'] != data_sub2['preds']].tolist())

    #print(data_sub.loc[138])
    #print(data_sub.loc[138]['TEXT'])

def extract_w2vec(clf, type, clinical=True):
    if clinical:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../resources/bio_embedding_extrinsic", binary=True)
    else:
        model = api.load("word2vec-google-news-300")

    name = 'biowordvec' if clinical else "w2vec_news"
    for split in ['train', 'test']:
        for i in range(10):
            print("reading file ../LING/data/{split}_mimic_{type}{clf}_{i}.csv".format(split=split, type=type, clf=clf, i=i))
            merged = pd.read_csv(f"../LING/data/{split}_mimic_{type}{clf}_{i}.csv".format(split=split, type=type, clf=clf, i=i), index_col=None)
            merged['text'] = [clean(row['TEXT']) for index, row in merged.iterrows()]
            df = pd.DataFrame([get_vector(row, model) for row in merged['text']])
            print("writing file ../LING/features/mimic_{split}_{name}_{type}{clf}_{i}.csv".format(split=split, type=type, clf=clf, name=name, i=i))
            df.to_csv('../LING/features/mimic_{split}_{name}_{type}{clf}_{i}.csv'.format(split=split, type=type, clf=clf, name=name, i=i), index=False)


def clean(text):

    stopwords = ['is', 'was', 'are', 'were', 'on', 'in', 'up', 'by', 'or', 'and', 'the', 'a', 'an',
                   'at', 'which', 'when',  'after', 'with', 'as']
    text = text.replace("\n", " ")
    text = text.replace("*", " ")
    text = text.replace("[",  " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("-", " ")
    text = text.replace(":", " ")
    text = text.replace("%", " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace("#", " ")
    text = text.replace("/", " ")
    text = text.replace("@", " ")
    text = text.replace('?', ' ')
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = ''.join([i for i in text if not i.isdigit()])
    for word in stopwords:
        text = text.replace('\b'+word+'\b', " ")
        text = text.replace(' ' + word + ' ', " ")
    text = text.replace("  ", " ")
    return text.lower()

def explain(pipeline, index_num, method, clf):
    class_names = ['not abuse', 'abuse']
    print("index", index_num)
    idx = y[1].index[index_num]
    explainer = LimeTextExplainer(class_names=class_names)

    text1 = clean(y[1]['TEXT'][idx].iloc[0])
    text2 = clean(y[1]['TEXT'][idx].iloc[1])

    # print(text1)
    # text2 = clean(y[1]['TEXT'][idx].iloc[1])
    # print(text2)
    exp1 = explainer.explain_instance(text1
    , pipeline.predict_proba, num_features=50, labels=(1,))

    exp1.save_to_file('../expl2/lime1_{index}_{method}_{clf}_{clf_}.html'.format(index=index_num, method=method, clf=clf, clf_=clf_))

    exp2 = explainer.explain_instance(text2
                                   , pipeline.predict_proba, num_features=50, labels=(1,))
    #exp.as_pyplot_figure()
    exp2.save_to_file('../expl2/lime2_{index}_{method}_{clf}_{clf_}.html'.format(index=index_num, method=method, clf=clf, clf_=clf_))
    pd.DataFrame(exp1.as_list()).to_csv("../expl2/lime1_{index}_{method}_{clf}_{clf_}.csv".format(index=index_num, method=method, clf=clf, clf_=clf_))
    pd.DataFrame(exp2.as_list()).to_csv(
        "../expl2/lime2_{index}_{method}_{clf}_{clf_}.csv".format(index=index_num, method=method, clf=clf, clf_=clf_))

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
            merged = pd.read_csv("../LING/data/" + split + "_mimic_{type}{clf}_{i}.csv".format(type=type, clf=clf, i=i), index_col=None)
            merged['text'] = [clean(row['TEXT']) for index, row in merged.iterrows()]
            df = pd.DataFrame([get_word_vector([bert_model, bert_tokenizer], row, 1) for row in merged['text']])
            df.to_csv('../LING/features/mimic_{split}_{name}_{type}{clf}_{i}.csv'.format(split=split, type=type, clf=clf, name=name, i=i), index=False)

# Step 1: create swapped and neutralized samples.
def create_MIMIC(clf):
    for split in ['train', 'test']:
        for i in range(10):
            merged = pd.read_csv("../LING/data/" + split + "_mimic_{clf}_{i}.csv".format(clf=clf, i=i), index_col=None)
            for val, str_ in [[True, 'neutr'], [False, 'swapped']]:
                tmp = merged.copy(deep=True)
                (tmp['TEXT']) = [gender_swapping(row['TEXT'], row['GENDER'], neutralize=val) for index, row in merged.iterrows()]
                (tmp['GENDER']) = [swap_gender(row['GENDER']) for index, row in merged.iterrows()]
                tmp.to_csv("../LING/data/{split}_mimic_{str}_{clf}_{i}.csv".format(split=split, str=str_, clf=clf, i=i), index=False)

# Step 2: extract features for original, swapped and neutralized test samples.
def extract_all_feat(clf, type):
    print("extraction: ")
    extract_w2vec(clf, type, clinical=True)
    extract_w2vec(clf, type, clinical=False)

    #extract_bert(clf, type, clinical=True)
    #extract_bert(clf, type, clinical=False)
    return

# Step 3: run experiments with different learners.

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

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

def prep_split_t_stats(X, y, C):

    if method == 'augmented': 
        C = C*2
    data = pd.concat([X[0], X[1]])
    Y = pd.concat([y[0][[attr, label]], y[1][[attr, label]]])

    all = pd.concat([data, Y], axis=1)

    male_data = all[all[attr] == 'M']
    female_data = all[all[attr] == 'F']

    male_pos = np.where(male_data[label] == 1)
    female_pos = np.where(female_data[label] == 1)

    male_neg = np.where(male_data[label] == 0)
    female_neg = np.where(female_data[label] == 0)

    male_train_neg = np.random.choice(male_neg[0], size=C, replace=False)
    female_train_neg = np.random.choice(female_neg[0], size=C, replace=False)
    male_train_pos = np.random.choice(male_pos[0], size=C, replace=False)
    female_train_pos = np.random.choice(female_pos[0], size=C, replace=False)


    index = np.concatenate([male_train_neg, female_train_neg, male_train_pos, female_train_pos])

    print("length ...", len(index))    
    df1 = pd.DataFrame(all[all.index.isin(index)].to_numpy()).sample(frac=1)
    df2 = pd.DataFrame(all[~all.index.isin(index)].to_numpy()).sample(frac=1)

    #new_y = collections.defaultdict(list)
    #new_x = collections.defaultdict(list)

    new_x = []
    new_y = []
    for d in [df1, df2]:

        new_y.append(d.iloc[:, -2:].reset_index(drop=True))
        new_y[-1].columns = [attr, label]
        new_y[-1].astype({attr: 'string', label: 'string'}) 
        print(new_y[-1][label])
        new_x.append(d.iloc[:, :-2].reset_index(drop=True))

    print(new_y[0].columns)
    return new_x, new_y



if __name__ == "__main__":
    print("*********** Bias analysis EXPERIMENTS ********")
    #clf_label = 'DEPRESSION'
    attr = 'GENDER'
    text_she = "she is currently receiving\nBumex 2 mg PO q.d. and Aldactone 20 mg PO q.d. for diuresis.\n\n#6.  PSYCHIATRY:  The Psychiatric Service was consulted.\nApparently, the patient had trouble falling asleep at night\nand then dozing off during the day.  Ritalin was discontinued\nand she was started on Risperidone 0.5 mg q.d. "
    text_he = "he is currently receiving\nBumex 2 mg PO q.d. and Aldactone 20 mg PO q.d. for diuresis.\n\n#6.  PSYCHIATRY:  The Psychiatric Service was consulted.\nApparently, the patient had trouble falling asleep at night\nand then dozing off during the day.  Ritalin was discontinued\nand she was started on Risperidone 0.5 mg q.d. "
#'ALCOHOL.ABUSE', 'DEPRESSION',

    mental_arr =  ['SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS_majority', 'ALCOHOL.ABUSE_majority', 'OTHER.SUBSTANCE.ABUSE_majority', 'DEPRESSION_majority']
    #, 'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS_majority', 'ALCOHOL.ABUSE_majority', 'OTHER.SUBSTANCE.ABUSE_majority']
    embeddings = ['w2vec_news', 'biowordvec']
    #, w2vec_news 'biowordvec', 'bert', 'clinical_bert']
    #
    method_arr = ['orig', 'augmented']
    #'orig', 'swapped','neutr', 'augmented'
    #mental_arr = ['DEPRESSION_majority']
     #
    #,
    #for data in mental_arr:
     #  create_MIMIC(data)
    
    #for clf in mental_arr:
     #   for type in ['neutr_']:
      #      print("Extraction started for {clf} and {type}".format(clf=clf, type=type))
       #     extract_all_feat(clf, type)


    # biowordvec = gensim.models.KeyedVectors.load_word2vec_format(
    #     "../resources/bio_embedding_extrinsic", binary=True)
    # test_vec_she = pd.DataFrame(get_vector(text_she, biowordvec))
    # test_vec_he = pd.DataFrame(get_vector(text_he, biowordvec))

    df_results = pd.DataFrame()
    #index_arr = False
    for clf_ in ['svm']:
        for clf in mental_arr:
            for range_i in range(10):
                embedding_arr = []
                scores_arr = collections.defaultdict(list)
                for embedding in embeddings:
                #'biowordvec', 'bert', 'clinical_bert'
                    for method in method_arr:
                    #'orig', 'swapped',  'augmented'
                        embedding_arr.append(embedding)
                        print("model and method", embedding, method)
                        X, y = read_vec(embedding, method, clf, range_i)
                        print("size of the training X: {x} and Y: {yi}".format(x=X[0].shape, yi=(y[0].shape)))
                        print("size of the test X: {x} and Y: {yi}".format(x=X[1].shape, yi=(y[1].shape)))

                        label = clf
                        indices = []
                        #X, y = prep_split_t_stats(X, y, 55)

                        y_preds, classify, y_preds_prob = classifier(X, y, label=label, attr=attr, model=clf_)
                        pd.concat([pd.DataFrame(y_preds, columns=['pred']),  pd.DataFrame(y_preds_prob[:, 1], columns=['prob'])], axis=1).to_csv('preds_{embedding}_{clf}_{clf_}.csv'.format(embedding=embedding, clf=clf, clf_=clf_))
                    #pipeline = make_pipeline(our_extractor, classify)

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


                        data = data_sub2.copy(deep=True)
                    #print(y_preds)
                    #print(y_preds_prob[:,1])

                    # if method == 'neutr' or (method == 'orig'):
                    #     index_arr = qualitative_analysis(label)
                    # else:
                    #     index_ = int(len(y_preds)/2)
                    #     data_sub = data[0:index_].reset_index()
                    #     data_sub2 = data[index_:(index_*2)].reset_index()
                    #     corrected_indices =  (data_sub.index[data_sub['preds'] == data_sub2['preds']].tolist())
                    #     #print("corrected indices: ", list(set(index_arr) & set(corrected_indices)))
                    #     indices= list(set(index_arr) & set(corrected_indices))
                    #     correct_indices = (data_sub.index[data_sub['preds'] == y[1][0:index_][label]].tolist())
                    #     print(list(set(indices) & set(corrected_indices)))
                    # print("method {m} and pred: {prob} {p}".format(m=method, prob=data.iloc[9]['prob'],  p=data.iloc[9]['preds']))
                    # for index_num in [1, 99, 100, 9, 122]:
                    #     print("method, index: " + method + " "+ y_preds[index_num])
                        # try:
                        #     explain(pipeline, index_num, method, clf)
                        # except Exception:
                        #     print("EXCEPTION")
                        #     pass
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
                            scores_arr['F1-' + t].append(round(f1_score(group[label], group['preds'], average='macro'), 2))
                            if t == 'M':
                                scores_arr['Pos-' + t].append(len(male_1))
                                scores_arr['Neg-' + t].append(len(male_0))
                            else:
                                scores_arr['Pos-' + t].append(len(female_1))
                                scores_arr['Neg-' + t].append(len(female_0))

                        scores_arr['F1'].append(round(f1_score(y[1][label], y_preds, average='macro'), 2))
                        scores_arr['mismatch'].append(mismatch)
                        scores_arr['total_num'].append(i)

                df_results['embedding'] = embedding_arr
                for mi in ['FNR', 'FPR', 'TPR', 'F1', 'Pos', 'Neg']:
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
                df_results.to_csv("../t-test/originaltest_experiments_{clf}_{clf_}_{i}.csv".format(clf=clf, clf_=clf_, i=range_i))
