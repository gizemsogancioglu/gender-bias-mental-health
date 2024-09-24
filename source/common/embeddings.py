import collections

import gensim
import gensim.downloader as api
import numpy as np
from source.common.text_processing import clean
from transformers import AutoTokenizer, AutoModel
import pandas as pd

w2vec_model = 'word2vec-google-news-300'
clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
bert_model = 'bert-base-cased'

(clinical_tokenizer, clinical_model) = [AutoTokenizer.from_pretrained(clinical_model_name), AutoModel.from_pretrained(clinical_model_name)]
(bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(bert_model), AutoModel.from_pretrained(bert_model)]


def set_length():
    embedding_length = collections.defaultdict(list)
    embedding_length['w2vec_news'] = 300
    embedding_length['biowordvec'] = 200
    embedding_length['bert'] = 768
    embedding_length['clinical_bert'] = 768
    return embedding_length

def extract_all_feat(type):
    print("extraction: ")
    extract_w2vec(type, clinical=True)
    extract_w2vec(type, clinical=False)
    extract_bert(type, clinical=True)
    extract_bert(type, clinical=False)
    return


def extract_w2vec(type, clinical=True):
    if clinical:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../resources/bio_embedding_extrinsic", binary=True)
    else:
        model = api.load(w2vec_model)

    name = 'biowordvec' if clinical else "w2vec_news"

    print("reading w2vec file")
    merged = pd.read_csv(f"../mimic_{type}.csv".format(type=type), index_col=None)
    merged['clean_text'] = [clean(row['TEXT']) for index, row in merged.iterrows()]
    vector_df = pd.DataFrame([get_vector(row, model) for row in merged['clean_text']])
    final_df = pd.concat([merged, vector_df], axis=1)
    print("writing file ..")
    final_df.to_csv('../mimic_{name}_{type}.csv'.format(name=name, type=type), index=False)


def extract_bert(type, clinical=True):
    if clinical:
        (bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(clinical_model_name),
                                        AutoModel.from_pretrained(clinical_model_name)]

    else:
        bert_model = 'bert-base-uncased'
        (bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(bert_model),
                                        AutoModel.from_pretrained(bert_model)]

    name = 'clinical_bert' if clinical else 'bert'

    merged = pd.read_csv(f"../mimic_{type}.csv".format(type=type), index_col=None)
    # merged['clean_text'] = [clean(row['TEXT'], stop_words=False) for index, row in merged.iterrows()]

    vector_df = pd.DataFrame([get_word_vector([bert_model, bert_tokenizer], row, 1) for row in merged['TEXT']])
    final_df = pd.concat([merged, vector_df], axis=1)
    print("writing file ..")
    final_df.to_csv('../mimic_{name}_{type}.csv'.format(name=name, type=type), index=False)


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
            continue
        vector_list.append(vectors[word.lower()])
    if len(vector_list) == 0:
        return None
    return np.average(np.asarray(vector_list), axis=0)

