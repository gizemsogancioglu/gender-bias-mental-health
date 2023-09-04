import numpy as np
from transformers import AutoTokenizer, AutoModel

clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
bert_model = 'bert-base-cased'
pubmed_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

(clinical_tokenizer, clinical_model) = [AutoTokenizer.from_pretrained(clinical_model_name), AutoModel.from_pretrained(clinical_model_name)]
(bert_tokenizer, bert_model) = [AutoTokenizer.from_pretrained(bert_model), AutoModel.from_pretrained(bert_model)]

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

