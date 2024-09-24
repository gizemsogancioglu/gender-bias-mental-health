import numpy as np
import json
from scipy import spatial
import pandas as pd
from sklearn.decomposition import PCA
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from source.common.embeddings import get_word_vector, bert_tokenizer, bert_model, clinical_model, clinical_tokenizer

synonym_dict = {}

#example templates.
gender_sentences = ['he is a father of a girl and boy and he loves his children',
                    'she is a mother of a girl and boy and she loves her children',
                    'she visited her grandmother and grandfather during the holiday',
                    'There same number of female and male employees in our company',
                    'There same number of woman and man in our company',
                    'she is a great gal',
                    'he is a great guy',
                    'she does it for herself',
                    'he does it for himself',
                    'she is a good daughter',
                    'he is a good son']

def read_disease_file():
    f = open('../../data/depression_synonyms.json', encoding='utf-8')
    synonym_dict = json.load(f)
    return synonym_dict


def compute_direct_bias(model, gender_vec, model_name):
    synonym_dict = read_disease_file()

    scores = defaultdict(list)
    word_list = []
    score_list = []

    context = synonym_dict['context']
    category = "depression"

    if not isinstance(synonym_dict[category], dict):
        for word in synonym_dict[category]:
            mental = get_word_vector(model, word + context[len(category) + 2:], 1, len(word.split(" ")))
            scores[category].append(cosine_similarity(gender_vec, mental))
            word_list.append(word)
            score_list.append(round(cosine_similarity(gender_vec, mental), 3))

    else:
        for key in synonym_dict[category].keys():
            for word in synonym_dict[category][key]:
                mental = get_word_vector(model, word + context[len(category) + 2:], 1, len(word.split(" ")))
                scores[category].append(cosine_similarity(gender_vec, mental))
                word_list.append(word)
                score_list.append(round(cosine_similarity(gender_vec, mental), 3))
    df = pd.DataFrame()
    df['score'] = score_list
    df['word'] = word_list

    df.to_csv('../output/' + model_name + '_word_bias_scores.csv')

    mean_scores_dict = defaultdict(list)
    for key in scores.keys():
        print(key, np.mean((scores[key])))
        mean_scores_dict[key] = [np.mean((scores[key])), np.max(np.absolute(scores[key])),
                                 np.median(np.absolute(scores[key]))]
    return mean_scores_dict


def identify_gender_space(model, model_name):
    vector_arr = []

    # 1.she - he, 2.her - his, 3.girl - boy, 4.mother - father, 5.grandmother - grandfather,
    # 6.female - male, 7.woman - man, 8.gal and guy, 9.daughter - son, 10.herself - himself
    for row_i, word_i, row_y, word_y in [[1, 1, 0, 1], [1, 13, 0, 13], [0, 7, 0, 9], [0, 4, 1, 4], [2, 4, 2, 6], [3, 5, 3, 7],
                                         [4, 5, 4, 7], [5, 5, 6, 5], [9, 5, 10, 5], [7, 5, 8,5]]:
        center = (get_word_vector(model, gender_sentences[row_i], word_i) + get_word_vector(model, gender_sentences[row_y], word_y)) / 2
        vector_arr.append(get_word_vector(model, gender_sentences[row_i], word_i) - center)
        vector_arr.append(get_word_vector(model, gender_sentences[row_y], word_y) - center)

    pca = PCA(n_components=5)
    pca.fit(np.array(vector_arr))
    print("explained variance: ", pca.explained_variance_ratio_)

    df = pd.DataFrame()
    df['explained variance'] = pca.explained_variance_ratio_
    df['components'] = [1, 2, 3, 4, 5]
    sns.barplot(data=df, x="components", y="explained variance")
    plt.ylim(0, 0.7)
    plt.savefig("../output/{model_name}_gender_space.png".format(model_name=model_name), bbox_inches='tight', pad_inches=0)
    return pca.components_[0]

def cosine_similarity(vec1, vec2):
    return (1 - spatial.distance.cosine(vec1, vec2))

def plot_direct_bias(plot_name):
    df = pd.read_csv("../output/" + plot_name + "_direct_bias_cat.csv", index_col=None)
    df.index = ['mean', 'max', 'median']
    df_T = df.T.reset_index()
    df_T = df_T.set_index('index').stack().reset_index()
    df_T = df_T.rename(columns={'index': 'category', 'level_1': 'summary', 0: 'bias score'})
    print(df_T)
    g = sns.barplot(x="category", y="bias score", hue="summary", data=df_T)
    for p in g.patches:
        g.annotate(format(p.get_height(), '.2f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')
    g.figure.savefig("../output/" + plot_name + ".png")


def project_to_sensitive_attr(gender_vec, disease_vector):
    projected_vector = gender_vec * np.dot(disease_vector, gender_vec) / np.dot(gender_vec, gender_vec)
    return projected_vector


if __name__ == "__main__":
    bert_model = [bert_model, bert_tokenizer]
    clinical_model = [clinical_model, clinical_tokenizer]

    for model, model_name in [[bert_model, 'bert-base-cased'], [clinical_model, 'clinical']]:
        gender_vec = identify_gender_space(model, model_name)
        print(cosine_similarity(get_word_vector(model, "he is a man", 1), gender_vec))
        print(cosine_similarity(get_word_vector(model, "she is a woman", 1), gender_vec))
        print(cosine_similarity(get_word_vector(model, "anxiety is a synonym of depression", 6), gender_vec))

        scores = compute_direct_bias(model, gender_vec, model_name)
        pd.DataFrame.from_dict(scores).to_csv("../output/" + model_name + "_direct_bias_cat.csv")
        plot_direct_bias(model_name)
