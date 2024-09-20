# Towards Fair AI in mental health 
[![Size](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)
[![License](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)
![GitHub top language](https://img.shields.io/github/languages/top/gizemsogancioglu/gender-bias-mental-health)

Here, we provide source code for the following papers:
- [The effects of gender bias in word embeddings on
patient phenotyping in the mental health domain, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]. Details are provided [here](https://github.com/gizemsogancioglu/gender-bias-mental-health/tree/main/source/study1). 

- [Sogancioglu, Gizem, Pablo Mosteiro, Albert Ali
Salah, Floortje Scheepers, Heysem Kaya. "Fairness in AI-Based Mental Health:
Clinician Perspectives and Bias Mitigation." Proceedings of the 2024 AAAI/ACM
Conference on AI, Ethics, and Society. 2024.]. Details are provided [here](https://github.com/gizemsogancioglu/gender-bias-mental-health/tree/main/source/study2). 

Available in this repository: 

        .
        ├── study1                          # including source files (feature extractor, training, and predictor scripts)                
        │   ├── embeddings.py               # feature extraction methods for word embeddings. 
        │   ├── phenotype_experiments.py    # main script for phenotype classification experiments with original/swapped/neutralized/augmented datasets.
        │   ├── text_processing.py          # includes gender pronun removal and text cleaning methods.
        ├── study2                          # including source files (feature extractor, training, and bias mitigation scripts)                
        │   ├── mimic_experiments.py        # main script to run experiments with different bias mitigation methods (gender-specific models, data augmentation/neutralization, pre/post-processing)
        │   ├── participatory.py            # gain-based model selection experiments.
        │   ├── *.py                        # other source files include helper functions for data preparation, preprocessing, embedding extraction, and bias mitigation. 
        ├── data                            #                 
        │   ├── depression_synonyms.json    # gathered from [Identifying Symptom Information in Clinical Notes Using Natural Language Processing](https://pubmed.ncbi.nlm.nih.gov/33196504/)
        └── features                        # extracted w2vec/biowordvec/bert/clinical_bert features (mimic_{embedding}_[orig/neutr/swapped].csv). 

