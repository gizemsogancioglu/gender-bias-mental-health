# The effects of gender bias in word embeddings on patient phenotyping in the mental health domain
[![Size](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)
[![License](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)
![GitHub top language](https://img.shields.io/github/languages/top/gizemsogancioglu/gender-bias-mental-health)

Here, we provide source code for the following paper: [The effects of gender bias in word embeddings on
patient phenotyping in the mental health domain, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]

Available in this repository: 
- Extracted word embeddings from the MIMIC-III dataset (features/feature_[train/test].csv)  
- Scripts for bias analysis in word embeddings and phenotype classification (source/*.py)
- Depressed mood syptom list 

        .
        ├── source                          # including source files (feature extractor, training, and predictor scripts)                
        │   ├── embeddings.py               # feature extraction methods for word embeddings. 
        │   ├── phenotype_experiments.py    # main script for phenotype classification experiments with original/swapped/neutralized/augmented datasets.
        │   ├── text_processing.py          # includes gender pronun removal and text cleaning methods.
        ├── data                         
        │   ├── depression_synonyms.json    # gathered from [Identifying Symptom Information in Clinical Notes Using Natural Language Processing](https://pubmed.ncbi.nlm.nih.gov/33196504/)
        └── features                        # extracted w2vec features. 

## Qualitative Analysis

![original treatment, male pronouns](FIG/orig_he.png?classes=caption "original treatment, male pronouns")
*1. original treatment, male pronouns*
![neutralized treatment, male pronouns](FIG/neutr_he.png?raw=true "neutralized treatment, male pronouns")
*2. neutralized treatment, male pronouns*
![augmented treatment, male pronouns](FIG/aug_he.png?raw=true "augmented treatment, male pronouns")
*3. augmented treatment, male pronouns*
![original treatment, female pronouns](FIG/orig_she.png?raw=true "original treatment, female pronouns")
*4. original treatment, female pronouns*
![neutralized treatment, female pronouns](FIG/neutr_she.png?raw=true "neutralized treatment, female pronouns")
*5. neutralized treatment, female pronouns*
![augmented treatment, female pronouns](FIG/aug_she.png?raw=true "augmented treatment, female pronouns")
*6. augmented treatment, female pronouns*



## TODO
--- 

## References
* Paper: [The effects of gender bias in word embeddings on patient phenotyping in the mental health domain, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]
* For more information or any problems, please contact: gizemsogancioglu@gmail.com
