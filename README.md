# The effects of gender bias in word embeddings on patient phenotyping in the mental health domain
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

![Alt text](pipeline.png?raw=true "The proposed apparent personality prediction model")


## TODO
--- 

## References
* Paper: [The effects of gender bias in word embeddings on patient phenotyping in the mental health domain, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]
* For more information or any problems, please contact: g.sogancioglu@uu.nl
