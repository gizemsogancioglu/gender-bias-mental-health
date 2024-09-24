# Fairness in AI-Based Mental Health: Clinician Perspectives and Bias Mitigation
[![Size](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/repo-size/gizemsogancioglu/gender-bias-mental-health)
[![License](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)](https://img.shields.io/github/license/gizemsogancioglu/gender-bias-mental-health)
![GitHub top language](https://img.shields.io/github/languages/top/gizemsogancioglu/gender-bias-mental-health)

Here, we provide source code for the following paper: [Sogancioglu, Gizem, Pablo Mosteiro, Albert Ali Salah, Floortje Scheepers, Heysem Kaya. "Fairness in AI-Based Mental Health: Clinician Perspectives and Bias Mitigation." Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society. 2024.]

Available in this repository: 
- Extracted word embeddings from the MIMIC-III dataset (features/*.csv)  
- Scripts for experiments (source/study2/*.py)
  
        .
        ├── study2                          # including source files (feature extractor, training, and bias mitigation scripts)                
        │   ├── mimic_experiments.py        # main script to run experiments with different bias mitigation methods (gender-specific models, data augmentation/neutralization, pre/post-processing)
        │   ├── participatory.py            # gain-based model selection experiments.
        │   ├── *.py                        # other source files include helper functions for data preparation, preprocessing, embedding extraction, and bias mitigation. 
        └── features                        # extracted w2vec/biowordvec/bert/clinical_bert features (mimic_{embedding}_[orig/neutr/swapped].csv). 


## References
* Paper: [Sogancioglu, Gizem, Pablo Mosteiro, Albert Ali Salah, Floortje Scheepers, Heysem Kaya. "Fairness in AI-Based Mental Health: Clinician Perspectives and Bias Mitigation." Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society. 2024.]
* For more information or any problems, please contact: gizemsogancioglu@gmail.com
