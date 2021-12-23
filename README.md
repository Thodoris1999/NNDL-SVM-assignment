# NNDL-SVM-assignment
2nd assignment for Neural Networks/Deep Learning course. As always, it is recommended to create an environment and install the requirements with

`pip install -r requirements.txt`

## Scripts
- `svm(_pca).py -m <joblib file>` is used to train an SVM (with PCA preprocessing) model with parameters defined inside the script and store the trained parameters in <joblib_file>
- `svm(_pca)_hyper_tune.py -m <joblib file>` is used to tune the hyperparameters of an SVM (with PCA preprocessing) model and store the trained parameters in <joblib_file>
- `svm(_pca)_eval.py -m <joblib file>` is used to evaluate an SVM (with PCA preprocessing) model using the saved parameters from \<joblib file\>


## Pretrained parameters
- SVM: https://drive.google.com/file/d/1J1ReYqx6hMweJdJhhpqyvhAji7YEF8Kf/view?usp=sharing
- SVM+PCA: https://drive.google.com/file/d/1nb6sRLNSZSrdpbH2fHZVqGvhZGj2hNyX/view?usp=sharing
