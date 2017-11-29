import time
import pickle
import Evaluation_Stage as es

print("Evaluating the pca results")
localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
print("\nNow reading datasets and labels from disk")
# read all datasets files
original_datasets = pickle.load(open('original_datasets.p', 'rb'))
datasets_labels = pickle.load(open('datasets_labels.p', 'rb'))
pca_reduced_datasets = pickle.load(open('pca_reduced_datasets.p', 'rb'))
print("Finished! \n")
# The following code evaluate Trustworthiness, Continuity and Generalization error on datasets reduced by PCA
for key in pca_reduced_datasets:
    es.full_dataset_evaluation("pca_reduced_" + key, pca_reduced_datasets[key], original_datasets[key],
                            datasets_labels[key])
