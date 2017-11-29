import time
import Evaluation as evaluation
import pickle


def full_dataset_evaluation(name, reduced_dataset, original_dataset, dataset_label):
    print("Now evaluating " + name)
    trust, continuity = evaluation.get_trustworthiness_and_continuity(reduced_dataset, original_dataset, 12)
    error = evaluation.get_generalization_error(reduced_dataset, dataset_label)
    print("The Trustworthiness of the " + name + " is: " + str(trust))
    print("The Continuity of the " + name + " is: " + str(continuity))
    print("The Generalization Error of the " + name + " is: " + str(error))
    print("Finished!\n")


# Following are the code for stage 2: technique evalutation
def run():
    print("Stage 2: Evaluating the datasets")
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    # read generated_datasets from disk
    print("\nNow reading datasets and labels from disk")
    # read all datasets files
    original_datasets = pickle.load(open('original_datasets.p', 'rb'))
    datasets_labels = pickle.load(open('datasets_labels.p', 'rb'))
    pca_reduced_datasets = pickle.load(open('pca_reduced_datasets.p', 'rb'))
    lle_reduced_datasets_under_diff_k = pickle.load(open('lle_reduced_datasets_under_diff_k.p', 'rb'))
    print("Finished! \n")

    # The following code evaluate Trustworthiness, Continuity and Generalization error on datasets reduced by PCA
    for key in pca_reduced_datasets:
        full_dataset_evaluation("pca_reduced_" + key, pca_reduced_datasets[key], original_datasets[key],
                                  datasets_labels[key])
    # The following code evaluate Trustworthiness, Continuity and Generalization error on datasets reduced by PCA
    for i in range(0, len(lle_reduced_datasets_under_diff_k)):
        lle_reduced_datasets = lle_reduced_datasets_under_diff_k[i]
        for key in lle_reduced_datasets:
            full_dataset_evaluation("lle_reduced(k=" + str(i + 5) + ")_" + key, lle_reduced_datasets[key], original_datasets[key],
                                      datasets_labels[key])
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    print("Stage 2 finished!")



