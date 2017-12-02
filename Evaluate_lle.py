import time
import pickle
import Evaluation_Stage as es


def run():
    print("Evaluating the lle results")
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    print("\nNow reading datasets and labels from disk")
    # read all datasets files
    original_datasets = pickle.load(open('original_datasets.p', 'rb'))
    datasets_labels = pickle.load(open('datasets_labels.p', 'rb'))
    lle_reduced_datasets_under_diff_k = pickle.load(open('lle_reduced_datasets_under_diff_k.p', 'rb'))
    print("Finished! \n")
    while True:
        k = input("Please specify the value of parameter k(5 - 15):\n")
        k = int(k)
        if k < 5 or k > 15:
            print("Invalid number, try again")
        else:
            break
    lle_reduced_datasets = lle_reduced_datasets_under_diff_k[k - 5]
    for key in lle_reduced_datasets:
        es.full_dataset_evaluation("lle_reduced(k=" + str(k) + ")_" + key, lle_reduced_datasets[key],
                                original_datasets[key],
                                datasets_labels[key])
    print("Local current time :", localtime)
    print("Finished lle results evaluation!")


