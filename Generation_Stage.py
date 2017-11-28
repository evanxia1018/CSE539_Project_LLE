import Dataset_Generator as dg
import Evaluation as evaluation
import MyLLE as lle
import numpy as np
import pickle as pk
import time

def run():
    # Following are the code for stage 1: Datasets creation and reduction
    # Please note that stage 1 must be done before stage 2
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    print("Stage 1: creating the five artificial dataset and reading the MNIST natural dataset, then generate datasets with reduced dimentionality, using LLE and PCA\n")

    # Note running 5000 samples may take awfully long time, while 1000 samples takes only around 30 mins.
    print("Now generating the five artificial datasets and reading the MNIST dataset")
    swiss_roll_dataset = dg.get_swiss_roll_dataset(5000)
    helix_dataset = dg.get_helix_dataset(5000)
    twin_peaks_dataset = dg.get_twin_peaks(5000)
    broken_swiss_dataset = dg.get_broken_swiss_roll_dataset(5000)
    hd_dataset = dg.get_hd_dataset(5000)
    MNIST_images, MNIST_labels = evaluation.get_natural_dataset_samples(5000)
    original_datasets = [swiss_roll_dataset, helix_dataset, twin_peaks_dataset, broken_swiss_dataset, hd_dataset,
                         MNIST_images]
    pk.dump(original_datasets, open('original_datasets.p', 'wb'))
    print("Finished! \n")

    print("Now getting labels for all datasets")
    swiss_roll_labels = evaluation.get_artificial_dataset_labels(swiss_roll_dataset)
    helix_labels = evaluation.get_artificial_dataset_labels(helix_dataset)
    twin_peak_labels = evaluation.get_artificial_dataset_labels(twin_peaks_dataset)
    broken_swiss_labels = evaluation.get_artificial_dataset_labels(broken_swiss_dataset)
    hd_labels = evaluation.get_artificial_dataset_labels(hd_dataset)
    datasets_labels = [swiss_roll_labels, helix_labels, twin_peak_labels, broken_swiss_labels, hd_labels, MNIST_labels]
    pk.dump(datasets_labels, open('datasets_labels.p', 'wb'))
    print("Finished! \n")

    # The following code reduces dimensionality using PCA and LLE
    print("Now using PCA to reduce dimensionality of each dataset")
    pca_reduced_swiss_roll = evaluation.pca_dim_reduction(swiss_roll_dataset, 2)
    pca_reduced_helix = evaluation.pca_dim_reduction(helix_dataset, 1)
    pca_reduced_twin_peaks = evaluation.pca_dim_reduction(twin_peaks_dataset, 2)
    pca_reduced_broken_swiss = evaluation.pca_dim_reduction(broken_swiss_dataset, 2)
    pca_reduced_hd = evaluation.pca_dim_reduction(hd_dataset, 2)
    pca_reduced_MNIST_images = evaluation.pca_dim_reduction(MNIST_images, 20)
    pca_reduced_datasets = [pca_reduced_swiss_roll, pca_reduced_helix, pca_reduced_twin_peaks, pca_reduced_broken_swiss,
                            pca_reduced_hd, pca_reduced_MNIST_images]
    pk.dump(pca_reduced_datasets, open('pca_reduced_datasets.p', 'wb'))
    print("Finished! \n")

    lle_reduced_datasets_under_diff_k = []  # this list contains results under different k parameter, where idx i is the result for k = i + 5
    print(
        "Now using LLE to reduce dimensionality of each dataset. Note that the parameter k ranges from 5 to 15 so this step is gonna take a while")
    for k in range(5, 16):
        curr_k_results = []
        lle_reduced_swiss_roll = lle.locally_linear_embedding(np.array(swiss_roll_dataset, np.float64), k, 2)[
            0].tolist()
        lle_reduced_helix = lle.locally_linear_embedding(np.array(helix_dataset, np.float64), k, 1)[0].tolist()
        lle_reduced_twin_peaks = lle.locally_linear_embedding(np.array(twin_peaks_dataset, np.float64), k, 2)[
            0].tolist()
        lle_reduced_broken_swiss = lle.locally_linear_embedding(np.array(broken_swiss_dataset, np.float64), k, 2)[
            0].tolist()
        lle_reduced_hd = lle.locally_linear_embedding(np.array(hd_dataset, np.float64), k, 5)[0].tolist()
        lle_reduced_MNIST_images = lle.locally_linear_embedding(np.array(MNIST_images, np.float64), k, 20)[0].tolist()
        curr_k_results.append(lle_reduced_swiss_roll)
        curr_k_results.append(lle_reduced_helix)
        curr_k_results.append(lle_reduced_twin_peaks)
        curr_k_results.append(lle_reduced_broken_swiss)
        curr_k_results.append(lle_reduced_hd)
        curr_k_results.append(lle_reduced_MNIST_images)
        lle_reduced_datasets_under_diff_k.append(curr_k_results)
    pk.dump(lle_reduced_datasets_under_diff_k, open('lle_reduced_datasets_under_diff_k.p', 'wb'))
    print("Finished \n")
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    # ************************ End of the stage 1


