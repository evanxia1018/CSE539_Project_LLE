import time
import Evaluation as evaluation
import pickle
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
    # get all original datasets
    swiss_roll_dataset = original_datasets[0]
    helix_dataset = original_datasets[1]
    twin_peaks_dataset = original_datasets[2]
    broken_swiss_dataset = original_datasets[3]
    hd_dataset = original_datasets[4]
    MNIST_images = original_datasets[5]
    # labels
    datasets_labels = pickle.load(open('datasets_labels.p', 'rb'))
    swiss_roll_labels = datasets_labels[0]
    helix_labels = datasets_labels[1]
    twin_peaks_labels = datasets_labels[2]
    broken_swiss_labels = datasets_labels[3]
    hd_labels = datasets_labels[4]
    MNIST_labels = datasets_labels[5]
    # get all pca reduced datasets
    pca_reduced_swiss_roll = pca_reduced_datasets[0]
    pca_reduced_helix = pca_reduced_datasets[1]
    pca_reduced_twin_peaks = pca_reduced_datasets[2]
    pca_reduced_broken_swiss = pca_reduced_datasets[3]
    pca_reduced_hd = pca_reduced_datasets[4]
    pca_reduced_MNIST_images = pca_reduced_datasets[5]
    print("Finish! \n")

    # The following code evaluate Trustworthiness, Continuity and Generalization error on datasets reduced by PCA
    print("Now evaluating Trustworthiness, Continuity and Generalization error on datasets reduced by PCA")
    trust_pca_swiss_roll, continuity_pca_swiss_roll = evaluation.get_trustworthiness_and_continuity(
        pca_reduced_swiss_roll, swiss_roll_dataset, 12)
    trust_pca_helix, continuity_pca_helix = evaluation.get_trustworthiness_and_continuity(pca_reduced_helix,
                                                                                          helix_dataset, 12)
    trust_pca_twin_peaks, continuity_pca_twin_peak = evaluation.get_trustworthiness_and_continuity(
        pca_reduced_twin_peaks, twin_peaks_dataset, 12)
    trust_pca_broken_swiss, continuity_pca_broken_swiss = evaluation.get_trustworthiness_and_continuity(
        pca_reduced_broken_swiss, broken_swiss_dataset, 12)
    trust_pca_hd, continuity_pca_hd = evaluation.get_trustworthiness_and_continuity(pca_reduced_hd, hd_dataset, 12)
    trust_pca_MNIST, continuity_pca_MNIST = evaluation.get_trustworthiness_and_continuity(pca_reduced_MNIST_images,
                                                                                          MNIST_images, 12)

    error_pca_swiss_roll = evaluation.get_generalization_error(pca_reduced_swiss_roll, swiss_roll_dataset,
                                                               swiss_roll_labels)
    error_pca_helix = evaluation.get_generalization_error(pca_reduced_helix, helix_dataset, helix_labels)
    error_pca_twin_peaks = evaluation.get_generalization_error(pca_reduced_twin_peaks, twin_peaks_dataset,
                                                               twin_peaks_labels)
    error_pca_broken_swiss = evaluation.get_generalization_error(pca_reduced_broken_swiss, broken_swiss_dataset,
                                                                 broken_swiss_labels)
    error_pca_hd = evaluation.get_generalization_error(pca_reduced_hd, hd_dataset, hd_labels)
    error_pca_MNIST = evaluation.get_generalization_error(pca_reduced_MNIST_images, MNIST_images, MNIST_labels)

    print("The Trustworthiness of the swiss_roll dataset(reduced by PCA) is: " + str(trust_pca_swiss_roll))
    print("The Trustworthiness of the helix dataset(reduced by PCA) is: " + str(trust_pca_helix))
    print("The Trustworthiness of the twin peaks dataset(reduced by PCA) is: " + str(trust_pca_twin_peaks))
    print("The Trustworthiness of the broken swiss roll dataset(reduced by PCA) is: " + str(trust_pca_broken_swiss))
    print("The Trustworthiness of the HD dataset(reduced by PCA) is: " + str(trust_pca_hd))
    print("The Trustworthiness of the MNIST dataset(reduced by PCA) is: " + str(trust_pca_MNIST))
    print("The Continuity of the swiss_roll dataset(reduced by PCA) is: " + str(continuity_pca_swiss_roll))
    print("The Continuity of the helix dataset(reduced by PCA) is: " + str(continuity_pca_helix))
    print("The Continuity of the twin peaks dataset(reduced by PCA) is: " + str(continuity_pca_twin_peak))
    print("The Continuity of the broken swiss roll dataset(reduced by PCA) is: " + str(continuity_pca_broken_swiss))
    print("The Continuity of the HD dataset(reduced by PCA) is: " + str(continuity_pca_hd))
    print("The Continuity of the MNIST dataset(reduced by PCA) is: " + str(continuity_pca_MNIST))
    print("The Generalization error of the swiss_roll dataset(reduced by PCA) is: " + str(error_pca_swiss_roll))
    print("The Generalization error of the helix dataset(reduced by PCA) is: " + str(error_pca_helix))
    print("The Generalization error of the twin peaks dataset(reduced by PCA) is: " + str(error_pca_twin_peaks))
    print(
        "The Generalization error of the broken swiss roll dataset(reduced by PCA) is: " + str(error_pca_broken_swiss))
    print("The Generalization error of the hd dataset(reduced by PCA) is: " + str(error_pca_hd))
    print("The Generalization error of the MNIST dataset(reduced by PCA) is: " + str(error_pca_MNIST))

    print("Finished! \n")
    # The following code evaluate Generalization error on datasets reduced by PCA
    # The following code evaluate Trustworthiness and Continuity on datasets reduced by LLE under 16 different k (from 5 to 15)
    for i in range(0, len(lle_reduced_datasets_under_diff_k)):
        print(
            "Now evaluating Trustworthiness, Continuity and Generalization error of the artificial dataset reduced by LLE when k =  " + str(
                i + 5))
        lle_reduced_swiss_roll = lle_reduced_datasets_under_diff_k[i][0]
        lle_reduced_helix = lle_reduced_datasets_under_diff_k[i][1]
        lle_reduced_twin_peaks = lle_reduced_datasets_under_diff_k[i][2]
        lle_reduced_broken_swiss = lle_reduced_datasets_under_diff_k[i][3]
        lle_reduced_hd = lle_reduced_datasets_under_diff_k[i][4]
        lle_reduced_MNIST_images = lle_reduced_datasets_under_diff_k[i][5]

        trust_lle_swiss_roll, continuity_lle_swiss_roll = evaluation.get_trustworthiness_and_continuity(
            lle_reduced_swiss_roll, swiss_roll_dataset, 12)
        trust_lle_helix, continuity_lle_helix = evaluation.get_trustworthiness_and_continuity(lle_reduced_helix,
                                                                                              helix_dataset, 12)
        trust_lle_twin_peaks, continuity_lle_twin_peak = evaluation.get_trustworthiness_and_continuity(
            lle_reduced_twin_peaks, twin_peaks_dataset, 12)
        trust_lle_broken_swiss, continuity_lle_broken_swiss = evaluation.get_trustworthiness_and_continuity(
            lle_reduced_broken_swiss, broken_swiss_dataset, 12)
        trust_lle_hd, continuity_lle_hd = evaluation.get_trustworthiness_and_continuity(lle_reduced_hd, hd_dataset, 12)
        trust_lle_MNIST, continuity_lle_MNIST = evaluation.get_trustworthiness_and_continuity(lle_reduced_MNIST_images,
                                                                                              MNIST_images, 12)

        error_lle_swiss_roll = evaluation.get_generalization_error(lle_reduced_swiss_roll, swiss_roll_dataset,
                                                                   swiss_roll_labels)
        error_lle_helix = evaluation.get_generalization_error(lle_reduced_helix, helix_dataset, helix_labels)
        error_lle_twin_peaks = evaluation.get_generalization_error(lle_reduced_twin_peaks, twin_peaks_dataset,
                                                                   twin_peaks_labels)
        error_lle_broken_swiss = evaluation.get_generalization_error(lle_reduced_broken_swiss, broken_swiss_dataset,
                                                                     broken_swiss_labels)
        error_lle_hd = evaluation.get_generalization_error(lle_reduced_hd, hd_dataset, hd_labels)
        error_lle_MNIST = evaluation.get_generalization_error(lle_reduced_MNIST_images, MNIST_images, MNIST_labels)

        print("The Trustworthiness of the swiss_roll dataset(reduced by LLE) is: " + str(trust_lle_swiss_roll))
        print("The Trustworthiness of the helix dataset(reduced by LLE) is: " + str(trust_lle_helix))
        print("The Trustworthiness of the twin peaks dataset(reduced by LLE) is: " + str(trust_lle_twin_peaks))
        print("The Trustworthiness of the broken swiss roll dataset(reduced by LLE) is: " + str(trust_lle_broken_swiss))
        print("The Trustworthiness of the HD dataset(reduced by LLE) is: " + str(trust_lle_hd))
        print("The Trustworthiness of the MNIST dataset(reduced by LLE) is: " + str(trust_lle_MNIST))
        print("The Continuity of the swiss_roll dataset(reduced by LLE) is: " + str(continuity_lle_swiss_roll))
        print("The Continuity of the helix dataset(reduced by LLE) is: " + str(continuity_lle_helix))
        print("The Continuity of the twin peaks dataset(reduced by LLE) is: " + str(continuity_lle_twin_peak))
        print("The Continuity of the broken swiss roll dataset(reduced by LLE) is: " + str(continuity_lle_broken_swiss))
        print("The Continuity of the HD dataset(reduced by LLE) is: " + str(continuity_lle_hd))
        print("The Continuity of the MNIST dataset(reduced by LLE) is: " + str(continuity_lle_MNIST))
        print("The Generalization error of the swiss_roll dataset(reduced by LLE) is: " + str(error_lle_swiss_roll))
        print("The Generalization error of the helix dataset(reduced by LLE) is: " + str(error_lle_helix))
        print("The Generalization error of the twin peaks dataset(reduced by LLE) is: " + str(error_lle_twin_peaks))
        print("The Generalization error of the broken swiss roll dataset(reduced by LLE) is: " + str(
            error_lle_broken_swiss))
        print("The Generalization error of the hd dataset(reduced by LLE) is: " + str(error_lle_hd))
        print("The Generalization error of the MNIST dataset(reduced by LLE) is: " + str(error_lle_MNIST))

        print("Finished! \n")

    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)
    print("Stage 2 finished!")

