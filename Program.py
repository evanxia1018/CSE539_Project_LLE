print("**********************************************************************")
print("Hello. This is CSE569 Project Demo, produced by Haisi Yi and Zheng Xia")
print("**********************************************************************\n\n")
import time;
localtime = time.asctime( time.localtime(time.time()) )
print ("Local current time :", localtime)

# Following are the code for stage 1: Datasets creation and reduction
# Please note that stage 1 must be done before stage 2
print("Stage 1: creating the five artificial dataset and reading the MNIST natural dataset, then generate datasets with reduced dimentionality, using LLE and PCA\n")
import Dataset_Generator as dg
import Evaluation as evaluation
swiss_roll_dataset = dg.get_swiss_roll_dataset(5000)
helix_dataset = dg.get_helix_dataset(5000)
twin_peaks_dataset = dg.get_twin_peaks(5000)
broken_swiss_dataset = dg.get_broken_swiss_roll_dataset(5000)
hd_dataset = dg.get_hd_dataset(5000)
MNIST_images, MNIST_labels = evaluation.get_natural_dataset_samples()
print("Finish datasets generation and reading successfully\n")
import MyLLE as lle
import numpy as np
print("Stage 2: evaluating PCA and LLE on each dataset\n")

# The following code reduces dimensionality using PCA and LLE
print("Now using PCA to reduce dimensionality of each dataset\n")
pca_reduced_swiss_roll = evaluation.pca_dim_reduction(swiss_roll_dataset, 2)
pca_reduced_helix = evaluation.pca_dim_reduction(helix_dataset, 1)
pca_reduced_twin_peaks = evaluation.pca_dim_reduction(twin_peaks_dataset, 2)
pca_reduced_broken_swiss = evaluation.pca_dim_reduction(broken_swiss_dataset, 2)
pca_reduced_hd = evaluation.pca_dim_reduction(hd_dataset, 2)
pca_reduced_MNIST_images = evaluation.pca_dim_reduction(MNIST_images, 20)

print("Now using LLE to reduce dimensionality of each dataset. Note that the parameter k is set to 5 for the sake of efficiency\n")
lle_reduced_swiss_roll = lle.locally_linear_embedding(np.array(swiss_roll_dataset, np.float64), 5, 2)[0].tolist()
lle_reduced_helix = lle.locally_linear_embedding(np.array(helix_dataset, np.float64), 5, 1)[0].tolist()
lle_reduced_twin_peaks = lle.locally_linear_embedding(np.array(twin_peaks_dataset, np.float64), 5, 2)[0].tolist()
lle_reduced_broken_swiss = lle.locally_linear_embedding(np.array(broken_swiss_dataset, np.float64), 5, 2)[0].tolist()
lle_reduced_hd = lle.locally_linear_embedding(np.array(hd_dataset, np.float64), 5, 5)[0].tolist()
lle_reduced_MNIST_images = lle.locally_linear_embedding(np.array(MNIST_images, np.float64), 5, 20)[0].tolist()
# ************************ End of the stage 1





# Following are the code for stage 2: technique evalutation
# The following code evaluate Trustworthiness of each technique on artificial datasets
print("Now evaluating Trustworthiness of on each reduced artificial dataset produced by each technique")
trust_pca_swiss_roll = evaluation.get_trustworthiness(pca_reduced_swiss_roll, swiss_roll_dataset, 12)
trust_lle_swiss_roll = evaluation.get_trustworthiness(lle_reduced_swiss_roll, swiss_roll_dataset, 12)
print("swiss_roll dataset(reduced by PCA) is: " + str(trust_pca_swiss_roll))
print("swiss_roll dataset(reduced by LLE) is: " + str(trust_lle_swiss_roll))

trust_pca_helix = evaluation.get_trustworthiness(pca_reduced_helix, helix_dataset, 12)
trust_lle_helix = evaluation.get_trustworthiness(lle_reduced_helix, helix_dataset, 12)
print("helix dataset(reduced by PCA) is: " + str(trust_pca_helix))
print("helix dataset(reduced by LLE) is: " + str(trust_lle_helix))

trust_pca_twin_peak = evaluation.get_trustworthiness(pca_reduced_twin_peaks, twin_peaks_dataset, 12)
trust_lle_twin_peak = evaluation.get_trustworthiness(lle_reduced_twin_peaks, twin_peaks_dataset, 12)
print("twin peak dataset(reduced by PCA) is: " + str(trust_pca_twin_peak))
print("twin peak dataset(reduced by LLE) is: " + str(trust_lle_twin_peak))

trust_pca_broken_swiss = evaluation.get_trustworthiness(pca_reduced_broken_swiss, broken_swiss_dataset, 12)
trust_lle_broken_swiss = evaluation.get_trustworthiness(lle_reduced_broken_swiss, broken_swiss_dataset, 12)
print("broken swiss roll dataset(reduced by PCA) is: " + str(trust_pca_broken_swiss))
print("broken swiss roll dataset(reduced by LLE) is: " + str(trust_lle_broken_swiss))

trust_pca_hd = evaluation.get_trustworthiness(pca_reduced_hd, hd_dataset, 12)
trust_lle_hd = evaluation.get_trustworthiness(lle_reduced_hd, hd_dataset, 12)
print("HD dataset(reduced by PCA) is: " + str(trust_pca_hd))
print("HD dataset(reduced by LLE) is: " + str(trust_lle_hd))
# ************** end of Trustworthiness evaluation of each technique

# The following code evaluate Continuity of each technique on artificial datasets
print("\nNow evaluating Continuity of each technique on artificial datasets")
continuity_pca_swiss_roll = evaluation.get_continuity(pca_reduced_swiss_roll, swiss_roll_dataset, 12)
continuity_lle_swiss_roll = evaluation.get_continuity(lle_reduced_swiss_roll, swiss_roll_dataset, 12)
print("swiss_roll dataset(reduced by PCA) is: " + str(continuity_pca_swiss_roll))
print("swiss_roll dataset(reduced by LLE) is: " + str(continuity_lle_swiss_roll))

continuity_pca_helix = evaluation.get_continuity(pca_reduced_helix, helix_dataset, 12)
continuity_lle_helix = evaluation.get_continuity(lle_reduced_helix, helix_dataset, 12)
print("helix dataset(reduced by PCA) is: " + str(continuity_pca_helix))
print("helix dataset(reduced by LLE) is: " + str(continuity_lle_helix))

continuity_pca_twin_peak = evaluation.get_continuity(pca_reduced_twin_peaks, twin_peaks_dataset, 12)
continuity_lle_twin_peak = evaluation.get_continuity(lle_reduced_twin_peaks, twin_peaks_dataset, 12)
print("twin peak dataset(reduced by PCA) is: " + str(continuity_pca_twin_peak))
print("twin peak dataset(reduced by LLE) is: " + str(continuity_lle_twin_peak))

continuity_pca_broken_swiss = evaluation.get_continuity(pca_reduced_broken_swiss, broken_swiss_dataset, 12)
continuity_lle_broken_swiss = evaluation.get_continuity(lle_reduced_broken_swiss, broken_swiss_dataset, 12)
print("broken swiss roll dataset(reduced by PCA) is: " + str(continuity_pca_broken_swiss))
print("broken swiss roll dataset(reduced by LLE) is: " + str(continuity_lle_broken_swiss))

continuity_pca_hd = evaluation.get_continuity(pca_reduced_hd, hd_dataset, 12)
continuity_lle_hd = evaluation.get_continuity(lle_reduced_hd, hd_dataset, 12)
print("HD dataset(reduced by PCA) is: " + str(continuity_pca_hd))
print("HD dataset(reduced by LLE) is: " + str(continuity_lle_hd))
# **************End of evaluation of Continuity on artificial datasets


# The following code evaluate Generalization errors of 1-NN classifiers trained on the artificial datasets reduced by each technique
print("\nNow evaluating Generalization errors of 1-NN classifiers trained on the artificial datasets reduced by each technique")
swiss_roll_labels = evaluation.get_artificial_dataset_labels(swiss_roll_dataset)
error_pca_swiss_roll = evaluation.get_generalization_error(pca_reduced_swiss_roll, swiss_roll_dataset, swiss_roll_labels)
error_lle_swiss_roll = evaluation.get_generalization_error(lle_reduced_swiss_roll, swiss_roll_dataset, swiss_roll_labels)
print("swiss_roll dataset(reduced by PCA) is: " + str(error_pca_swiss_roll))
print("swiss_roll dataset(reduced by LLE) is: " + str(error_lle_swiss_roll))

helix_labels = evaluation.get_artificial_dataset_labels(helix_dataset)
error_pca_helix = evaluation.get_generalization_error(pca_reduced_helix, helix_dataset, helix_labels)
error_lle_helix = evaluation.get_generalization_error(lle_reduced_helix, helix_dataset, helix_labels)
print("helix dataset(reduced by PCA) is: " + str(error_pca_helix))
print("helix dataset(reduced by LLE) is: " + str(error_lle_helix))

twin_peak_labels = evaluation.get_artificial_dataset_labels(twin_peaks_dataset)
error_pca_twin_peak = evaluation.get_generalization_error(pca_reduced_twin_peaks, twin_peaks_dataset, twin_peak_labels)
error_lle_twin_peak = evaluation.get_generalization_error(lle_reduced_twin_peaks, twin_peaks_dataset, twin_peak_labels)
print("twin peak dataset(reduced by PCA) is: " + str(error_pca_twin_peak))
print("twin peak dataset(reduced by LLE) is: " + str(error_lle_twin_peak))

broken_swiss_labels = evaluation.get_artificial_dataset_labels(broken_swiss_dataset)
error_pca_broken_swiss = evaluation.get_generalization_error(pca_reduced_broken_swiss, broken_swiss_dataset, broken_swiss_labels)
error_lle_broken_swiss = evaluation.get_generalization_error(lle_reduced_broken_swiss, broken_swiss_dataset, broken_swiss_labels)
print("broken swiss dataset(reduced by PCA) is: " + str(error_pca_broken_swiss))
print("broken swiss dataset(reduced by LLE) is: " + str(error_lle_broken_swiss))

hd_labels = evaluation.get_artificial_dataset_labels(hd_dataset)
error_pca_hd = evaluation.get_generalization_error(pca_reduced_hd, hd_dataset, hd_labels)
error_lle_hd = evaluation.get_generalization_error(lle_reduced_hd, hd_dataset, hd_labels)
print("HD dataset(reduced by PCA) is: " + str(error_pca_hd))
print("HD dataset(reduced by LLE) is: " + str(error_lle_hd))
# **************End of evaluation of Generalization errors on artificial datasets



print("Now evaluating Trustworthiness of each technique on MNIST dataset, this is going to take a while\n")
trust_pca_MNIST = evaluation.get_trustworthiness(pca_reduced_MNIST_images, MNIST_images, 12)
trust_lle_MNIST = evaluation.get_trustworthiness(lle_reduced_MNIST_images, MNIST_images, 12)
print("MNIST dataset(reduced by PCA) is: " + str(trust_pca_MNIST))
print("MNIST dataset(reduced by LLE) is: " + str(trust_lle_MNIST))

print("Now evaluating Continuity of each technique on MNIST dataset, this is going to take a while\n")
continuity_pca_MNIST = evaluation.get_continuity(pca_reduced_MNIST_images, MNIST_images, 12)
continuity_lle_MNIST = evaluation.get_continuity(lle_reduced_MNIST_images, MNIST_images, 12)
print("MNIST dataset(reduced by PCA) is: " + str(continuity_pca_MNIST))
print("MNIST dataset(reduced by LLE) is: " + str(continuity_lle_MNIST))

print("Now evaluating generalization errors of each technique on MNIST dataset, this is going to take a while\n")
error_pca_MNIST = evaluation.get_generalization_error(pca_reduced_MNIST_images, MNIST_images, MNIST_labels)
error_lle_MNIST = evaluation.get_generalization_error(lle_reduced_MNIST_images, MNIST_images, MNIST_labels)
print("MNIST dataset(reduced by PCA) is: " + str(error_pca_MNIST))
print("MNIST dataset(reduced by LLE) is: " + str(error_lle_MNIST))

localtime = time.asctime( time.localtime(time.time()) )
print ("Local current time :", localtime)
