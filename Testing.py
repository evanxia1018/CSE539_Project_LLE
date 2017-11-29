import Dataset_Generator as dg
import Evaluation as eval
import Plot_Graph as ploter

hd_dataset = dg.get_hd_dataset(5000)
reduced_hd = eval.pca_dim_reduction(hd_dataset, 3)
ploter.plot3D(reduced_hd)

broken_swiss_roll_dataset = dg.get_broken_swiss_roll_dataset(5000)
ploter.plot3D(broken_swiss_roll_dataset)
reduced_broken_swiss = eval.pca_dim_reduction(broken_swiss_roll_dataset, 2)
ploter.plot2D(reduced_broken_swiss)

broken_helix_dataset = dg.get_helix_dataset(5000)
ploter.plot3D(broken_helix_dataset)
reduced_helix = eval.pca_dim_reduction(broken_helix_dataset, 2)
ploter.plot2D(reduced_helix)

swiss_roll_dataset = dg.get_swiss_roll_dataset(5000)
ploter.plot3D(swiss_roll_dataset)
reduced_swiss = eval.pca_dim_reduction(swiss_roll_dataset, 2)
ploter.plot2D(reduced_swiss)

twin_peaks_dataset = dg.get_twin_peaks(5000)
ploter.plot3D(twin_peaks_dataset)
reduced_twin_peaks = eval.pca_dim_reduction(twin_peaks_dataset, 2)
ploter.plot2D(reduced_twin_peaks)

# ***********************************scripts to evaluate Trust

# Swiss roll
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_swiss_roll_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)

# Helix
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_helix_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 1)
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)

# Twin peaks
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_twin_peaks(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)

# Broken Swiss
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_broken_swiss_roll_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)

# HD
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_hd_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 5)
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)


# ***********************************scripts to evaluate Continuity

# Swiss roll
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_swiss_roll_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# Helix
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_helix_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 1)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# Twin peaks
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_twin_peaks(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# Broken Swiss
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_broken_swiss_roll_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# HD
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_hd_dataset(5000)
reduced_dataset = eval.pca_dim_reduction(dataset, 5)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)


# ***********************************scripts to test label_generation and generalization error
import Dataset_Generator as dg
import Evaluation as eval
dataset = dg.get_broken_swiss_roll_dataset(5000)
labels = eval.get_artificial_dataset_labels(dataset)
reduced_dataset = eval.pca_dim_reduction(dataset, 2)
error = eval.get_generalization_error(reduced_dataset, dataset)

# ***********************************scripts to read MNIST
from mnist import MNIST
mndata = MNIST('/Users/evanxia/Dropbox/CSE569/MNIST_dataset')
images, labels = mndata.load_training()


# ***********************************scripts to test Trustworthiness and continuity in Natural dataset using LLE

# MNIST
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset, labels = eval.get_natural_dataset_samples()
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 20)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# ***********************************scripts to test Trustworthiness and continuity in artificial dataset using LLE
# Swiss roll
import Dataset_Generator as dg
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset = dg.get_swiss_roll_dataset(5000)
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 2)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)


# Helix
import Dataset_Generator as dg
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset = dg.get_helix_dataset(5000)
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 1)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# Twin peaks
import Dataset_Generator as dg
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset = dg.get_twin_peaks(5000)
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 2)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# Broken Swiss
import Dataset_Generator as dg
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset = dg.get_broken_swiss_roll_dataset(5000)
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 2)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)

# HD
import Dataset_Generator as dg
import Evaluation as eval
import MyLLE as lle
import numpy as np
dataset = dg.get_hd_dataset(5000)
reduced_dataset, error = lle.locally_linear_embedding(np.array(dataset, np.float64), 5, 5)
reduced_dataset = reduced_dataset.tolist()
trust = eval.get_trustworthiness(reduced_dataset, dataset, 12)
continuity = eval.get_continuity(reduced_dataset, dataset, 12)
