import time
import Generation_Stage
import Evaluate_lle
import Evaluate_pca
import Evaluation_Stage
print("**********************************************************************")
print("Hello. This is CSE569 Project Demo, produced by Haisi Yi and Zheng Xia")
print("**********************************************************************\n\n")

while True:
    option = input("\nPlease specify the task to perform:\n"
                   "1: Generate five artificial dataset and read MNIST_images dataset\n"
                   "2: Perform PCA to all artificial dataset and MNIST_images dataset\n"
                   "3: Perform LLE 11 * 6 times, using parameter k = 5, 6, ..., 15, to all artificial dataset and MNIST_images dataset\n"
                   "4: Do task 1, task 2 and task 3. This task will take about 20 min\n"
                   "5: Evaluate the data produced by PCA. This task will take about 40 min.\n"
                   "6: Evaluate the data produced by LLE. This task will take about 40 min.\n"
                   "7: Run everything. This will take about 8 hours.\n"
                   "0: Exit this Demo\n")
    option = int(option)

    if option == 1:
        Generation_Stage.generate_original_datasets()

    elif option == 2:
        Generation_Stage.perform_pca_to_original_datasets()

    elif option == 3:
        Generation_Stage.perform_lle_to_orginal_datasets()

    elif option == 4:
        Generation_Stage.run()

    elif option == 5:
        Evaluate_pca.run()

    elif option == 6:
        Evaluate_lle.run()

    elif option == 7:
        Evaluation_Stage.run()
        break
    elif option == 0:
        break
    else:
        print("Invalid option, try again")





