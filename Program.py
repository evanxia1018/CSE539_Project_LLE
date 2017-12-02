import time
import Generation_Stage
import Evaluate_lle
import Evaluate_pca
import Evaluation_Stage
print("**********************************************************************")
print("Hello. This is CSE569 Project Demo, produced by Haisi Yi and Zheng Xia")
print("**********************************************************************\n\n")

while True:
    option = input("Please specify the task to perform:\n"
                   "1: Generate all the data for evaluation. Existing data, if any, will be overwritten. This task will take about 20 min\n"
                   "2: Evaluate the data produced by PCA. This task will take about 40 min.\n"
                   "3: Evaluate the data produced by LLE. This task will take about 40 min.\n"
                   "4: Run everything. This will take about 8 hours.\n"
                   "0: Exit this Demo\n")
    option = int(option)
    if option == 1:
        Generation_Stage.run()  # Uncomment this if and only if need to generate datasets
        break
    elif option == 2:
        Evaluate_pca.run()
        break
    elif option == 3:
        Evaluate_lle.run()
        break
    elif option == 4:
        Evaluation_Stage.run()
        break
    elif option == 0:
        break
    else:
        print("Invalid option, try again")





