import time
import Generation_Stage

print("**********************************************************************")
print("Hello. This is CSE569 Project Demo, produced by Haisi Yi and Zheng Xia")
print("**********************************************************************\n\n")

localtime = time.asctime( time.localtime(time.time()))
print("Local current time :", localtime)


Generation_Stage.run()



