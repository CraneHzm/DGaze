# Copyright (c) 2019/7/13 Hu Zhiming jimmyhu@pku.edu.cn All Rights Reserved.

# process files and directories.


#################### Libs ####################
import os
import shutil
import time


# remove a directory
def RemoveDir(dirName):
	if os.path.exists(dirName):
		shutil.rmtree(dirName)
	else:
		print("Invalid Directory Path!")


# remake a directory
def RemakeDir(dirName):
	if os.path.exists(dirName):
		shutil.rmtree(dirName)
		os.makedirs(dirName)
	else:
		os.makedirs(dirName)
	
# calculate the number of lines in a file
def FileLines(fileName):
	if os.path.exists(fileName):
		with open(fileName, 'r') as fr:
			return len(fr.readlines())
	else:
		print("Invalid File Path!")
		return 0
	
# make a directory if it does not exist.
def MakeDir(dirName):
	if os.path.exists(dirName):
		print("Directory "+ dirName + " already exists.")
	else:
		os.makedirs(dirName)
	
	
	
if __name__ == "__main__":
	dirName = "test"
	RemakeDir(dirName)
	time.sleep(3)
	MakeDir(dirName)
	RemoveDir(dirName)
	time.sleep(3)
	MakeDir(dirName)
	#print(FileLines('233.txt'))