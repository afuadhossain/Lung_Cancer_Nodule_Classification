import shutil
import os
import csv
import numpy as np

SUBSET_NUMBER = 0

current_directory = './patches/subset' + str(SUBSET_NUMBER) + '/benign/'
cancer_directory = './patches/subset' + str(SUBSET_NUMBER) + '/cancer/'


def readCSV(filename):
	lines = []
	with open(filename, "rb") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			lines.append(line)
	return lines

cands = readCSV('./data/candidates.csv')


def find_cancer():
	cancer_set = set()
	for cand in cands:
		if (cand[4] == '1'):
			worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
			candidate_name = 'patch_' + str(worldCoord[0]) + '_' + str(worldCoord[1]) + '_' + str(worldCoord[2]) + '.png'
			cancer_set.add(candidate_name)
	
	return cancer_set

def main():

	#Finds Images that have cancer and moves them
	cancer_set = find_cancer()
	total_count = 0
	cancer_count = 0
	for filename in os.listdir(current_directory):
		#print(filename)
		total_count += 1
		if filename in cancer_set:
			shutil.move(current_directory + filename, cancer_directory + str(SUBSET_NUMBER) + '_' + filename[:-4] + '_C' + '.png')
			cancer_count+=1
		else:
			shutil.move(current_directory + filename, current_directory + str(SUBSET_NUMBER) + '_' + filename)

	print ('Total Count: {}       Cancer Count: {}'.format(total_count, cancer_count))



if __name__ == '__main__':
	main()