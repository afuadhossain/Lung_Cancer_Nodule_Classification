import SimpleITK as sitk
import csv
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import winsound



'''
#save all images in one mhd file as png
img_path = './1.3.6.1.4.1.14519.5.2.1.6279.6001.272042302501586336192628818865.mhd'
itkimage = sitk.ReadImage(img_path)
numpyImage = sitk.GetArrayFromImage(itkimage)
for index in range(numpyImage.shape[0]):
	Image.fromarray(numpyImage[index]).convert('L').save(str(index) + '.png')
	
'''

def Alarm():
	#plays a sound when script ends
	duration = 1000  # millisecond
	freq = 440  # Hz
	winsound.Beep(freq, duration)

#Returns all .mhd file path in a list
def list_files(directory):

	list = []
	for filename in os.listdir(directory):
		if filename.endswith(".mhd"):
			fp = os.path.join(directory, filename)
			list.append(fp)
	return list

def load_itk_image(filename):

	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)

	numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
	numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

	return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
	lines = []
	with open(filename, "rb") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			lines.append(line)
	return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
	stretchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = stretchedVoxelCoord / spacing
	return voxelCoord


def normalizePlanes(npzarray):
	maxHU = 400.
	minHU = -1000.

	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray > 1] = 1.
	npzarray[npzarray < 0] = 0.
	return npzarray


def main():
	#subset to convert
	s_number = 9
	subset = 'subset' + str(s_number) + '/'


	directory_path = './data/' + subset  # path for mhd file
	mhd_list = list_files(directory= directory_path)

	#create the subset directory
	os.mkdir('./patches/' + subset)

	outputDir = './patches/' + subset + 'benign/'
	os.mkdir(outputDir)

	outputDir_cancer = './patches/' + subset + 'cancer/'
	os.mkdir(outputDir_cancer)	


	for mhd_file in mhd_list:
		img_path = mhd_file
		# img_path = './1.3.6.1.4.1.14519.5.2.1.6279.6001.272042302501586336192628818865.mhd'
		cand_path = './data/candidates.csv'
		numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
		# print numpyImage.shape
		# print numpyOrigin
		# print numpySpacing
		print(img_path.replace(".mhd",""))

		cands = readCSV(cand_path)
		for cand in cands:
			# get candidates
			# Extract and visualize patch for each candidate in the list
			if cand[0] == img_path.replace(".mhd","").replace(directory_path, ""):

				label = cand[4]
				print(label)

				print('matched candidate and path. Label = {}'.format(label))

				worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
				voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
				
				slice = int(voxelCoord[0])

				coord_start = [0, 0, 0]
				coord_end = [0, 512, 512]	

				'''	
				coord_start = [0, 0, 0]
				coord_end = [0, 0, 0]
				voxelWidth = 65
				coord_start[1] = int(voxelCoord[1] - voxelWidth / 2.0)
				coord_end[1] = int(voxelCoord[1] + voxelWidth / 2.0)
				coord_start[2] = int(voxelCoord[2] - voxelWidth / 2.0)
				coord_end[2] = int(voxelCoord[2] + voxelWidth / 2.0)
				'''

				patch = numpyImage[slice, coord_start[1]:coord_end[1], coord_start[2]:coord_end[2]]

				patch = normalizePlanes(patch)


				#plt.imshow(patch, cmap='gray')
				#plt.show()
				#save the file as png format

				if (label == '1'):	#if label = 1, nodule is cancerous. Save to cancerous folder
					Image.fromarray(patch * 255).convert('L').save(os.path.join(outputDir_cancer,
																				str(s_number) +'_patch_' + str(worldCoord[0]) + '_' + str(
																				worldCoord[1]) + '_' + str(
																				worldCoord[2]) + '_C.png'))
					print('File Saved as 1')
				else:				#if label = 0, nodule is benign. Save to benign folder
					Image.fromarray(patch * 255).convert('L').save(os.path.join(outputDir,
																				str(s_number) +'_patch_' + str(worldCoord[0]) + '_' + str(
																					worldCoord[1]) + '_' + str(
																					worldCoord[2]) + '.png'))
					print('File Saved as 0')

	print('All files save in ' + outputDir)
	Alarm()


main()



