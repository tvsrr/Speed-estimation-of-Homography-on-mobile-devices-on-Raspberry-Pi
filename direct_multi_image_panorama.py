import sys
import cv2
import numpy as np
import time
import homography_ransac as hr
import time
import os
from glob import glob

def main():
	file_list = sorted(os.listdir('./images/h/'))
#	print file_list
	base_img_path = './images/h/'+str(file_list[0])
	base_img = cv2.imread(base_img_path)
	base_img = cv2.resize(base_img, (640, 480), interpolation = cv2.INTER_CUBIC)
#	cv2.imshow('base_img', base_img)
#	cv2.waitKey(0)
	
	M_prev = np.identity(3)
	trans_points = []
	for i in range(6):#len(file_list)-1):
		print i
		t = time.time()
		img1_path = './images/h/'+str(file_list[i])
		img1 = cv2.imread(img1_path)
		img1 = cv2.resize(img1, (640, 480), interpolation = cv2.INTER_CUBIC)
		img2_path = './images/h/'+str(file_list[i+1])
		img2 = cv2.imread(img2_path)
		img2 = cv2.resize(img2, (640, 480), interpolation = cv2.INTER_CUBIC)

		M21, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_sift_homography(img2, img1)
		M = M_prev.dot(M21)
		result_img, trans_point = hr.get_stitched_image(base_img, img2, M)
		result_img_path = './results/test'+str(i)+'.png'
		cv2.imwrite(result_img_path, result_img)
		M_prev = M
		trans_points.append(trans_point)
		print 'stitching time ', time.time()-t

	canvas = np.zeros((19*base_img.shape[0], 19*base_img.shape[1], base_img.shape[2]))
	print trans_points

	t = time.time()
	result_list = sorted(glob('./results/*.png'))
	for i in range(len(result_list)):
		print i
		stitched_img = cv2.imread(result_list[i])
		row_ref, col_ref, d_ref = base_img.shape
		row_sti, col_sti, d_sti = stitched_img.shape
		row = int(trans_points[i][1])
		col = int(trans_points[i][0])
		canvas[10*row_ref-row:10*row_ref-row+row_sti, 10*col_ref-col:10*col_ref-col+col_sti, :][stitched_img != 0] =stitched_img[stitched_img != 0]
##############################################
	for l1 in range(canvas.shape[0]):
		temp_ary = canvas[l1, :,:]
		if np.count_nonzero(temp_ary) != 0:
			break
	print l1
################################################
	for l2 in range(canvas.shape[1]):
		temp_ary = canvas[:, l2,:]
		if np.count_nonzero(temp_ary) != 0:
			break
	print l2
####################################################
	for l3 in range(canvas.shape[0]-1,0,-1):
		temp_ary = canvas[l3, :,:]
		if np.count_nonzero(temp_ary) != 0:
			break
	print l3
######################################################
	for l4 in range(canvas.shape[1]-1,0,-1):
		temp_ary = canvas[:, l4,:]
		if np.count_nonzero(temp_ary) != 0:
			break
	print l4
###################################################
	panorama = canvas[l1:l3, l2:l4, :]
	cv2.imwrite('panorama.png', panorama)
	print 'canvas time ', time.time()-t

	exit()



# Call main function
if __name__=='__main__':
	main()











