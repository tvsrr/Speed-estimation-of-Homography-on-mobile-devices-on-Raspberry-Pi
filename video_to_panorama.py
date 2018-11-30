import sys
import cv2
import numpy as np
import time
import homography_ransac as hr
import time
import os
from glob import glob

def main():

	cap = cv2.VideoCapture('./images/h_640.avi')
	ret, base_img = cap.read()
#	cv2.imshow('base', base_img)
#	cv2.waitKey(0)

	img1 = np.copy(base_img)
	count = 0	
	M_prev = np.identity(3)
	trans_points = []
	rate = 15
	while(cap.isOpened()):
		count = count+1
#		print count
		ret, img2 = cap.read()
		if ret == False:
			break
		cv2.imshow('frame',img2)
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break
		if (count % rate == 0) & (count < 350) & (ret != False):

			M21, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_surf_homography(img2, img1)
			M = M_prev.dot(M21)
			result_img, trans_point = hr.get_stitched_image(base_img, img2, M)
			result_img_path = './results_vid/test10'+str(int(count/rate))+'.png'
			cv2.imwrite(result_img_path, result_img)
			M_prev = M
			img1 = np.copy(img2)
			trans_points.append(trans_point)
		else:
			pass
		
	cap.release()
	cv2.destroyAllWindows()
	
	print 'Creating panorama...'		
	canvas = np.zeros((19*base_img.shape[0], 19*base_img.shape[1], base_img.shape[2]))
	print trans_points

	t = time.time()
	result_list = sorted(glob('./results_vid/*.png'))
	result_list.sort(key=lambda f: int(filter(str.isdigit, f)))
	print result_list
	for i in range(len(result_list)):
		print i
		stitched_img = cv2.imread(result_list[i])
		row_ref, col_ref, d_ref = base_img.shape
		row_sti, col_sti, d_sti = stitched_img.shape
		row = int(trans_points[i][1])
		col = int(trans_points[i][0])
		canvas[10*row_ref-row:10*row_ref-row+row_sti, 10*col_ref-col:10*col_ref-col+col_sti, :][stitched_img != 0] =stitched_img[stitched_img != 0]
		
#		cv2.imshow('frame',frame)
#		if cv2.waitKey(30) & 0xFF == ord('q'):
#			break
	
#	exit()	
	
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











