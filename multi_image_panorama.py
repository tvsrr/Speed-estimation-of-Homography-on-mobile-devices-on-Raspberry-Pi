import sys
import cv2
import numpy as np
import time
import homography_ransac as hr
import time

def main():
	# Get input set of images
	t1 = time.time()
	img1 = cv2.imread(sys.argv[3])
	img2 = cv2.imread(sys.argv[2])
	img3 = cv2.imread(sys.argv[1])

	img1 = cv2.resize(img1, (640, 480), interpolation = cv2.INTER_CUBIC)
	img2 = cv2.resize(img2, (640, 480), interpolation = cv2.INTER_CUBIC)
	img3 = cv2.resize(img3, (640, 480), interpolation = cv2.INTER_CUBIC)

	if sys.argv[4] == 'sift':
		M21, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_sift_homography(img2, img1)
	elif sys.argv[4] == 'surf':
		M21, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_surf_homography(img2, img1)
	else:
		print 'invalid argument... last argument must be \'sift\' or \'surf\' '
		exit()
	result_img1, trans_point1 = hr.get_stitched_image(img1, img2, M21)
	
#	for i in ranges(3)
	if sys.argv[4] == 'sift':
		M32, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_sift_homography(img3, img2)
	elif sys.argv[4] == 'surf':
		M32, img1_pts, img2_pts, new_matches, matchesMask =  hr.get_surf_homography(img3, img2)
	else:
		print 'invalid argument... last argument must be \'sift\' or \'surf\' '
		exit()
	M31 = M21.dot(M32)
	result_img2, trans_point2 = hr.get_stitched_image(img1, img3, M31)
	#print result_img2.shape, trans_point2 

	result_img2[trans_point2[1]- trans_point1[1]:result_img1.shape[0] + trans_point2[1]- trans_point1[1], trans_point2[0]-trans_point1[0]:result_img1.shape[1] + trans_point2[0]-trans_point1[0]][result_img1 != 0] = result_img1[result_img1 != 0]
	cv2.imwrite('panorama.png', result_img2)
	print 'total time taken', time.time()-t1
#	cv2.imshow('panorama', result_img2)
#	cv2.waitKey(0)

# Call main function
if __name__=='__main__':
	main()











