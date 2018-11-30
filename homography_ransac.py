import sys
import cv2
import numpy as np
import time

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):


        # Get width and height of input images
        w1,h1 = img1.shape[:2]
        w2,h2 = img2.shape[:2]

        # Get the canvas dimesions
        img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
        img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


        # Get relative perspective of second image
        img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

        # Resulting dimensions
        result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

        # Getting images together
        # Calculate dimensions of match points
        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        # Create output array after affine transformation
        transform_dist = [-x_min,-y_min]
        transform_array = np.array([[1, 0, transform_dist[0]],
                                                                [0, 1, transform_dist[1]],
                                                                [0,0,1]])

        # Warp images to get the resulting image
        result_img = cv2.warpPerspective(img2, transform_array.dot(M), (x_max-x_min, y_max-y_min),cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT)

        result_img[transform_dist[1]:w1+transform_dist[1], transform_dist[0]:h1+transform_dist[0]] = img1
#	cv2.circle(result_img,(transform_dist[0], transform_dist[1]), 5, (0,0,255), 3)
#	print transform_dist
#	cv2.imwrite('result.png', result_img)
#	cv2.imshow('Frame', result_img)
#	cv2.waitKey(0)
        # Return the result
        return result_img, transform_dist

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):

	# Initialize SIFT 
#	sift = cv2.ORB_create()
	sift = cv2.xfeatures2d.SIFT_create()

	# Extract keypoints and descriptors
	k1, d1 = sift.detectAndCompute(img1, None)
	k2, d2 = sift.detectAndCompute(img2, None)

	# Bruteforce matcher on the descriptors
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(d1,d2, k=2)

	new_matches = []
	for m1,m2 in matches:
		new_matches.append(m1)

	# Mimnum number of matches
	min_matches = 8
	if len(new_matches) > min_matches:
		
		# Array to store matching points
		img1_pts = []
		img2_pts = []

		# Add matching points to array
		for match in new_matches:
			img1_pts.append(k1[match.queryIdx].pt)
			img2_pts.append(k2[match.trainIdx].pt)
		img1_pts = np.float32(img1_pts).reshape(-1,1,2)
		img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
		# Compute homography matrix
		M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		return M, k1, k2, new_matches, matchesMask
	else:
		print 'Error: Not enough matches'
		exit()
# Find SURF Homography Matrix
def get_surf_homography(img1, img2):

	surf = cv2.xfeatures2d.SURF_create()

	# Extract keypoints and descriptors
	k1, d1 = surf.detectAndCompute(img1, None)
	k2, d2 = surf.detectAndCompute(img2, None)

	# Bruteforce matcher on the descriptors
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(d1,d2, k=2)

	new_matches = []
	for m1,m2 in matches:
		new_matches.append(m1)

	# Mimnum number of matches
	min_matches = 8
	if len(new_matches) > min_matches:
		
		# Array to store matching points
		img1_pts = []
		img2_pts = []

		# Add matching points to array
		for match in new_matches:
			img1_pts.append(k1[match.queryIdx].pt)
			img2_pts.append(k2[match.trainIdx].pt)
		img1_pts = np.float32(img1_pts).reshape(-1,1,2)
		img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
		# Compute homography matrix
		M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		return M, k1, k2, new_matches, matchesMask
	else:
		print 'Error: Not enough matches'
		exit()


def show_matching_points(img1, img2, M, img1_pts, img2_pts, matchesMask, new_matches):
	
	np.savetxt('M_speed.txt', M)	

	drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
	match_img = cv2.drawMatches(img1, img1_pts, img2, img2_pts, new_matches, None, **drawParameters)

	cv2.imshow ('Result', match_img)
#	cv2.imwrite('homography.png', match_img)
	cv2.waitKey()

def main():
	# Get input set of images
	img1 = cv2.imread(sys.argv[1])
	img2 = cv2.imread(sys.argv[2])
	save_as = sys.argv[3]

	if img1.shape[0] != 640:
		img1 = cv2.resize(img1, (640, 480), interpolation = cv2.INTER_CUBIC)
		img2 = cv2.resize(img2, (640, 480), interpolation = cv2.INTER_CUBIC)
	# Use SIFT to find keypoints and return homography matrix
	M, img1_pts, img2_pts, new_matches, matchesMask =  get_sift_homography(img1, img2)
#	show_matching_points(img1, img2, M, img1_pts, img2_pts, matchesMask, new_matches)
	result_img, pt = get_stitched_image(img2, img1, M)
	cv2.imwrite(save_as, result_img)

# Call main function
if __name__=='__main__':
	main()











