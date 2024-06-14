import numpy as np
import cv2

# Load your data
sfm03_img = cv2.imread('./Data/sfm03.jpg')
sfm03_kp = np.load('./two_view_recon_info/sfm03_keypoints.npy')
sfm03_des = np.load('./two_view_recon_info/sfm03_descriptors.npy')
sfm03_m = np.load('./two_view_recon_info/sfm03_matched_idx.npy')
sfm03_cp = np.load('./two_view_recon_info/sfm03_camera_pose.npy')

sfm04_img = cv2.imread('./Data/sfm04.jpg')
sfm04_kp  = np.load('./two_view_recon_info/sfm04_keypoints.npy')
sfm04_des = np.load('./two_view_recon_info/sfm04_descriptors.npy')
sfm04_m   = np.load('./two_view_recon_info/sfm04_matched_idx.npy')
sfm04_cp  = np.load('./two_view_recon_info/sfm04_camera_pose.npy')

points_3d = np.load('./two_view_recon_info/3D_points.npy')
inlier_indices = np.load('./two_view_recon_info/inlinear.npy')

# 3번의 이미지와 4번의 이미지의 keypoint를 비교하여 매칭을 함
# 그 매칭된 keypoint index임
print(sfm03_m, len(sfm03_m))
print(sfm04_m, len(sfm04_m))

# sfm03_m, sfm04_m 의 index와, 3d_points를 연결해줌
# points_3d 0번은 sfm03_m 에서 inlier_indices[0] 번째의 keypoints에 해당하는 점임 
print(inlier_indices, len(inlier_indices))

idx = inlier_indices[1]
print(sfm03_kp[sfm03_m[idx]])
print(sfm04_kp[sfm04_m[idx]])

print(sfm03_des[sfm03_m[idx]])
print(sfm04_des[sfm04_m[idx]])

# image 의 keypoint x, y임
#print(sfm03_kp, len(sfm03_kp))
#print(sfm04_kp, len(sfm04_kp))