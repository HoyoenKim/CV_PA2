import random
import os
from warnings import catch_warnings

import cv2
import numpy as np
import matlab.engine
import pickle

debug = False
if debug:
    image_filename_to_match = [
        'sfm02.jpg',
        'sfm05.jpg',
    ]
    image_filename_done_match = [
        'sfm03.jpg',
        'sfm04.jpg',
    ]
    image_filename_all = [
        'sfm02.jpg',
        'sfm03.jpg',
        'sfm04.jpg',
        'sfm05.jpg',
    ]
    image_filename_sequence = [
        'sfm03.jpg',
        'sfm02.jpg',
        'sfm05.jpg',
        'sfm04.jpg',
    ]
    left_images_index = 1
    right_images_index = 2
else:
    image_filename_to_match = [
        'sfm00.jpg',
        'sfm01.jpg',
        'sfm02.jpg',
        'sfm05.jpg',
        'sfm06.jpg',
        'sfm07.jpg',
        'sfm08.jpg',
        'sfm09.jpg',
        'sfm10.jpg',
        'sfm11.jpg',
        'sfm12.jpg',
        'sfm13.jpg',
        'sfm14.jpg',
    ]
    image_filename_done_match = [
        'sfm03.jpg',
        'sfm04.jpg',
    ]
    image_filename_all = [
        'sfm00.jpg',
        'sfm01.jpg',
        'sfm02.jpg',
        'sfm03.jpg',
        'sfm04.jpg',
        'sfm05.jpg',
        'sfm06.jpg',
        'sfm07.jpg',
        'sfm08.jpg',
        'sfm09.jpg',
        'sfm10.jpg',
        'sfm11.jpg',
        'sfm12.jpg',
        'sfm13.jpg',
        'sfm14.jpg',
    ]
    image_filename_sequence = [
        'sfm03.jpg',
        'sfm02.jpg',
        'sfm01.jpg',
        'sfm00.jpg',
        'sfm14.jpg',
        'sfm13.jpg',
        'sfm12.jpg',
        'sfm11.jpg',
        'sfm10.jpg',
        'sfm09.jpg',
        'sfm08.jpg',
        'sfm07.jpg',
        'sfm06.jpg',
        'sfm05.jpg',
        'sfm04.jpg',
    ]
    left_images_index = 1   
    right_images_index = 13


def save_images():
    global image_filename_to_match
    global image_filename_done_match

    image_to_match = {}
    image_root_dir = './Data'
    sift = cv2.SIFT_create()
    for image_filename in image_filename_to_match:
        image_path = os.path.join(image_root_dir, image_filename)
        image = cv2.imread(image_path)
        kp, des = sift.detectAndCompute(image, None)
        image_to_match[image_filename] = {
            'img' : image,
            'kp'  : kp,
            'des' : des
        }

    save_dir = './to_image_info'
    for filename in image_to_match:
        data = image_to_match[filename]
        name = filename.split('.')[0]
        kps = np.array([np.array(kp.pt) for kp in data['kp']])
        np.save(os.path.join(save_dir, f'{name}_keypoints.npy'), kps)
        np.save(os.path.join(save_dir, f'{name}_descriptors.npy'), data['des'])

def load_images():
    global image_filename_to_match
    global image_filename_done_match

    image_to_match = {}
    for image_filename in image_filename_to_match:
        filename = image_filename.split('.')[0]
        image_to_match[image_filename] = {
            'kp'  : np.load(f'./to_image_info/{filename}_keypoints.npy'),
            'des' : np.load(f'./to_image_info/{filename}_descriptors.npy')
        }

    image_done_match = {}
    for image_filename in image_filename_done_match:
        filename = image_filename.split('.')[0]
        image_done_match[image_filename] = {
            'kp'  : np.load(f'./two_view_recon_info/{filename}_keypoints.npy'),
            'des' : np.load(f'./two_view_recon_info/{filename}_descriptors.npy'),
            'm'   : np.load(f'./two_view_recon_info/{filename}_matched_idx.npy'),
            'cp'  : np.load(f'./two_view_recon_info/{filename}_camera_pose.npy')
        }

    return image_to_match, image_done_match

def load_3d_points(image_done_match):
    # 3D points
    points_3d = np.load('./two_view_recon_info/3D_points.npy')
    inlier_indices = np.load('./two_view_recon_info/inlinear.npy')

    # Get the 3d points description
    done_image_info = image_done_match['sfm03.jpg']
    points_3d_des = np.array([done_image_info['des'][done_image_info['m'][inlier_index]] for inlier_index in inlier_indices])

    uv_info = {}
    for image_filename in image_filename_done_match: 
        done_image_info = image_done_match[image_filename]
        uv_info[image_filename] = {
            'kp': np.array([done_image_info['kp'][done_image_info['m'][inlier_index]] for inlier_index in inlier_indices]),
            '3p_index': np.array(list(range(0, len(points_3d))))
        }
    return points_3d, points_3d_des, uv_info

def choose_to_image_all(image_to_match, image_to_match_done, points_3d_des):
    best_match = []
    best_to_image_name = None
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for to_image_name in image_to_match:
        if to_image_name in image_to_match_done:
            continue

        to_image_des = image_to_match[to_image_name]['des']
        matches = bf.knnMatch(points_3d_des, to_image_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if  len(good_matches) > len(best_match):
            best_match = good_matches
            best_to_image_name = to_image_name

    return best_match, best_to_image_name

def choose_to_image_sequence(points_3d_des, image_to_match):
    global left_images_index
    global right_images_index

    best_match = []
    best_to_image_name = None    
    image_d = None
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    if left_images_index <= len(image_filename_all) - 1 and right_images_index >= 0:
        left_image_name = image_filename_sequence[left_images_index]
        left_des = image_to_match[left_image_name]['des']
        left_matches = bf.knnMatch(points_3d_des, left_des, k=2)
        left_matches = [m for m, n in left_matches if m.distance < 0.75 * n.distance]
        
        right_image_name = image_filename_sequence[right_images_index]
        right_des = image_to_match[right_image_name]['des']
        right_matches = bf.knnMatch(points_3d_des, right_des, k=2)
        right_matches = [m for m, n in right_matches if m.distance < 0.75 * n.distance]
        
        if len(left_matches) >= len(right_matches):
            image_d = 0
            best_match = left_matches
            best_to_image_name = left_image_name
        else:
            image_d = 1
            best_match = right_matches
            best_to_image_name = right_image_name
    
    elif right_images_index < len(image_filename_all) // 2:
        left_image_name = image_filename_sequence[left_images_index]
        left_des = image_to_match[left_image_name]['des']
        left_matches = bf.knnMatch(points_3d_des, left_des, k=2)
        left_matches = [m for m, n in left_matches if m.distance < 0.75 * n.distance]
        
        image_d = 0
        best_match = left_matches
        best_to_image_name = left_image_name
    else:
        right_image_name = image_filename_sequence[right_images_index]
        right_des = image_to_match[right_image_name]['des']
        right_matches = bf.knnMatch(points_3d_des, right_des, k=2)
        right_matches = [m for m, n in right_matches if m.distance < 0.75 * n.distance]
        
        image_d = 1
        best_match = right_matches
        best_to_image_name = right_image_name
    
    return best_match, best_to_image_name, image_d

intrinsic_matrix = np.array([
    [3451.5, 0.0, 2312],
    [0.0, 3451.5, 1734],
    [0.0, 0.0, 1.0]
])
def RANSAC(best_match, to_image_info, points_3d):
    ransac_threshold = 1
    ransac_iterations = 1000
    
    max_inliers = 0
    best_camera_pose = None
    for _ in range(ransac_iterations):
        random_matches = random.sample(best_match, 3)
        random_image_points = np.array([to_image_info['kp'][match.trainIdx] for match in random_matches], dtype=np.float32)
        random_object_points = np.array([points_3d[match.queryIdx] for match in random_matches], dtype=np.float32)
        
        _, rotation_vectors, translation_vectors = cv2.solveP3P(random_object_points, random_image_points, intrinsic_matrix, None, flags=cv2.SOLVEPNP_P3P)
        for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            camera_pose = np.hstack((rotation_matrix, translation_vector))
            project_matrix = np.dot(intrinsic_matrix, camera_pose)
            
            inliers = 0
            for match in best_match:
                image_point = np.array(to_image_info['kp'][match.trainIdx], dtype=np.float32).reshape(2, 1)

                object_point = np.array(points_3d[match.queryIdx], dtype=np.float32).reshape(3, 1)
                object_point = np.vstack((object_point, [1]))
                
                projected_point = np.dot(project_matrix, object_point)
                projected_point = (projected_point[:2] / projected_point[2])
                
                distance = np.linalg.norm(projected_point - image_point)
                if distance < ransac_threshold:
                    inliers += 1
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_camera_pose = camera_pose

    return best_camera_pose

def choose_done_image_all(to_image_info, image_done_match):
    best_done_image_name = None
    best_match = []
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for done_image_name in image_done_match:
        done_image_des = image_done_match[done_image_name]['des']
        matches = bf.knnMatch(to_image_info['des'], done_image_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if  len(good_matches) > len(best_match):
            best_match = good_matches
            best_done_image_name = done_image_name

    return best_match, best_done_image_name

def choose_done_image_sequence(to_image_info, image_done_match, image_d):
    global left_images_index
    global right_images_index

    if image_d == 0:
        left_index = left_images_index - 1
        right_index = left_images_index + 1
        left_images_index += 1
    elif image_d == 1:
        left_index = right_images_index - 1
        right_index = right_images_index + 1
        right_images_index -= 1
    
    if left_index < 0:
        left_index += len(image_filename_all)
    if right_index >= len(image_filename_all):
        right_index = 0

    bf = cv2.BFMatcher(cv2.NORM_L2)

    left_done_image_name = image_filename_sequence[left_index]
    right_done_image_name = image_filename_sequence[right_index]

    left_match = []
    if left_done_image_name in image_done_match:
        left_match = bf.knnMatch(to_image_info['des'], image_done_match[left_done_image_name]['des'], k=2)
        left_match = [m for m, n in left_match if m.distance < 0.75 * n.distance]
    
    right_match = []
    if right_done_image_name in image_done_match:
        right_match = bf.knnMatch(to_image_info['des'], image_done_match[right_done_image_name]['des'], k=2)
        right_match = [m for m, n in right_match if m.distance < 0.75 * n.distance]
    
    
    if len(left_match) > len(right_match):
        best_match = left_match
        best_done_image_name = left_done_image_name
    else:
        best_match = right_match
        best_done_image_name = right_done_image_name
    
    return best_match, best_done_image_name

def get_normalized_points(image_done_match, best_to_image_name, best_done_image_name, best_match):
    global intrinsic_matrix
    inverse_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    
    kp1 = image_done_match[best_to_image_name]['kp']
    kp2 = image_done_match[best_done_image_name]['kp']

    nip1 = []
    nip2 = []
    for match in best_match:
        ip1 = kp1[match.queryIdx]
        if isinstance(ip1, cv2.KeyPoint):
            ip1 = np.array(ip1.pt)
        ip1 = np.vstack((ip1.reshape(2, 1), [1]))
        ip1 = np.dot(inverse_intrinsic_matrix, ip1)
        nip1.append(ip1)

        ip2 = kp2[match.trainIdx]
        if isinstance(ip2, cv2.KeyPoint):
            ip2 = np.array(ip2.pt)  
        ip2 = np.vstack((ip2.reshape(2, 1), [1]))
        ip2 = np.dot(inverse_intrinsic_matrix, ip2)
        nip2.append(ip2)
    
    return nip1, nip2

def reconstruct_points_3d(nip1, nip2, cp1, cp2, points_3d_mean, points_3d_std, best_to_image_name):
    reprojection_threshold = 5e-4
    if best_to_image_name in ['sfm12.jpg', 'sfm13.jpg', 'sfm14.jpg']:
        # For bundle adjustment
        reprojection_threshold = 5e-3

    points_3d_new_good = []
    points_3d_new_indices = []
    for i in range(len(nip1)):
        # 1.3.4. Triangulation
        x1, y1, _ = nip1[i]
        x2, y2, _ = nip2[i]
        A = np.array([
            x1 * cp1[2, :] - cp1[0, :],
            y1 * cp1[2, :] - cp1[1, :],
            x2 * cp2[2, :] - cp2[0, :],
            y2 * cp2[2, :] - cp2[1, :]
        ])
        _, _, vt = np.linalg.svd(A)
        point_3d_new = vt[-1]
        point_3d_new = point_3d_new[:3] / point_3d_new[3]
        
        # 1.3.5. Reprojection Error
        projected_point = cp2 @ np.hstack((point_3d_new, [1])).reshape(4, 1)
        projected_point /= projected_point[2]
        
        error = np.linalg.norm(projected_point[:2] - np.array([x2, y2]))
        if error < reprojection_threshold:
            threshold = 4 * points_3d_std
            
            if all(np.abs(point_3d_new - points_3d_mean) <= threshold):
                # Kill the Outlier
                points_3d_new_good.append(point_3d_new)
                points_3d_new_indices.append(i)
            
            else:
                #print(error)
                None

    return points_3d_new_good, points_3d_new_indices


def growing_step():
    image_to_match, image_done_match = load_images()
    points_3d, points_3d_des, uv_info = load_3d_points(image_done_match)

    image_to_match_done = []
    for _ in range(len(image_filename_to_match)):
        # 1.1. Find the best image that have the largest knn match with 3d_des
        # best_match, best_to_image_name = choose_to_image_all(image_to_match)
        best_match, best_to_image_name, image_d = choose_to_image_sequence(points_3d_des, image_to_match)
        to_image_info = image_to_match[best_to_image_name]
        print('To match image: ', best_to_image_name)

        # 1.2. Estimate the camera pose of the best to_image using 3-point PnP RANSAC 
        camera_pose = RANSAC(best_match, to_image_info, points_3d)
        print('Camera Pose')
        print(camera_pose)
        
        # 1.3. Reconstruct 3D points from the best image keypoints and camera pose using triangulation
        # 1.3.1. Find the best match
        # best_match, best_done_image_name = choose_done_image_all(to_image_info, image_done_match)
        best_match, best_done_image_name = choose_done_image_sequence(to_image_info, image_done_match, image_d)
        print('Done match image: ', best_done_image_name)
        print('Candidate 3D Matches: ', len(best_match))

        # 1.3.2. Convert image points to the normalized plane
        image_done_match[best_to_image_name] = {
            'kp'  : to_image_info['kp'],
            'des' : to_image_info['des'],
            'cp'  : camera_pose
        }
        nip1, nip2 = get_normalized_points(image_done_match, best_to_image_name, best_done_image_name, best_match)

        # 1.3.3. Project Matrix
        # pm = K-1 * (K * cp) = cp
        cp1 = image_done_match[best_to_image_name]['cp']
        cp2 = image_done_match[best_done_image_name]['cp']

        # 1.3. Reconstruct 3D points
        points_3d_mean = np.mean(points_3d, axis=0)
        points_3d_std = np.std(points_3d, axis=0)
        points_3d_new_good, points_3d_new_indices = reconstruct_points_3d(nip1, nip2, cp1, cp2, points_3d_mean, points_3d_std, best_to_image_name)
        num_of_points_3d = len(points_3d)
        num_of_points_3d_new = len(points_3d_new_good)
        print('New 3D points: ', num_of_points_3d_new)

        # 1.4. Repeat
        des1 = image_done_match[best_to_image_name]['des']
        ip_des1 = np.array([des1[match.queryIdx] for match in best_match])

        kp1 = image_done_match[best_to_image_name]['kp']
        ip_kp1 = np.array([kp1[match.queryIdx] for match in best_match])

        uv_info[best_to_image_name] = {
            'kp': [],
            '3p_index': []
        }

        # Add new points
        if len(points_3d_new_good) > 0:
            # for bundle adjustments
            uv_info[best_to_image_name]['kp'] = np.array([ip_kp1[index] for index in points_3d_new_indices])
            uv_info[best_to_image_name]['3p_index'] = list(range(num_of_points_3d, num_of_points_3d + num_of_points_3d_new))
            print(num_of_points_3d, num_of_points_3d + num_of_points_3d_new)

            # Add 3D points
            points_3d_new_good = np.array(points_3d_new_good)
            points_3d = np.concatenate([points_3d, points_3d_new_good], axis=0)

            # Add 3D points des
            points_3d_des_new_good = np.array([ip_des1[index] for index in points_3d_new_indices])
            points_3d_des = np.concatenate([points_3d_des, points_3d_des_new_good], axis=0)

        image_to_match_done.append(best_to_image_name)
    
    print()
    print('Generated 3d points: ', len(points_3d))
    print(points_3d)

    # Save the result
    with open('points_3d.obj', 'w') as obj_file:
        for point in points_3d:
            obj_file.write(f'v {point[0]} {point[1]} {point[2]}\n')

    return points_3d, image_done_match, uv_info


def save_obj(points_3d, image_done_match, uv_info):
    with open('points_3d.pkl', 'wb') as file:
        pickle.dump(points_3d, file)
    with open('image_done_match.pkl', 'wb') as file:
        pickle.dump(image_done_match, file)
    with open('uv_info.pkl', 'wb') as file:
        pickle.dump(uv_info, file)

def load_obj():
    with open('points_3d.pkl', 'rb') as file:
        points_3d = pickle.load(file)
    with open('image_done_match.pkl', 'rb') as file:
        image_done_match = pickle.load(file)
    with open('uv_info.pkl', 'rb') as file:
        uv_info = pickle.load(file)    
    return points_3d, image_done_match, uv_info

if __name__ == '__main__':
    # 0. Initialize
    # save_images() # Do once
    
    # 1. Growing Step
    points_3d, image_done_match, uv_info = growing_step()
    save_obj(points_3d, image_done_match, uv_info)

    # 2. Bundle Adjustment
    #points_3d, image_done_match, uv_info = load_obj()

    # 2.1. Generate x
    x = []
    # 2.1.1. Add rotation and translation vectors
    for image_name in image_filename_all:
        camera_pose = image_done_match[image_name]['cp']
        rotation_matrix = camera_pose[:, :3]
        translation_vector = camera_pose[:, 3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        rotation_vector = rotation_vector.reshape(3)
        x.extend(rotation_vector)
        x.extend(translation_vector)

    # 2.1.2. Add all 3D points
    for point_3d in points_3d:
        x.extend(point_3d)
    x_matlab = matlab.double(x)

    # 2.2. Generate params
    # 2.2.1. Add uv
    uv = []
    for image_name in image_filename_all:
        kps = uv_info[image_name]['kp']
        indices_3d_points = uv_info[image_name]['3p_index']

        uv_e = []
        for kp, index_point_3d in zip(kps, indices_3d_points):
            uv_e.append([kp[0], kp[1], 1, index_point_3d + 1])
        
        uv_e = np.array(uv_e).T
        uv_e = matlab.double(uv_e.tolist())
        uv.append(uv_e)

    param = {
        'K': matlab.double(intrinsic_matrix.tolist()),
        'uv': uv,
        'nX': len(points_3d),
        'key1': 4,
        'key2': 5,
        'optimization': 1,
        'dof_remove': 0
    }

    # 2.3. Call LM2_iter_dof MATLAB function
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.addpath(r'./functions', nargout=0)

    x_BA = matlab_engine.LM2_iter_dof(x_matlab, param)

    matlab_engine.quit()
    
    ba_points_3d = np.array(x_BA[-len(points_3d)*3:]).reshape(-1, 3)
    print()
    print('Generated 3d points by BA: ', len(ba_points_3d))
    print(ba_points_3d)
    with open('ba_points_3d.obj', 'w') as obj_file:
        for point in ba_points_3d:
            obj_file.write(f'v {point[0]} {point[1]} {point[2]}\n')