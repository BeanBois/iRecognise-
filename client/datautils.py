import numpy as np
import os
import cv2

IMG_SIZE = (64, 64)
def load_data(folder_path, img_size = IMG_SIZE):
    
    vids = []
    keypoints_set = []
    ratio = np.ones((4,))
    for filename in os.listdir(folder_path):
        # Check if the file is a .npz file
        if filename.endswith('.npz'):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)
            
            # Load the .npz file
            data = np.load(file_path)
            imgs = data['colorImages']            
            kps = data['boundingBox']
            time = imgs.shape[-1]
            vid = []
            keypoints = []
            # l2d = data['landmarks2D']
            # l3d = data['landmarks3D']
            for t in range(time):
                img = imgs[:,:,:,t]
                kp = kps[:,:,t]
                # Resize image to 64x64
                resized_img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
                
                # Normalize image to [0, 1]
                normalized_img = resized_img.astype(np.float32) / 255.0
                
                # Scale keypoints according to the new image size
                original_size_x = img.shape[0]  
                original_size_y = img.shape[1]
                scale_factor_x = IMG_SIZE[0] / original_size_x
                scale_factor_y = IMG_SIZE[1] / original_size_y
                
                scaled_kp_x = kp[0] * scale_factor_x
                scaled_kp_y = kp[1] * scale_factor_y
                
                vid.append(normalized_img)
                keypoints.append(np.array([scaled_kp_x, scaled_kp_y]))
            
            vids.append(np.array(vid))
            keypoints_set.append(np.array(keypoints))
                
    X_train = np.concatenate(vids, axis=0)  # Stack features
    y_train = np.concatenate(keypoints, axis=0)
    return X_train, y_train