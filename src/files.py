import os
import cv2
import numpy as np

def read(path):    
    data_set = []
    for dir_path, subfolders_path, files_path in os.walk(path):
        for file_path in files_path:
            relative_path = os.path.join(dir_path, file_path)
            classification = str(dir_path[len(dir_path)-1])
            img = cv2.imread(relative_path)
            img = cv2.resize(img, (256,256))
            data_set.append((img, classification))
    return data_set

def read_val(path):    
    data_set = []
    for dir_path, subfolders_path, files_path in os.walk(path):
        for file_path in files_path:
            relative_path = os.path.join(dir_path, file_path)
            classification = str(dir_path[len(dir_path)-1])
            img = cv2.imread(relative_path)
            img = cv2.resize(img, (256,256))
            data_set.append((img, classification))
    return data_set


def read_data_set(path, amount=10, binary=False):    
    data_set = []
    data_set_label = []
    i = 0
    for dir_path, subfolders_path, files_path in os.walk(path):
        for file_path in files_path:
            if(i == amount):
                continue
            i+=1
            relative_path = os.path.join(dir_path, file_path)
            classification = int(str(dir_path[len(dir_path)-1]))
            
            if(binary):
                classification = 0 if classification < 2 else 1
                
            img = cv2.imread(relative_path)
            data_set.append(img)
            data_set_label.append(classification)
        i=0
    return  np.array(data_set), np.array(data_set_label)

def equalize_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img

def flip_image(img):
    img = cv2.flip(img, 1)
    return img

def create_images(data_set, dir):
    dir = f'{dir}/data_set'
    classifications = set(list(map(lambda data: data[1] , data_set)))
    
    if(not os.path.exists(dir)):
        os.mkdir(f'{dir}')

    for classification in classifications:
        if(not os.path.exists(f'{dir}/{classification}')):
            os.mkdir(f'{dir}/{classification}')
    
    iterator = 1
    for (img, classification) in data_set:
        equalized_img = equalize_image(img)
        flipped_img = flip_image(equalized_img)
        cv2.imwrite(os.path.join(f'{dir}/{classification}', f'{iterator}_equalized.png'), equalized_img)
        cv2.imwrite(os.path.join(f'{dir}/{classification}', f'{iterator}_flipped.png'), flipped_img)   
        iterator+=1


def read_data_set_val(path, amount=10, binary=False):    
    data_set = []
    data_set_label = []
    i = 0
    for dir_path, subfolders_path, files_path in os.walk(path):
        for file_path in files_path:
            if(i == amount):
                continue
            i+=1
            relative_path = os.path.join(dir_path, file_path)
            classification = int(str(dir_path[len(dir_path)-1]))
            
            if(binary):
                classification = 0 if classification < 2 else 1
                
            img = cv2.imread(relative_path)
            data_set.append(img)
            data_set_label.append(classification)
            print(relative_path)
        i=0
    return  np.array(data_set), np.array(data_set_label)

def create_images_val(data_set, dir):
    dir = f'{dir}/data_set_val'
    classifications = set(list(map(lambda data: data[1] , data_set)))
    
    if(not os.path.exists(dir)):
        os.mkdir(f'{dir}')

    for classification in classifications:
        if(not os.path.exists(f'{dir}/{classification}')):
            os.mkdir(f'{dir}/{classification}')
    
    iterator = 1
    for (img, classification) in data_set:
        equalized_img = equalize_image(img)
        flipped_img = flip_image(equalized_img)
        cv2.imwrite(os.path.join(f'{dir}/{classification}', f'{iterator}_equalized.png'), equalized_img)
        cv2.imwrite(os.path.join(f'{dir}/{classification}', f'{iterator}_flipped.png'), flipped_img)   
        iterator+=1