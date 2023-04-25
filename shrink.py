import os
import cv2
from glob import glob
from tqdm import tqdm

def shrink(data_name):
    
    root = os.getcwd()
    image_paths = glob(os.path.join(root,"data",data_name,"*.JPG"))
    save_paths = os.path.join(root,"data_small",data_name)
    
    os.makedirs(save_paths, exist_ok=True)
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        width = int(img.shape[1] * 0.1)
        height = int(img.shape[0] * 0.1)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_name = image_path.split("/")[-1]
        cv2.imwrite(os.path.join(save_paths, img_name), img)
        

if __name__=="__main__":
    shrink("data1")
    shrink("data2")

