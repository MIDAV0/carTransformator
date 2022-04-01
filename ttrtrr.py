import os
from glob import glob

dataset_path = r'C:\Users\magil\Desktop\NMSC\CS\NEA\carsdataset'
dataset_path = os.path.join(dataset_path, 'train')

domain_list = ['front', 'rear', 'side']
images = []

for ind, domain in enumerate(domain_list):
    img_list = glob(os.path.join(dataset_path, domain)+'/*.jpg')
    images.extend(img_list)