import os
from os import listdir
from os.path import isfile, join


mypath = r'C:\Users\magil\Desktop\NMSC\CS\NEA\GUI\carsIMagesdb'
mypath = os.path.join(mypath, 'front')

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [int(name.replace('Car.jpg', '')) for name in onlyfiles]
ss = sorted(onlyfiles)
newnames = [str(str(f)+'Car.jpg') for f in ss]
img_list = []
for i in newnames:
    im_path = os.path.join(mypath, i)
    img_list.append(im_path)