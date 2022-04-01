import cv2
import os

image_folder = 'imaforvideo'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 2, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

videodims = (384,384)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video = cv2.VideoWriter("test_vid.mp4",fourcc, 3, videodims)
img = Image.new('RGB', videodims, color = 'darkred')
#draw stuff that goes on every frame here
for i in range(0,60*60):
    imtemp = img.copy()
    # draw frame specific stuff here.
    video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
video.release()