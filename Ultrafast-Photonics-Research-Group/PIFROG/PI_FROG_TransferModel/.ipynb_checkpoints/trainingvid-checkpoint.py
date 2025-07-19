import cv2
import os

folder = 'trainingimg'

# Sorted list of file paths
images = sorted(
    [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')],
    key=lambda x: int(x.split('frog_inloss')[-1].split('.png')[0])
)

# Read first image to get size
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Output video
video = cv2.VideoWriter('frog_training_64.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

for image in images:
    video.write(cv2.imread(image))

video.release()
cv2.destroyAllWindows()
