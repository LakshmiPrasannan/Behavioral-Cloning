import os
import csv
import cv2

path = '/opt/carnd_p3/data/'

imgs = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip first row
    next(reader)
    for img in reader:
        imgs.append(img)
i=0    
for img in imgs:
    i = i + 1
    if(i<5):
        # center image
        name = '/opt/carnd_p3/data/IMG/'+img[0].split('/')[-1]
        center_image = cv2.imread(name)
        # convert to RGB
        center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('examples/centered_image.jpg', center_image_rgb)
        cv2.imwrite('examples/centered_image_flip.jpg', cv2.flip(center_image_rgb, 1))
    
        left_name = '/opt/carnd_p3/data/IMG/'+img[1].split('/')[-1]
        left_image = cv2.imread(left_name)
        # convert to RGB
        left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('examples/leftside_image.jpg', left_image_rgb)
        cv2.imwrite('examples/leftside_image_flip.jpg', cv2.flip(left_image_rgb, 1))
    
        right_name = '/opt/carnd_p3/data/IMG/'+img[2].split('/')[-1]
        right_image = cv2.imread(right_name)
        # convert to RGB
        right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('examples/rightside_image.jpg', right_image_rgb)
        cv2.imwrite('examples/rightside_image_flip.jpg', cv2.flip(right_image_rgb, 1))
exit()