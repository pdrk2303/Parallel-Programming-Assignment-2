

import cv2
import numpy as np
import os

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})
        
def process_image(file_path, output_file):
    img = cv2.imread(file_path, 0)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    img = img.reshape(28, 28, -1)
    img = img / 255.0
    
    flat_img = img.flatten()
    
    with open(output_file, 'w') as file:
        for value in flat_img:
            file.write('%.6f\n' % value)
        
        
folder_path = 'img/'

# Iterate through all files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith('.png'):
        file = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        output_file = f'pre-proc-img/{file}.txt'
        process_image(file_path, output_file)
        #print(f'Processed: {filename} --> {output_file}')
        
        
        
        