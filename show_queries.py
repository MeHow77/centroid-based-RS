import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from random import sample

results = np.load('./output-dir/results.npy', allow_pickle=True)
results = results.item()
f = open(f'datasets/df/full_json/query_reid_cropped_{320}_{320}.json')
data = json.load(f)
idx = sample(range(3000), 10)
query_path = '.\\datasets\\df\\320_320_cropped_images\\query\\'
gallery_path = '.\\datasets\\df\\320_320_cropped_images\\gallery\\'
plt.figure()

for query in idx:
    images = data['images']
    file_name = images[query]['file_name']
    print(query_path+file_name)
    query_img = cv2.imread(query_path+file_name)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(query_img)
    plt.title(query_path+file_name)
    
    plt.show()
    
    paths = results[query_path+file_name]['paths']
    for idx, gallery_img_path in enumerate(paths):
        print(gallery_img_path)
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        plt.imshow(gallery_img)
        plt.title(gallery_img_path)
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        plt.show()
