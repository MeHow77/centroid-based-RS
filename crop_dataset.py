from calendar import c
import json

subsets = ['train', 'test', 'val', 'query', 'gallery']
#subsets = ['train']
min_prcnt = 0.1
size_w = 320
size_h = 320
for subset in subsets:
    f = open(f'datasets/df/full_json/{subset}_reid_cropped_{size_w}_{size_h}.json')
    data = json.load(f)
    
    anno_len = len(data['annotations'])
    img_len = len(data['images'])
    
    min_records = int(len(data['images'])*min_prcnt)
    print(f"Min records to obtain: {min_records}")
    curr_record = 0
    if_still_current_item = True
    cropped_dataset = {'images':[], 'annotations': []}
    
    # take another record until it exceeds needed number or  it still holds to the previous record
    while curr_record <= min_records or if_still_current_item:
        # get current record
        sample_item = data['images'][curr_record]
        sample_anno = data['annotations'][curr_record]
        next_sample = data['images'][curr_record+1]
        cropped_dataset['images'].append(sample_item)
        cropped_dataset['annotations'].append(sample_anno)
        # take 'consumer' or 'shop' factor for next record
        next_item_id = next_sample['file_name'].split('_')[1]
        next_id = sample_item['file_name'].split('_')[1]
        # create True flag for consumer
        if_still_current_item = True if next_item_id == next_id else False
        curr_record += 1
        
    # print(cropped_dataset['images'])
    # print(cropped_dataset['annotations'])
    with open(f'datasets/df/cropped_json/{subset}_reid_cropped_{size_w}_{size_h}.json', 'w') as json_f:
       json.dump(cropped_dataset, json_f)