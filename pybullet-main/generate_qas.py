import json
import numpy as np
import os
from tqdm import tqdm
import random


arm_list = ["CR5", "FR5", "UR5", "PANDA"]
save_path = "phys_bench_sim_qas.json"

final_labels = []
inrange_num = 0
uninrange_num = 0
for ARM_CHOOSE in arm_list:
    in_range_path = f"./val/{ARM_CHOOSE}_inrange.json"
    data_path = f"./val/{ARM_CHOOSE}"
    
    root_path = "val"

    with open(in_range_path, 'r') as f:
        inrange_labels = json.load(f)
        
    for inrange_label in tqdm(inrange_labels):
        # image, true, false
        image_name = inrange_label['image']
        inrange_list = inrange_label['true']
        un_inrange_list = inrange_label['false']
        inrange_num += len(inrange_list)
        uninrange_num += len(un_inrange_list)
        
        data = json.load(open(os.path.join(data_path, image_name.replace("rgb", "data").replace(".jpg", ".json")), 'r'))
        boxes = data['bounding_boxes'] # {"cls": [xyxy], ...}
            
        add_qas = []
        ##### question2 #####
        for cls in inrange_list:
            question2 = f"Is the {cls} in the robot's reachable space?"
            answer2 = "Yes"
            add_qas.append({"question": question2, "answer": answer2})
        for cls in un_inrange_list:
            question2 = f"Is the {cls} in the robot's reachable space?"
            answer2 = "No"
            add_qas.append({"question": question2, "answer": answer2})
        
        for qa in add_qas:
            label = {
                "id": image_name.split(".")[0],
                "image": os.path.join(root_path, ARM_CHOOSE, image_name),
                "depth": os.path.join(root_path, ARM_CHOOSE, image_name.replace("rgb", "sp_map")),
                "question": qa["question"],
                "answer": qa["answer"]
            }
            final_labels.append(label)
            
            
        ##### question4 #####
        add_qas = []
        for cls in inrange_list:
            question4 = f"Can the robot directly pick the {cls}?"
            answer4 = "Yes"
            add_qas.append({"question": question4, "answer": answer4})
        for cls in un_inrange_list:
            question4 = f"Can the robot directly pick the {cls}?"
            answer4 = "No"
            add_qas.append({"question": question4, "answer": answer4})
        
        for qa in add_qas:
            label = {
                "id": image_name.split(".")[0],
                "image": os.path.join(root_path, ARM_CHOOSE, image_name),
                "depth": os.path.join(root_path, ARM_CHOOSE, image_name.replace("rgb", "sp_map")),
                "question": qa["question"],
                "answer": qa["answer"]
            }
            final_labels.append(label)
        
print(inrange_num, uninrange_num)

with open(save_path, 'w+') as f:
    json.dump(final_labels, f, indent=4)
