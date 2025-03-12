import requests
import os
import cv2
import json
from tqdm import tqdm 

def send_inference_request(server_url, request_data):
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(server_url, headers=headers, json=request_data)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if response.content:
            print(f"Response content: {response.content}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None
        
    
if __name__ == "__main__":
    server_url = "http://localhost:8001/v1/inference"
    
    with open("/root/private_data/code_space/physvlm_back/pybullet_main/phys_bench_sim_qas.json", 'r') as f:
        datas = json.load(f)
    
    root_path = "/root/private_data/code_space/physvlm_back/pybullet_main"
    total_num = len(datas)
    UR5_num, CR5_num, FR5_num, PANDA_num = 0, 0, 0, 0
    UR5_correct_num, CR5_correct_num, FR5_correct_num, PANDA_correct_num = 0, 0, 0, 0
    for data in tqdm(datas):
        image_path = os.path.join(root_path, data["image"])
        # if "PANDA" not in image_path:
        #     continue
        depth_path = os.path.join(root_path, data["depth"])
        question = data["question"]
        label = data["answer"]
        request_data = {
                        "image_paths": [image_path],
                        "depth_paths": [depth_path],
                        "query": question
                    }
        
        response = send_inference_request(server_url, request_data)
        answer = response["response"]
        if "UR5" in image_path:
            UR5_num += 1
        elif "CR5" in image_path:
            CR5_num += 1
        elif "FR5" in image_path:
            FR5_num += 1
        elif "PANDA" in image_path:
            PANDA_num += 1
        else:
            print("error")
        if answer.lower()[:3] == label.lower()[:3]:
            if "UR5" in image_path:
                UR5_correct_num += 1
            elif "CR5" in image_path:
                CR5_correct_num += 1
            elif "FR5" in image_path:
                FR5_correct_num += 1
            elif "PANDA" in image_path:
                PANDA_correct_num += 1
                
        if UR5_num == 0:
            UR5_num = 1
        if CR5_num == 0:
            CR5_num = 1
        if FR5_num == 0:
            FR5_num = 1
        if PANDA_num == 0:
            PANDA_num = 1
        print(f"ur5_rate: {round(UR5_correct_num/UR5_num, 3)}, cr5_rate: {round(CR5_correct_num/CR5_num, 3)}, fr5_rate: {round(FR5_correct_num/FR5_num, 3)}, panda_rate: {round(PANDA_correct_num/PANDA_num, 3)} | {label} | {answer[:3]}")
        
        
