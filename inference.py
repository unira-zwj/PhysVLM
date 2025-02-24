import requests
import os
import cv2

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
    image_path = input("image_path: ")
    depth_path = input("depth_path: ")
    question = ""
    if image_path[0] != "/":
        image_paths = [os.path.join("/mnt/usb2/datasets", image_path)]
        depth_paths = [os.path.join("/mnt/usb2/datasets", depth_path)]
    else:
        image_paths = [image_path]
        depth_paths = [depth_path]
    while question != "stop":
        question = input("question: ")
        request_data = {
                        "image_paths": image_paths,
                        "depth_paths": depth_paths,
                        "query": question
                    }
        response = send_inference_request(server_url, request_data)
        answer = response["response"]
        print("\nanswer: ", answer)
        
        image = cv2.imread(image_paths[0])
        if depth_path != "":
            sp_map = cv2.imread(depth_paths[0])
            
        if answer[0] == '[' and answer[-1] == ']':
            bbox = answer[1:-1].split(',')
            bbox = [float(b.strip()) for b in bbox]
            # 将bbox从归一化的数值转化到像素坐标
            bbox[0] = int(bbox[0] * image.shape[1])
            bbox[1] = int(bbox[1] * image.shape[0])
            bbox[2] = int(bbox[2] * image.shape[1])
            bbox[3] = int(bbox[3] * image.shape[0])
            # 在image上画出框
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        if "x:" in question and "y:" in question:
            pointx = int(float(question.split("(x:")[1].split(", y:")[0]) * image.shape[1])
            pointy = int(float(question.split(", y:")[1].split(")")[0]) * image.shape[0])
            cv2.circle(sp_map, (pointx, pointy), 1, (0, 0, 255), 4)
        
        cv2.imwrite("result_rgb.jpg", image)
        if depth_path != "":
            cv2.imwrite("result_sp_map.jpg", sp_map)
        
