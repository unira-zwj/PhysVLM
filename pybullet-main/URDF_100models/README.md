# URDF_models

### Overview
This repository is a comprehensive collection of over 100 URDF (Unified Robot Description Format) models, specifically designed for use with Pybullet. 
It serves as a valuable resource for researchers, developers, and hobbyists in the field of robotics and simulation.

### Model Details

**urdf_dataset.xlsx**: Each model in this repository is accompanied by detailed information to facilitate its use and integration. The following table outlines key attributes for each model:


| index | subfolder_name   | relative_path                | size_x     | size_y     | size_z     | object_name     | euler_x | euler_y | euler_z | object_description                                                                                           |
|-------|------------------|------------------------------|------------|------------|------------|-----------------|---------|---------|---------|-------------------------------------------------------------------------------------------------------------|
| 0     | black_marker     | black_marker/model.urdf      | 0.023168   | 0.151316   | 0.022861   | black_marker    | 0       | 0       | 0       | a black marker pen                                                                                          |
| 1     | bleach_cleanser  | bleach_cleanser/model.urdf   | 0.109653   | 0.07443    | 0.257586   | bleach_cleanser | 0       | 0       | 0       | a white bleach cleanser with the "Soft Scrub" brand label on it. The label is blue and green.                |
| 2     | blue_cup         | blue_cup/model.urdf          | 0.087695   | 0.112706   | 0.074358   | blue_cup        | 0       | 0       | 0       | A greyish-blue mug.                                                                                         |
| 3     | blue_marker      | blue_marker/model.urdf       | 0.023173   | 0.15136    | 0.022867   | blue_marker     | 0       | 0       | 0       | A marker with a blue cap and a black body.                                                                  |
| ...     | ...        | ...         | ...   | ...   | ...   | ...       | ...       | ...       | ...       | ... |

<!-- | 4     | blue_moon        | blue_moon/model.urdf         | 0.063152   | 0.102187   | 0.258420   | blue_moon       | 0       | 0       | 0       | Laundry detergent in a red bottle with a Chinese brand label reading "Blue Moon". The bottle cap is darker. | -->

#### Attributes Explained
- **subfolder_name**: Name of the folder containing the URDF model.

- **relative_path**: Path to the URDF model file.

- **size_x, size_y, size_z**: Dimensions of the model in meters.

- **object_name**:  Identical to the subfolder name, for easy reference.

- **euler_x, euler_y, euler_z**: Suggested orientation for realistic positioning.

- **object_description**: Detailed annotation of each model.


### Tools and Resources
- Model Visualization: To view and interact with these URDF models, use our provided: 
  
```
python main.py
```
[![Demo](https://img.youtube.com/vi/If3RQEv8TvQ/0.jpg)](https://www.youtube.com/watch?v=If3RQEv8TvQ)

### Acknowledgement

- Special thanks to the [pybullet-URDF-models](https://github.com/ChenEating716/pybullet-URDF-models) project for some of the models included in this repository.

- We continuously strive to update and enhance this collection. Contributions and suggestions are always welcome! 💪💪💪