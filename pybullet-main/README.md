# To generate phys-bench-sim

We will primarily introduce the process of constructing an S-P Map as an evaluation set. Example with robot `UR5`, you also can use `CR5`, `FR5`, and `PANDA`.

If you wish to utilize the robot's physical information in other ways, you can also obtain the necessary states from `simulation_utils.py`, `camera_utils.py`, and `data_utils.py`.

## 1. Offline stage of S-P Map construction (option)

```bash
python offline_calculate_robot_ws.py --robot UR5
```

We already finished it, the reachable space voxel at: `arm_ws_pcb/`

## 2. Online stage of S-P Map construction

### 2.1 Get data, image, depth from environment

```bash
python main.py --robot UR5 --dataset val
```

### 2.2 Generate S-P Map

```bash
python generate_sp_map.py  --robot UR5
```

### 2.3 Generate qas

```bash
python generate_qas.py 
```

