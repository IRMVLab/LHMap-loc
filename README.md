## LHMap-loc: Cross-Modal Monocular Localization Using LiDAR Point Cloud Heat Map (ICRA 2024)



The dependencies installation and data prepration please refer to [CMRNet](https://github.com/cattaneod/CMRNet)

## LHMap construction
  ```
python main_single_mapping.py with batch_size=8 data_folder=./KITTI_ODOMETRY/sequences/ epochs=120 max_r=10 max_t=2 BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=00 
  ```

## LHMap save
Change folder path to save LiDAR point cloud in "./models/CMRNet/CMRNet_single_save.py" 
  ```
python main_single_save.py with batch_size=1 data_folder=./KITTI_ODOMETRY/sequences/  weights='./checkpoints/xxx.tar' test_sequence=00
python main_single_save.py with batch_size=1 data_folder=./KITTI_ODOMETRY/sequences/  weights='./checkpoints/xxx.tar' test_sequence=03
python main_single_save.py with batch_size=1 data_folder=./KITTI_ODOMETRY/sequences/  weights='./checkpoints/xxx.tar' test_sequence=05
...
  ```


## LHMap localization
Change folder path of LiDAR point cloud in "Dataset_kitti_save.py" 
  ```
python main_single_loc.py with batch_size=12 data_folder=./KITTI_ODOMETRY/sequences/ epochs=150 max_r=10 max_t=2 BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=00 
  ```


## Evaluation
  ```
python evaluate.py with test_sequence=00 maps_folder=local_maps data_folder=./KITTI_ODOMETRY/sequences/  weight="['./checkpoints/iter1.tar','./checkpoints/iter2.tar','./checkpoints/iter3.tar']"
  ```
