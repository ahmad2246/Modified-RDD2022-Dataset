# Modified-RDD2022-Dataset
This code will apply some modifications on the RDD2022 dataset avaliable at this [link](https://doi.org/10.48550/arXiv.2209.08538), and combine it with a pothole dataset avaliable at this [link](https://doi.org/10.1016/j.dib.2023.109206).





# The original RDD2022 dataset comes with 8 classes as follow:
* D00: Wheel mark part (Longitudinal) [Linear Crack]
* D01: Construction joint part (Longitudinal) [Linear Crack]
* D10: Equal interval (Lateral) [Linear Crack]
* D11: Construction joint part (Lateral) [Linear Crack]
* D20: Partial pavement, overall pavement (Alligator Crack)
* D40: Rutting, bump, pothole, separation (Other Corruption)
* D43: Crosswalk blur (Other Corruption)
* D44: White line blur (Other Corruption)


# The modification that will be done to the RDD2022 dataset includes:
- Changing the following classes: [D00, D01, D10, D11] into a single class "Linear-Crack"
- Keeping the D20 class as "Alligator-Cracl"
- Removing the following classes: [D40, D43, D44]
- Combining the "pothole" dataset with the modified RDD2022 dataset

# After running the code, the output is a new dataset with the following distribution:

- class (0): Linear-Crack: 38070 objects
- class (1): Alligator-Crack: 10617 objects
- class (2): pothole: 1156 objects

### Class distribution in train split:
- Linear-Crack: 26750 objects
- Alligator-Crack: 7419 objects
- pothole: 824 objects

### Class distribution in valid split:
- Linear-Crack: 3788 objects
- Alligator-Crack: 1073 objects
- pothole: 107 objects

### Class distribution in test split:
- Linear-Crack: 7532 objects
- Alligator-Crack: 2125 objects
- pothole: 225 objects

# Run the code
To run the code, you need to follow these steps:
1- specify the path to RDD2022 dataset **"RDD_dataset_path"**
2- specify the path to pothole dataset **"pothole_dataset_path"**
3- specify where you want the combined dataset to be created **"final_dataset_path"**
4- Run the code 

Note: The code might take upto 45 minutes to finish.