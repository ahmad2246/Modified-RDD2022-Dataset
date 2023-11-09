# Modified-RDD2022-Dataset
This code is designed to apply several modifications to the RDD2022 dataset, which is available at this [link](https://doi.org/10.48550/arXiv.2209.08538), and combine it with a pothole dataset, accessible at this [link](https://doi.org/10.1016/j.dib.2023.109206).





# Original RDD2022 Dataset
The original RDD2022 dataset comprises eight classes, including:
* D00: Wheel mark part (Longitudinal) [Linear Crack]
* D01: Construction joint part (Longitudinal) [Linear Crack]
* D10: Equal interval (Lateral) [Linear Crack]
* D11: Construction joint part (Lateral) [Linear Crack]
* D20: Partial pavement, overall pavement (Alligator Crack)
* D40: Rutting, bump, pothole, separation (Other Corruption)
* D43: Crosswalk blur (Other Corruption)
* D44: White line blur (Other Corruption)


# Modifications to RDD2022 Dataset
The modifications applied to the RDD2022 dataset are as follows:
- Changing the following classes: [D00, D01, D10, D11] into a single class "Linear-Crack"
- Keeping the D20 class as "Alligator-Crack"
- Removing the following classes: [D40, D43, D44]
- Combining the "pothole" dataset with the modified RDD2022 dataset


# Additional Functions
Apart from the modifications, this code also performs the following tasks:
- Checks if each label file in each dataset has a corresponding image
- Converts annotations from ".xml" to ".txt"
- Performs undersampling of the majority classes to create a balanced dataset based on class objects
- Splits the dataset into train/valid/test sets with a ratio of 0.7/0.1/0.2 while maintaining a balanced split based on class objects
- Plots the class distribution after generating the desired dataset


# Running the Code
To run the code, follow these steps:
1) Download the RDD2022 dataset from this [link](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547?file=38030910).
2) Download the pothole dataset from this [link](https://data.mendeley.com/datasets/tp95cdvgm8/1).
3) Extract the datasets:
	* Extract the pothole dataset.
	* Extract the RDD2022 dataset (including the country datasets folders inside it).
3) Download the processing.py file.
4) In the processing.py file:
	* Specify the path to the RDD2022 dataset using the variable **"RDD_dataset_path"**.
	* Specify the path to the pothole dataset using the variable **"pothole_dataset_path"**.
5) Run the code 

**Note**: The execution time of the code may be up to 30 minutes.




