# Fall-Detection-System

**Official implementation of 'Comparative Study on Performance of ML Models for Fall Detection in Older People'**

## Introduction
Fall detection among the elderly is a critical issue in healthcare due to the high risk of injury and associated complications. Falls are a leading cause of hospitalizations and long-term disabilities in older adults, making early and accurate detection essential for timely intervention. With advancements in wearable technologies and machine learning, automated fall detection systems can help monitor and safeguard the elderly, offering both autonomy and safety.

The SisFall dataset has been developed specifically to address this problem. It provides a rich source of sensor data, collected from accelerometers and gyroscopes, to study and build effective fall detection systems. Various machine learning algorithms can be trained on this dataset to distinguish between normal activities and falls, allowing for automated, real-time monitoring and response.

## SisFall Dataset Overview
The SisFall dataset is designed to support research in fall detection, specifically targeting elderly individuals who are more prone to falls. It is composed of both simulated fall events and normal activities, recorded using wearable sensors such as accelerometers and gyroscopes. These sensors are placed on subjects' bodies to capture movement data, allowing researchers to analyze the distinct patterns associated with different activities.

### Key Characteristics of the SisFall Dataset:

1. **Sensor Data:** The dataset includes raw time-series data from accelerometers and gyroscopes, which record the motion and orientation of the body. These sensors are typically worn at the waist, as this position provides a good balance between detecting upper- and lower-body movements.

2. **Falls and Activities:** SisFall contains data from a variety of fall scenarios, such as:
* Forward falls
* Backward falls
* Sideways falls These events are simulated to represent the most common types of real-world falls in elderly individuals. In addition, normal activities, such as walking, sitting, standing, and bending, are included to help machine learning models distinguish between fall and non-fall actions.

3. **Subjects:** The dataset includes data from both young and elderly participants, ensuring a diverse range of movement characteristics. This helps to generalize the models across different age groups and physical conditions, making them more applicable in real-world settings.

4. **Sampling Rate:** The sensor data is collected at a high sampling rate, ensuring that detailed movement patterns are captured. This enables the development of more accurate fall detection algorithms.

## How to run?
1. install libraries in requirements.txt
```
!pip install -r requirements.txt
```
2. Download the dataset
3. Go to Main.ipynb
4. Adjust the variables (relative paths)
5. Run the cells

## How to find the dataset?
1. The SisFall Dataset
```
!gdown -q 1-E-TLd5_J-DDWZXkuYL-moMpoezlMn4Z
```
2. The SisFall Enhanced Dataset (labels)
```
!gdown -q 1gvOuxPc8dNgTnxuvPcVuCKifOf98-TV0
```
## Implementation details

### data_processor.py
* Objective: Process the dataset and convert it to time-series
* Input: The SisFall Dataset
* Output: X_train, y_train, X_test, y_test

### data_processor.py
* Objective: Process the dataset and convert it to time-series
* Input: The SisFall Dataset
* Output: X_train, y_train, X_test, y_test

### deep_models.py
* Objective: Implementation and evaluation of Deep ML Models
* Output: Separate dictionaries, including results and prediction

### traditional_models.py
* Objective: Implementation and evaluation of Traditional ML Models
* Output: Separate dictionaries, including results and prediction

### utils.py
* Objective: Collection of proper functions which are used in this project

### Main.ipynb

* Main Interface of our project
* Output: A data frame including the detailed results

#### Feel free to contact me for any possible issues.
