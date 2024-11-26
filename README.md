# Fall Detection System

**Official Implementation of "[Comparative Study on Performance of ML Models for Fall Detection in Older People](https://www.preprints.org/manuscript/202312.2027/download/final_file)"**

## Introduction
Fall detection among the elderly is a crucial issue in healthcare due to the high risk of injury and related complications. Falls are a leading cause of hospitalizations and long-term disabilities in older adults, making early and accurate detection vital for timely intervention. With advancements in wearable technology and machine learning, automated fall detection systems can help monitor and safeguard the elderly, offering both autonomy and security.

The SisFall dataset was specifically developed to address this problem. It provides a rich source of sensor data collected from accelerometers and gyroscopes, enabling the development of effective fall detection systems. Various machine learning algorithms can be trained on this dataset to distinguish between normal activities and falls, allowing for automated, real-time monitoring and response.

## SisFall Dataset Overview
The SisFall dataset is designed to support fall detection research, focusing on elderly individuals prone to falls. It includes simulated fall events and normal activities recorded using wearable sensors, such as accelerometers and gyroscopes. These sensors are worn on the subjects' bodies to capture movement data, enabling researchers to analyze the patterns associated with different activities.

### Key Characteristics of the SisFall Dataset:

1. **Sensor Data:** The dataset includes raw time-series data from accelerometers and gyroscopes that record the body’s motion and orientation. These sensors are typically worn at the waist, providing an optimal balance between detecting upper and lower body movements.

2. **Falls and Activities:** The SisFall dataset contains data from a variety of simulated fall scenarios, such as:
   * Forward falls
   * Backward falls
   * Sideways falls  
   These events are simulated to represent the most common real-world falls in elderly individuals. Additionally, normal activities like walking, sitting, standing, and bending are included to help machine learning models differentiate between fall and non-fall actions.

3. **Subjects:** The dataset includes data from young and elderly participants, ensuring a diverse range of movement characteristics. This helps to generalize the models across different age groups and physical conditions, making them more applicable in real-world settings.

4. **Sampling Rate:** The sensor data is collected at a high sampling rate, capturing detailed movement patterns. This enhances the accuracy of fall detection algorithms.

## How to Run

1. Install the required libraries from the `requirements.txt` file:
   ```
   !pip install -r requirements.txt
   ```
2. Download the dataset.
3. Open `Main.ipynb`.
4. Adjust the variables (e.g., relative paths).
5. Run the notebook cells.

## How to Access the Dataset

1. **SisFall Dataset:**
   ```
   !gdown -q 1-E-TLd5_J-DDWZXkuYL-moMpoezlMn4Z
   ```
2. **SisFall Enhanced Dataset (Labels):**
   ```
   !gdown -q 1gvOuxPc8dNgTnxuvPcVuCKifOf98-TV0
   ```

## Implementation Details

#### `data_processor.py`
* **Objective:** Process the dataset and convert it into time-series format.
* **Input:** SisFall dataset.
* **Output:** `X_train`, `y_train`, `X_test`, `y_test`.

#### `deep_models.py`
* **Objective:** Implementation and evaluation of deep learning models.
* **Output:** Separate dictionaries containing results and predictions.

#### `traditional_models.py`
* **Objective:** Implementation and evaluation of traditional machine learning models.
* **Output:** Separate dictionaries containing results and predictions.

#### `utils.py`
* **Objective:** Collection of utility functions used throughout the project.

#### `Main.ipynb`
* **Main Interface:** Provides the main interface for the project.
* **Output:** A data frame with detailed results.

### Citation
For more details, you can read our publicly available paper:

```
@article{Esfahani2023FallDetection,
  title={Comparative Study on Performance of ML Models for Fall Detection in Older People},
  author={Mohammadali Sefidi Esfahani and Mohammad Fattahian},
  journal={Preprints.org},
  year={2023},
  doi={https://doi.org/10.20944/preprints202312.2027.v1}
}
```
---

Feel free to contact me for any questions or issues.

--- 
