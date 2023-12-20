# Fall-Detection-System

**Official implementation of 'Comparative Study on Performance of ML Models for Fall Detection in Older People'**

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
1. The SisFall Dataset (features)
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
