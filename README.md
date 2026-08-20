# Fall Detection System

**Official Implementation of "[Comparative Study on Performance of ML Models for Fall Detection in Older People](https://www.preprints.org/manuscript/202312.2027/download/final_file)"**

## Introduction
Fall detection among the elderly is a crucial issue in healthcare due to the high risk of injury and related complications. Falls are a leading cause of hospitalizations and long-term disabilities in older adults, making early and accurate detection vital for timely intervention. With advancements in wearable technology and machine learning, automated fall detection systems can help monitor and safeguard the elderly, offering both autonomy and security.

The SisFall dataset was specifically developed to address this problem. It provides a rich source of sensor data collected from accelerometers and gyroscopes, enabling the development of effective fall detection systems. Various machine learning algorithms can be trained on this dataset to distinguish among normal activity, the pre-fall stage, and the fall stage, allowing for more informative automated monitoring and response.

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
   pip install -r requirements.txt
   ```
2. Download the dataset.
3. Open `Main.ipynb`.
4. Configure the dataset, enhanced-label, cache, and result paths in the configuration cell.
5. Run the notebook cells.

The notebook works locally or in a cloud notebook. Google Drive mounting is optional.
Paths can be configured in the notebook or through these environment variables:

```
SISFALL_DATASET_PATH
SISFALL_ENHANCED_PATH
SISFALL_CACHE_PATH
SISFALL_RESULT_PATH
SISFALL_EXPERIMENT
SISFALL_SPLIT_PROTOCOL
SISFALL_PREPROCESSING_MODE
SISFALL_RANDOM_SEED
SISFALL_N_JOBS
SISFALL_BOOTSTRAP_ITERATIONS
SISFALL_LOCAL_STAGE_PATH
```

Versioned NumPy arrays are loaded before preprocessing unless `REPROCESS_DATA` is enabled.
When preprocessing runs, the exact train/test recording assignment, fitted scaler,
and traceable train/test window metadata are saved with the cached arrays. Set
`SISFALL_N_JOBS` controls CPU parallelism for supported traditional models and
hyperparameter searches; its default is `-1` (all available CPU cores). Caches
and results are isolated by split protocol, preprocessing mode, window size, and
random seed.
On Colab or another environment with slow mounted storage, setting
`SISFALL_LOCAL_STAGE_PATH` copies both dataset directories to that faster local
directory once. It is disabled by default and is unnecessary on a local laptop.

For the main comparison, use the notebook's `EXPERIMENT_NAME` setting:

1. Run all cells with `primary_compatibility`.
2. Restart the kernel.
3. Run all cells with `primary_raw`.

The final notebook cell combines completed runs in
`experiment_comparison.csv`. The additional `subject_compatibility` and
`subject_raw` presets run the separate subject-independent protocol. The same
selection can be provided through `SISFALL_EXPERIMENT` without editing the
notebook.

### Experimental protocols

The default `stratified_trial` protocol evaluates unseen repetitions from known
subjects. Every complete subject/activity group with five repetitions contributes
four trials to training and one to testing. Single-trial recordings are balanced
globally by activity and class. A fixed seed makes the assignment reproducible,
and `split_manifest.csv` records the split and assignment reason for every trial.
Model selection and deep-learning early stopping use a second fixed, trial-aware
fit/validation assignment created only from the training recordings. Its exact
membership is stored in `validation_manifest.csv`; test recordings are never used
for validation.

The optional `subject_independent` protocol assigns each subject entirely to one
side of the split. It is stored and reported separately because it answers a
different question. Select it with:

```
SISFALL_SPLIT_PROTOCOL=subject_independent
```

## Three-class preprocessing

The enhanced labels are preserved as three model targets:

* `0`: no fall
* `2`: pre-fall
* `1`: fall

Each recording is windowed independently into non-overlapping one-second windows
(200 samples at 200 Hz), so no window can cross a trial boundary. Incomplete tails
are dropped. A window containing event samples is assigned its dominant non-zero
stage; a tie is assigned to fall. Window composition, event position, and source
location are available in `train_window_metadata.csv` and
`test_window_metadata.csv`.

Two preprocessing modes are available without changing the model families:

* `previous_window_difference` (default) preserves the original representation by
  calculating the current window minus its preceding window at corresponding
  sample positions. The transform is reset at trial boundaries.
* `raw_windows` uses normalized sensor windows without the difference transform.

Run the corrected raw-window experiment separately with:

```
SISFALL_PREPROCESSING_MODE=raw_windows
```

### Focused reporting

The final analysis saves one compact model table, one overlaid fall
precision–recall plot, one metric comparison, and combined confusion matrices with
counts and row percentages. CSV reports cover per-activity false positives,
per-fall-type false negatives, per-subject performance, window composition,
false alarms per ADL hour, runtime, model size, and trial-clustered bootstrap
confidence intervals.

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
* **Objective:** Process the dataset and convert it into a time-series format.
* **Input:** SisFall dataset.
* **Output:** Three-class `X_train`, `y_train`, `X_test`, `y_test`, plus per-window metadata.

#### `deep_models.py`
* **Objective:** Implementation and evaluation of deep learning models.
* **Output:** Separate dictionaries containing results and predictions.

#### `traditional_models.py`
* **Objective:** Implementation and evaluation of traditional machine learning models.
* **Output:** Separate dictionaries containing results and predictions.

#### `utils.py`
* **Objective:** Collection of utility functions used throughout the project.

#### `dataset_analysis.py`
* **Objective:** Build the dataset index, validate sensor-label pairs, summarize dataset coverage, and visualize one fall trial across all nine sensor channels.

#### `splitting_protocols.py`
* **Objective:** Create, validate, and summarize reproducible trial-based or subject-independent recording assignments.

#### `evaluation_analysis.py`
* **Objective:** Produce focused metrics, error analysis, confidence intervals, and non-repetitive comparison plots.

#### `experiment_utils.py`
* **Objective:** Lightweight stage timing, platform information, logging, and run-manifest export.

#### `Main.ipynb`
* **Main Interface:** Provides the main interface for the project.
* **Output:** A data frame with detailed results.

### Citation
For more details, you can read our publicly available paper:

```
@article{Esfahani2025FallDetection,
  title={Comparative Study on Performance of ML Models for Fall Detection in Older People},
  author={Mohammadali Sefidi Esfahani and Mohammad Fattahian},
  journal={Journal of Artificial Intelligence, Virtual Reality, and Human-Centered Computing},
  year={2025},
  pages={1-5}
}
```
---

Feel free to contact me for any questions or issues.

--- 
