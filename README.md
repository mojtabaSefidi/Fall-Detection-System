# Fall Detection System

Official implementation of [Comparative Study on Performance of ML Models for
Fall Detection in Older People](https://www.preprints.org/manuscript/202312.2027/download/final_file).

## Baseline objective

This repository provides a reproducible binary baseline for the SisFall dataset.
Normal windows are label `0`; a window containing at least one pre-fall or fall
sample is label `1`. The experiment compares fixed traditional baselines with one
compact deep-learning baseline without feature engineering or hyperparameter
search.

## Dataset

SisFall contains two accelerometers and one gyroscope sampled at 200 Hz. The nine
input channels are retained: three axes from each sensor. The original directory
provides sensor values and SisFall Enhanced provides sample labels.

Only original recordings with a matching enhanced-label recording are used.
Unmatched original files are reported and skipped; source files are never renamed
or modified.

## Run the notebook

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the datasets under `data/SisFall_Dataset` and
   `data/SisFall_Enhanced`, or configure their paths in `Main.ipynb`.
3. Open `Main.ipynb` and run all cells from top to bottom.

Local and Colab paths can also be set through:

```text
SISFALL_DATASET_PATH
SISFALL_ENHANCED_PATH
SISFALL_CACHE_PATH
SISFALL_RESULT_PATH
SISFALL_N_JOBS
```

By default, the notebook keeps all generated data and results in memory and writes
nothing to disk. Set `SAVE_ARTIFACTS = True` to enable the versioned preprocessing
cache and save reports, plots, predictions, and manifests. Model persistence is a
separate `SAVE_MODELS` option and is also disabled by default. When caching is
enabled, set `REPROCESS_DATA = True` after changing the data or preprocessing.

## Experimental protocol

### Chronological trial split

The baseline evaluates later repetitions from known subjects. In every complete
five-repetition subject/activity group, `R01–R04` are assigned to development and
`R05` to testing. The latest development repetition is reserved for CNN
validation, leaving the earlier trials for model fitting.

Activities recorded once per subject cannot be divided chronologically. They are
reproducibly stratified by activity, activity class, and age group. Windows from
one recording never cross split boundaries. The notebook reports subject,
activity, and binary-label coverage for every split.

### Binary raw windows

Each recording is independently divided into non-overlapping one-second windows
of 200 samples. Incomplete tails are dropped. Window targets are:

- `0`: every sample is normal.
- `1`: at least one sample is pre-fall (`2`) or fall (`1`).

No channel selection, handcrafted feature extraction, difference transformation,
or window-level resampling is applied.

### Scaling

A `StandardScaler` fits one independent mean and standard deviation for each of
the nine channels using fit-trial samples only. The same fitted statistics are
applied to validation and test samples. Test data never influence scaling.

### Class imbalance

Learned models use balanced class weights. Training windows are not duplicated,
and validation and test distributions remain unchanged.

## Models

The fixed traditional set contains:

- Majority-class dummy reference
- Linear logistic classifier trained with stochastic gradient descent
- Random forest

KNN is excluded because flattened time-series windows are high-dimensional,
prediction is comparatively expensive, and KNN has no native class weighting.

The deep baseline is one compact 1D CNN with two convolution blocks, global
average pooling, and a small dense classifier. It uses Adam, binary
cross-entropy, balanced class weights, and early stopping on validation PR-AUC.

## Evaluation

The notebook reports only standard binary outcomes:

- Accuracy and balanced accuracy
- Positive-class precision, recall, and F1
- ROC-AUC and PR-AUC
- Classification reports
- Confusion matrices with counts and row percentages
- Overlaid ROC and precision–recall curves
- CNN training and validation curves
- Training and inference time

Predictions, probability scores, reports, plots, split manifests, fitted models,
and the run configuration are saved under the configured result directory.

## Project structure

- `Main.ipynb`: presentation-oriented experiment runner.
- `dataset_analysis.py`: dataset indexing, pairing, summaries, and trial plots.
- `splitting_protocols.py`: reproducible trial-level assignments.
- `data_processor.py`: sensor parsing, binary windowing, and metadata.
- `baseline_pipeline.py`: caching and train-only channel scaling.
- `baseline_models.py`: traditional baselines and compact 1D CNN.
- `baseline_evaluation.py`: binary metrics, reports, and plots.
- `experiment_utils.py`: lightweight logging, timing, and run manifests.

## Citation

```bibtex
@article{Esfahani2025FallDetection,
  title={Comparative Study on Performance of ML Models for Fall Detection in Older People},
  author={Mohammadali Sefidi Esfahani and Mohammad Fattahian},
  journal={Journal of Artificial Intelligence, Virtual Reality, and Human-Centered Computing},
  year={2025},
  pages={1-5}
}
```
