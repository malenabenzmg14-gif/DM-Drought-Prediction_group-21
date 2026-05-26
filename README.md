# DM Drought Prediction - Group 21

This repository contains the final project code for the Data Mining Kaggle competition on natural disaster severity prediction.

The goal of the project is to predict drought severity scores for the next five weeks for each region using historical meteorological data.

## Project Structure

```text
DM-Drought-Prediction_group-21/
├── data/
│   └── sample_submission.csv
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── final_predict.py
├── main.py
├── README.md
└── .gitignore
```

## Required Data Files

The large Kaggle data files are not included in this repository.

Before running the code, place the Kaggle competition files in the `data/` folder:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Only `sample_submission.csv` is included in the repository because it is small and defines the required submission format.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn lightgbm
```

The code was developed with Python 3.11.

## How to Run

From the project root directory, run:

```bash
python main.py
```

The script trains the models from scratch and creates the final Kaggle submission file:

```text
outputs/submission_final.csv
```

## Method Summary

The final solution uses a two-stage LightGBM forecasting pipeline.

First, daily meteorological observations are aggregated into weekly region-level features. The feature set includes precipitation, humidity, temperature, wind, surface pressure, and rolling weather statistics over multiple weekly windows.

Second, the missing drought scores for the 13 observed test weeks are estimated. These estimated scores are then used as recent lag features for the final five-week forecast.

The final prediction step uses horizon-specific LightGBM models, meaning one separate model is trained for each forecast week from week 1 to week 5.

The final feature set includes:

- weekly meteorological aggregations
- multi-week weather rollups
- recent drought score lag features
- rolling score statistics
- region-level historical averages
- month-level seasonal averages
- day-of-year sine and cosine features
- horizon-specific target construction

The best public leaderboard result was achieved using a 450-week training window per region.

## Experiments

| Experiment | Description | Public Score |
|---|---|---:|
| Early baseline | Initial simple forecast logic | 1.0384 |
| Step 6 | Predicted test-week score lags | 0.8853 |
| Step 7 | Added 91-day weather rollup features | 0.8341 |
| Step 12 | 390-week training window | 0.8234 |
| Final | 450-week training window | 0.8233 |
| 520-week window | More history, worse generalization | 0.8267 |
| Horizon smoothing | Postprocessing experiment, rejected | 0.8371 / 0.8393 |
| Horizon downscale | Calibration experiment, rejected | 0.8425 |
| Persistence blend | Persistence postprocessing, rejected | 0.8362 |

## Final Submission

The final submission is generated at:

```text
outputs/submission_final.csv
```

The best public leaderboard score achieved by the final version was:

```text
0.8233
```

## Reproducibility

The repository does not require pre-trained model files.

To reproduce the final submission:

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` folder.
2. Install the required dependencies.
3. Run:

```bash
python main.py
```

The output file will be written to:

```text
outputs/submission_final.csv
```

## Notes

Generated submission files, model artifacts, archives, and large Kaggle data files are excluded from Git using `.gitignore`.