# MLB Bullpen Optimization: Machine Learning–Driven Pitching Decisions

## Project Overview

This project develops a data-driven decision support system for MLB bullpen management. It combines an XGBoost plate-appearance outcome prediction model with Monte Carlo simulation to evaluate potential relief pitchers in historical game situations.

The objective is to identify pitchers who minimize expected runs allowed or walk-off risk, depending on the game situation.

## Methodology

The system consists of three main components:

1. **Data Preparation:** Historical Statcast data and MLB Stats API records are used to construct pitcher–batter matchup features and reconstruct game situations.
2. **Machine Learning:** An XGBoost multiclass classification model estimates probabilities for six plate-appearance outcomes: single, extra-base hit, walk/hit-by-pitch, home run, strikeout, and other out.
3. **Monte Carlo Simulation:** Candidate relief pitchers are evaluated by repeatedly simulating the remainder of a half-inning using predicted plate-appearance probabilities.

The system ranks pitchers according to the relevant decision objective:

- **Expected Runs Allowed:** Used to evaluate run prevention.
- **Walk-off Risk:** Used when the defensive team faces a potential walk-off situation.

## Repository Structure

```text
bullpen_project/
├── bullpen_train_quick.py
├── bullpen_predict.py
├── bullpen_game_setup.py
├── simulation.py
├── requirements.txt
├── README.md
└── results/
    └── bullpen_quick/
        ├── xgb_matchup_model.json
        ├── model_metadata.json
        ├── feature_history.pkl
        └── pitcher_daily_workload.parquet
```

The `results/bullpen_quick/` directory also contains model evaluation artifacts.

## Installation

Python 3.9 or a compatible environment is recommended for the pinned dependencies.

Clone the repository:

```bash
git clone https://github.com/josechen16/mlb-bullpen-optimization.git
cd mlb-bullpen-optimization
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

An internet connection is required to retrieve historical MLB game information through the MLB Stats API.

## Reproducing the Analysis

The repository includes a pretrained XGBoost model, allowing bullpen simulations to run without retraining the model.

### Example 1: Kansas City Royals vs. Cleveland Guardians

September 11, 2025 — Eighth inning.

```bash
python3 simulation.py --date 2025-09-11 --team KC --opponent CLE --inning 8
```

### Example 2: Arizona Diamondbacks vs. Minnesota Twins

September 12, 2025 — Ninth inning.

```bash
python3 simulation.py --date 2025-09-12 --team AZ --opponent MIN --inning 9
```

In these commands, `--team` specifies the pitching team, `--opponent` specifies the batting team, and `--inning` identifies the inning being evaluated.

The program saves visualization outputs under `results/bullpen_demo/`, organized by game date, matchup, and inning.

## Model Training

The model training pipeline is implemented in `bullpen_train_quick.py`. Historical Statcast training data is stored locally under `data/` and is not included in this repository.

The pretrained model and associated feature-history artifacts are included for simulation reproducibility. Reproducing model training from scratch requires obtaining the corresponding historical data.

## Limitations

- Public MLB data cannot fully establish bullpen availability, pitcher health, or warm-up status.
- Simulated plate appearances use simplified baserunning and game-state transitions.
- Model predictions depend on historical data and feature coverage.
- Results are intended to support baseball decision-making rather than replace managerial judgment.
- Supported Game Dates: Automatic game lookup currently supports MLB games after September 5, 2025. The XGBoost model was trained using data available through September 4, 2025. Games before September 5 are not supported in automatic mode to prevent data leakage.

## Project Status

This repository represents the Checkpoint #1 implementation of an ongoing MLB bullpen decision support research project. Further work will focus on validation, robustness, improved game-state modeling, and decision quality.
