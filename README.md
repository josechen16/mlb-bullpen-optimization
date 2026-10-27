# MLB Bullpen Optimization: Machine Learning–Driven Pitching Decisions

## Project Overview

This project develops a machine learning–based decision support system for MLB bullpen management. It combines an XGBoost plate-appearance outcome prediction model with Monte Carlo simulation to evaluate potential relief pitchers in historical game situations.

The primary objective is to identify pitchers who minimize expected runs allowed or walk-off risk based on the current game situation.

Rather than relying exclusively on traditional pitching statistics, the system incorporates historical pitcher–batter matchup information, game context, and probabilistic simulation to compare bullpen alternatives.

This repository contains the Checkpoint #1 implementation of an ongoing research project.

## Methodology

The system consists of three main components.

### 1. Data Preparation

Historical Statcast data and MLB Stats API records are used to construct pitcher–batter matchup features and reconstruct game situations.

The system retrieves historical game information, identifies potential relief pitchers, and determines the relevant batting lineup and game state.

### 2. Machine Learning: XGBoost

An XGBoost multiclass classification model estimates the probabilities of six plate-appearance outcomes:

1. Single
2. Extra-base hit
3. Walk or hit-by-pitch
4. Home run
5. Strikeout
6. Other out

The model uses historical matchup features to estimate the probability distribution of these outcomes for each pitcher–batter matchup.

The training data cutoff is September 4, 2025. Games evaluated after this date are outside the model's training period.

### 3. Monte Carlo Simulation

The system evaluates potential relief pitchers by simulating the remainder of a half-inning.

For each candidate pitcher:

1. Retrieve the current game situation.
2. Generate plate-appearance outcome probabilities using the trained XGBoost model.
3. Simulate subsequent plate appearances and update the game state.
4. Repeat the simulation 10,000 times.
5. Calculate expected runs allowed and relevant scoring probabilities.
6. Rank pitchers according to the selected decision objective.

### Decision Metrics

**Expected Runs Allowed**

Measures the average number of runs surrendered across simulated half-innings. Lower values indicate better projected run prevention.

**Walk-off Probability**

Measures the simulated probability that the batting team scores enough runs to win during the evaluated half-inning.

For eligible walk-off situations, the system prioritizes minimizing walk-off probability rather than expected runs allowed.

All reported probabilities are model-based simulation estimates, not observed real-world outcome frequencies.

## Repository Structure

```text
mlb-bullpen-optimization/
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
        ├── pitcher_daily_workload.parquet
        └── additional model evaluation artifacts
```

### Main Files

| File | Description |
|---|---|
| `simulation.py` | Reconstructs historical game situations, evaluates bullpen candidates, runs Monte Carlo simulations, and generates visualizations. |
| `bullpen_train_quick.py` | Implements the XGBoost model training pipeline. |
| `bullpen_predict.py` | Contains prediction-related functionality. |
| `bullpen_game_setup.py` | Supports historical game setup and data preparation. |
| `requirements.txt` | Lists the Python dependencies required by the project. |
| `results/bullpen_quick/` | Contains the pretrained model, feature history, workload data, and evaluation artifacts. |

Generated simulation outputs are stored separately under `results/bullpen_demo/`.

## Installation

### Requirements

- Python 3.10 or later is recommended, subject to compatibility with the pinned dependencies.
- Git
- Internet connection for retrieving historical MLB game information
- Python packages listed in `requirements.txt`

The project was developed and tested on macOS.

### Step 1: Clone the Repository

```bash
git clone https://github.com/josechen16/mlb-bullpen-optimization.git
cd mlb-bullpen-optimization
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the environment on macOS or Linux:

```bash
source venv/bin/activate
```

On Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Step 4: Verify the Installation

Check that the required libraries can be imported:

```bash
python3 -c "import xgboost, pandas, numpy, sklearn, matplotlib, pybaseball; print('Dependencies loaded successfully')"
```

The pretrained model and associated feature-history files are already included in the repository. Retraining is not required to reproduce the provided bullpen simulations.

## Reproducing the Analysis

### Supported Game Dates

Automatic historical game lookup currently supports games dated September 5, 2025, through December 31, 2025.

The XGBoost model uses training data available through September 4, 2025.

This separation helps prevent the evaluated games from being included in the model's training period.

The supported date range does not imply that an MLB game exists on every date. Users must select a valid game and inning.

### Command Format

```bash
python3 simulation.py --date YYYY-MM-DD --team TEAM --opponent OPPONENT --inning N
```

Arguments:

| Argument | Description |
|---|---|
| `--date` | Historical game date in YYYY-MM-DD format |
| `--team` | Pitching team being evaluated |
| `--opponent` | Batting team |
| `--inning` | Inning to evaluate |

**Important:** The `--team` argument always refers to the defensive team whose bullpen is being evaluated, not necessarily the home team.

Team abbreviations must match those recognized by the MLB Stats API. For example, use `AZ` for the Arizona Diamondbacks.

### Example 1: Kansas City Royals vs. Cleveland Guardians

**Date:** September 11, 2025  
**Situation:** Eighth inning  
**Pitching team:** Kansas City Royals  
**Batting team:** Cleveland Guardians

Run:

```bash
python3 simulation.py --date 2025-09-11 --team KC --opponent CLE --inning 8
```

The system reconstructs the selected historical situation and compares eligible Kansas City relief pitchers.

The simulation evaluates their projected run prevention and ranks the candidates according to the applicable decision metric.

### Example 2: Arizona Diamondbacks vs. Minnesota Twins

**Date:** September 12, 2025  
**Situation:** Bottom of the ninth inning  
**Pitching team:** Arizona Diamondbacks  
**Batting team:** Minnesota Twins

Run:

```bash
python3 simulation.py --date 2025-09-12 --team AZ --opponent MIN --inning 9
```

For this historical situation, Arizona enters the evaluated half-inning with a two-run lead.

The system evaluates potential relief pitchers using 10,000 Monte Carlo simulations per candidate and ranks them by simulated walk-off probability.

### Reference Results: Example 2

The following results were obtained from a successful local execution of the pretrained model.

| Pitcher ID | Throws | Expected Runs | Walk-off Probability |
|---|---|---:|---:|
| 685314 | L | 0.343 | 3.80% |
| 543101 | R | 0.391 | 4.41% |
| 688497 | R | 0.397 | 4.68% |
| 663765 | R | 0.416 | 4.95% |
| 595928 | R | 0.418 | 4.96% |

The model identifies pitcher ID 685314 as the lowest simulated walk-off-risk candidate in this example.

The estimated difference between the lowest-risk and highest-risk candidates is 1.16 percentage points.

These results are provided as a reproducibility reference. Individual simulation estimates may vary slightly due to Monte Carlo randomness.

### Expected Program Output

A successful simulation prints information including:

- Historical game date and inning
- Number of outs and baserunner configuration
- Current pitching-team lead
- Model training cutoff date
- Number of Monte Carlo simulations per pitcher
- Candidate pitcher rankings
- Expected runs allowed
- Scoring and walk-off probabilities, when applicable

The program also generates visualization outputs under:

```text
results/bullpen_demo/
```

Outputs are organized by game date, matchup, and inning.

For example, the Arizona–Minnesota simulation generates files within the corresponding September 12, 2025 matchup directory.

The exact filenames and directory contents depend on the simulation output.

### Reproducing Presentation Results

The examples above are intended to provide directly executable demonstrations of the bullpen decision-support workflow.

To reproduce a presented result:

1. Clone the repository and install the dependencies.
2. Use the pretrained model included in `results/bullpen_quick/`.
3. Run the exact game command associated with the presented example.
4. Review the terminal rankings and generated visualization files.
5. Compare the simulated expected runs and scoring probabilities with the presentation results.

The model does not need to be retrained to execute these demonstrations.

## Model Training

The training pipeline is implemented in:

```text
bullpen_train_quick.py
```

Historical Statcast training data is stored locally in the `data/` directory and is not included in this repository because of its size.

The repository includes the pretrained XGBoost model and associated feature-history artifacts required for the demonstrated simulations.

Reproducing model training from scratch requires obtaining the corresponding historical data and preparing it for the training pipeline.

**Scope of reproducibility:** The provided commands reproduce the simulation workflow using the included pretrained model. Full end-to-end retraining from raw Statcast data is not provided as a one-command workflow in this checkpoint.

## Simulation Assumptions

The current implementation uses several simplifying assumptions:

- Each candidate relief pitcher is assumed to pitch the remainder of the evaluated half-inning.
- Plate-appearance probabilities are estimated using the pretrained XGBoost model.
- Pitcher fatigue is not dynamically updated during a simulated half-inning.
- Same-day pitch counts may be used for eligibility filtering but are not direct XGBoost prediction features.
- Baserunning and game-state transitions are simplified relative to all possible MLB situations.
- The system uses publicly available information and cannot fully reconstruct every bullpen availability decision.

Special situations, particularly postseason extra innings, require additional verification because postseason rules differ from regular-season extra-inning rules.

## Limitations

The current system has several limitations.

**Bullpen Availability**

Publicly available data cannot fully determine pitcher health, warm-up status, coaching instructions, or actual game-day availability.

**Prediction Uncertainty**

The XGBoost model estimates outcome probabilities using historical data. Predictions may be less reliable for players or matchup situations with limited feature coverage.

**Simplified Game Simulation**

The Monte Carlo framework approximates plate appearances and baserunning rather than reproducing every detail of an MLB game.

**Monte Carlo Variability**

Simulation estimates may vary slightly between executions. Small differences in projected probabilities should not automatically be interpreted as meaningful differences in pitcher quality.

**Historical Evaluation**

The current implementation is designed for historical game analysis. Its recommendations should not be interpreted as validated real-time MLB bullpen decisions.

**Decision Support, Not Decision Replacement**

The objective is to provide quantitative information that can support managerial decisions, not to replace coaching expertise or game-specific judgment.

## Project Status and Future Work

This repository represents the Checkpoint #1 implementation of an ongoing MLB bullpen optimization research project.

Current capabilities include:

- Historical MLB game-state reconstruction
- XGBoost-based plate-appearance outcome prediction
- Monte Carlo evaluation of potential relief pitchers
- Expected-runs and walk-off-risk comparisons
- Automated pitcher rankings and visualization outputs

Planned improvements include:

- More comprehensive model validation and calibration
- Improved bullpen availability and fatigue modeling
- More realistic baserunning and game-state transitions
- Better handling of special game situations and postseason rules
- Expanded historical case studies
- Evaluation of decision quality against observed managerial choices

## Intended Use

This project is developed for academic research and educational purposes.

Simulation outputs represent estimated probabilities under the model's assumptions and should not be interpreted as guaranteed baseball outcomes.
