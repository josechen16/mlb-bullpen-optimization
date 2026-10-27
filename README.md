# MLB Bullpen Optimization: Machine Learning–Driven Pitching Decisions

## Project Overview

This project develops a machine learning–based decision support system for MLB bullpen management. It combines an XGBoost plate-appearance outcome prediction model with Monte Carlo simulation to evaluate potential relief pitchers in historical game situations.

The objective is to compare bullpen alternatives under the same game conditions and identify pitchers who minimize expected runs allowed or walk-off risk.

Rather than predicting a single game outcome, the system estimates the risk associated with different pitching decisions.

This repository contains the Checkpoint #1 implementation of an ongoing academic research project.

## Research Question

**Which relief pitcher offers the lowest estimated risk in a specific MLB game situation?**

The decision depends on the inning, score, base-out state, upcoming batting order, and potential relief pitchers.

The system produces a ranked comparison of pitching alternatives using a context-dependent objective.

## Methodology

The system consists of four stages.

### 1. Historical Game Reconstruction

Historical MLB game information is retrieved through the MLB Stats API.

The system reconstructs the selected game situation, including:

- Date and inning
- Score differential
- Number of outs
- Baserunner configuration
- Upcoming batting order
- Potential relief pitchers

### 2. XGBoost Outcome Prediction

A pretrained XGBoost multiclass classification model predicts six plate-appearance outcomes:

1. Single
2. Extra-base hit (double or triple)
3. Walk or hit-by-pitch
4. Home run
5. Strikeout
6. Other out

The model uses historical Statcast-derived features to estimate outcome probabilities for pitcher–batter matchups.

The training data cutoff is September 4, 2025.

### 3. Monte Carlo Simulation

For each candidate pitcher, the system simulates the remainder of the half-inning 10,000 times.

During each simulation, the system samples plate-appearance outcomes from the XGBoost predictions and updates the simulated game state.

The simulation produces estimates of:

- Expected runs allowed
- Probability of allowing at least one run
- Probability of allowing at least two runs
- Walk-off probability, when applicable

### 4. Pitcher Ranking

The decision objective depends on the game situation.

**Run Prevention:** Pitchers are ranked by expected runs allowed.

**Walk-off Prevention:** In applicable late-game situations, pitchers are ranked by the simulated probability of allowing a walk-off.

Lower estimated risk is preferred.

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
        └── [model evaluation artifacts]
```

### Main Components

| File | Purpose |
|---|---|
| `simulation.py` | Historical game reconstruction, bullpen evaluation, Monte Carlo simulation, and visualization |
| `bullpen_train_quick.py` | XGBoost training pipeline |
| `bullpen_predict.py` | Model prediction functionality |
| `bullpen_game_setup.py` | Historical game setup and data preparation |
| `requirements.txt` | Python dependencies |
| `results/bullpen_quick/` | Pretrained model, historical features, workload data, and evaluation artifacts |

Simulation dashboards are generated under `results/bullpen_demo/`.

## Installation

### Requirements

- Python 3.10 or later is recommended, subject to dependency compatibility.
- Git
- Internet connection for historical MLB game retrieval
- Dependencies listed in `requirements.txt`

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

Activate on macOS or Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Step 4: Verify Dependencies

```bash
python3 -c "import xgboost, pandas, numpy, sklearn, matplotlib, pybaseball; print('Dependencies loaded successfully')"
```

The pretrained model is included in the repository. Retraining is not required to execute the examples below.

## Reproducing the Analysis

### Supported Dates

Automatic game lookup currently supports historical MLB games from **September 5, 2025, through December 31, 2025**.

The XGBoost training data cutoff is September 4, 2025, so the evaluated games occur after the training period.

Games must exist on the specified date, and the selected inning must be valid.

### Command Format

```bash
python3 simulation.py --date YYYY-MM-DD --team TEAM --opponent OPPONENT --inning N
```

| Argument | Description |
|---|---|
| `--date` | Historical game date |
| `--team` | Pitching team being evaluated |
| `--opponent` | Batting team |
| `--inning` | Inning to evaluate |

**Important:** `--team` refers to the pitching team, not necessarily the home team.

For the Arizona Diamondbacks, use `AZ` rather than `ARI`.

## Checkpoint #1 Presentation Results

The following two cases correspond to the results presented in the Checkpoint #1 slides.

### Case A: Los Angeles Dodgers at Baltimore Orioles

**Date:** September 5, 2025  
**Inning:** 8  
**Pitching team:** Los Angeles Dodgers  
**Batting team:** Baltimore Orioles  
**Decision objective:** Minimize expected runs allowed

Run:

```bash
python3 simulation.py --date 2025-09-05 --team LAD --opponent BAL --inning 8
```

Reference results:

| Pitcher | Expected Runs Allowed |
|---|---:|
| Justin Wrobleski | 0.317 |
| Michael Kopech (actual) | 0.330 |
| Tanner Scott | 0.332 |
| Kirby Yates | 0.345 |
| Edgardo Henriquez | 0.377 |

The model ranks Justin Wrobleski first, while Michael Kopech was the actual pitching choice.

The difference between their simulated expected runs allowed is **0.013 runs**.

This difference is small and does not establish that the model's recommendation was superior.

### Case B: Pittsburgh Pirates at Baltimore Orioles

**Date:** September 9, 2025  
**Inning:** 11  
**Pitching team:** Pittsburgh Pirates  
**Batting team:** Baltimore Orioles  
**Decision objective:** Minimize walk-off probability

Run:

```bash
python3 simulation.py --date 2025-09-09 --team PIT --opponent BAL --inning 11
```

Reference results:

| Pitcher | Estimated Walk-off Probability |
|---|---:|
| Yohan Ramírez | 34.52% |
| Evan Sisk | 35.87% |
| Dauri Moreta (actual) | 36.16% |
| Bubba Chandler | 38.60% |
| Carmen Mlodzinski | 38.90% |
| Colin Holderman | 40.65% |

This example evaluates a tied game in the bottom of the eleventh inning, with an automatic runner starting on second base.

The model ranks Yohan Ramírez as the lowest simulated walk-off-risk candidate.

The actual pitcher, Dauri Moreta, has an estimated walk-off probability of 36.16%, compared with 34.52% for Yohan Ramírez.

The difference is **1.64 percentage points**.

These estimates are not evidence that a different managerial decision would necessarily have changed the outcome.

## Additional Live Demonstrations

The following examples demonstrate the same system on additional historical games.

### Example 1: Kansas City Royals vs. Cleveland Guardians

**Date:** September 11, 2025  
**Inning:** 8  
**Pitching team:** Kansas City Royals

```bash
python3 simulation.py --date 2025-09-11 --team KC --opponent CLE --inning 8
```

### Example 2: Arizona Diamondbacks vs. Minnesota Twins

**Date:** September 12, 2025  
**Inning:** 9  
**Pitching team:** Arizona Diamondbacks

```bash
python3 simulation.py --date 2025-09-12 --team AZ --opponent MIN --inning 9
```

In this example, Arizona enters the bottom of the ninth inning with a two-run lead.

Reference results from a successful local execution:

| Pitcher ID | Throws | Expected Runs | Walk-off Probability |
|---|---|---:|---:|
| 685314 | L | 0.343 | 3.80% |
| 543101 | R | 0.391 | 4.41% |
| 688497 | R | 0.397 | 4.68% |
| 663765 | R | 0.416 | 4.95% |
| 595928 | R | 0.418 | 4.96% |

Pitcher ID 685314 has the lowest simulated walk-off probability in this evaluation.

The estimated difference between the lowest-risk and highest-risk candidates is 1.16 percentage points.

### Example 3: Philadelphia Phillies vs. Los Angeles Dodgers

**Date:** October 9, 2025  
**Inning:** 11  
**Pitching team:** Philadelphia Phillies  
**Batting team:** Los Angeles Dodgers

```bash
python3 simulation.py --date 2025-10-09 --team PHI --opponent LAD --inning 11
```

**Postseason caution:** This example is included as an additional test case, not a validated reference result. MLB postseason extra innings do not use the regular-season automatic runner rule. The program's handling of this situation must be verified before using its output for baseball conclusions.

## Expected Outputs

After a successful simulation, the terminal displays:

- Game date and inning
- Number of outs and occupied bases
- Pitching-team score differential
- Model training cutoff
- Number of Monte Carlo simulations
- Candidate pitcher rankings
- Expected runs allowed
- Relevant scoring or walk-off probabilities

The program also saves visualization outputs under:

```text
results/bullpen_demo/
```

Results are organized by game date, matchup, and inning.

To reproduce a result from the presentation:

1. Install the project dependencies.
2. Use the included pretrained XGBoost model.
3. Run the exact command for Case A or Case B.
4. Compare the terminal rankings and generated visualization with the reference results above.

Monte Carlo results may vary slightly between runs because of simulation randomness.

## Model Training

The model training pipeline is implemented in `bullpen_train_quick.py`.

Historical Statcast data is stored locally under `data/` and is not included in the GitHub repository because of its size.

The pretrained model and associated feature-history artifacts are included to support reproduction of the bullpen simulations.

Reproducing model training from scratch requires obtaining the historical training data and preparing the corresponding features.

**Reproducibility scope:** This checkpoint supports reproducing bullpen simulations with the pretrained model. Full retraining from raw data is not provided as a one-command workflow.

## Simulation Assumptions and Limitations

The current implementation includes several simplifying assumptions:

- Each candidate pitcher is assumed to pitch the remainder of the evaluated half-inning.
- Pitcher fatigue is not dynamically updated during the simulated inning.
- Same-day pitch counts may be used for eligibility filtering but are not direct XGBoost prediction features.
- Baserunning and game-state transitions are simplified.
- Public data cannot fully determine pitcher health, warm-up status, or bullpen availability.
- Candidate eligibility and historical availability may be incomplete.
- Model predictions depend on historical feature coverage.
- Small estimated differences may not be statistically or practically meaningful.
- Monte Carlo uncertainty does not account for all sources of model uncertainty.
- Special game situations, including postseason extra innings, require additional validation.

The system is designed to support baseball decision-making rather than replace managerial judgment.

## Project Status and Future Work

This repository represents the Checkpoint #1 implementation of an ongoing MLB bullpen optimization research project.

Current capabilities include historical game reconstruction, XGBoost-based outcome prediction, Monte Carlo bullpen evaluation, context-specific pitcher rankings, and visualization dashboards.

Future work will focus on:

- Improved pitcher availability and fatigue modeling
- More realistic game-state transitions
- Model validation and calibration
- Better handling of postseason and special game rules
- Expanded historical case studies
- Evaluation of decision quality against actual managerial decisions

## Intended Use

This project is developed for academic research and educational purposes.

All reported simulation probabilities are model-based estimates and should not be interpreted as guaranteed real-world outcomes.
