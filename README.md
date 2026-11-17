
# MLB Bullpen Optimization: Machine Learning–Driven Pitching Decisions

## Project Overview

This project develops a machine learning–based decision support system for MLB bullpen management. It combines an XGBoost plate-appearance outcome prediction model with Monte Carlo simulation to evaluate potential relief pitchers in historical MLB game situations.

The goal is to identify which available pitching option has the lowest estimated risk under a specific game state.

The system considers the inning, score differential, number of outs, baserunner configuration, upcoming batting order, and candidate pitchers.

Unlike a traditional prediction model that produces a single outcome, this system simulates thousands of possible outcomes for each pitcher and compares their expected performance.

### Checkpoint #2 Update

The latest version introduces two major improvements:

1. **Game-State Reconstruction:** The system can evaluate pitching decisions at different points within an inning, including situations with 0, 1, or 2 outs and different baserunner configurations.

2. **Prediction vs. Reality:** The system compares the model-recommended pitcher, the manager-selected pitcher, and the observed outcome from the historical game.

These additions extend the project from a pitcher-ranking model into a more comprehensive retrospective bullpen decision-support system.

---

## Research Question

**Which relief pitcher offers the lowest estimated risk in a specific MLB game situation?**

The system evaluates pitching alternatives under identical game conditions and ranks them according to a context-dependent objective.

Two primary objectives are considered:

- **Run Prevention:** Minimize expected runs allowed during the remainder of the half-inning.
- **Walk-off Prevention:** Minimize the probability that the opposing team scores enough runs to win immediately in an applicable late-game situation.

The project also investigates how model recommendations compare with actual managerial decisions and observed game outcomes.

---

## Methodology

The system consists of five main stages.

### 1. Historical Game-State Reconstruction

Historical MLB game information is retrieved through the MLB Stats API.

The system reconstructs the game situation immediately before a selected plate appearance.

The reconstructed state includes:

- Game date and inning
- Top or bottom of the inning
- Current score differential
- Number of outs
- Runners on first, second, and third base
- Upcoming batting order
- Pitcher actually used
- Potential alternative pitchers

The latest implementation supports decisions occurring at different points within the same inning.

For example, the model can evaluate:

- 0 outs, bases empty
- 1 out, runner on first
- 1 out, bases loaded
- 2 outs, runners on first and second

The `--outs` argument selects the desired number of outs.

If multiple plate appearances begin with the same number of outs, the `--state-index` argument identifies the specific decision state.

The system reconstructs these states using historical plate-appearance records and recorded runner movements.

### 2. XGBoost Plate-Appearance Prediction

A pretrained XGBoost multiclass classification model predicts six possible plate-appearance outcomes:

1. Single
2. Extra-base hit (double or triple)
3. Walk or hit-by-pitch
4. Home run
5. Strikeout
6. Other out

The model estimates the probability distribution of these outcomes using historical Statcast-derived features and pitcher-batter information.

**Training data cutoff: September 4, 2025.**

All historical evaluation cases presented in this checkpoint occur after the training cutoff.

The model is not retrained when evaluating different game states.

### 3. Monte Carlo Simulation

For each candidate pitcher, the system simulates the remainder of the selected half-inning.

The simulation starts from the reconstructed game state rather than assuming every inning begins with zero outs and empty bases.

At each simulated plate appearance:

1. The model estimates the probability of each possible outcome.
2. An outcome is sampled from the predicted distribution.
3. The simulated outs, baserunners, and score are updated.
4. The simulation continues until the half-inning ends or a walk-off occurs.

The system produces:

- Expected runs allowed
- Probability of allowing at least one run
- Probability of allowing at least two runs
- Walk-off probability, when applicable

The number of simulations is configurable using `--sims`.

For the main Checkpoint #2 evaluation, **100,000 Monte Carlo simulations per pitcher** were used.

Smaller simulation counts can be used for faster demonstrations.

### 4. Pitcher Ranking

Candidate pitchers are ranked using the objective appropriate for the selected game situation.

**Run Prevention**

The system ranks pitchers by expected runs allowed.

**Walk-off Prevention**

In applicable late-game situations, pitchers are ranked by estimated walk-off probability.

Lower estimated risk is preferred.

The model recommendation represents the lowest estimated risk among the candidate pitchers considered by the system.

Candidate availability remains provisional because historical public data does not fully establish pitcher health, rest, warm-up status, or other managerial constraints.

### 5. Prediction vs. Reality

Checkpoint #2 introduces retrospective comparisons between model predictions and observed historical outcomes.

The comparison includes:

1. **Model Recommended Pitcher:** The pitcher ranked first by the simulation.
2. **Manager Selected Pitcher:** The pitcher actually used in the historical game, evaluated using the same model.
3. **Observed Game Outcome:** The actual number of runs scored after the selected decision point, or whether a walk-off occurred.

The observed outcome is retrieved from historical MLB game data.

For run-prevention scenarios, the observed result is the batting team's runs from the selected decision point through the end of the half-inning.

This is not necessarily the official earned-run total charged to the selected pitcher, particularly if additional pitching changes occur.

For walk-off scenarios, the observed outcome indicates whether the batting team actually completed a walk-off.

This comparison provides additional context for interpreting the model's recommendations.

However, a single observed outcome cannot establish whether a probabilistic prediction was accurate or whether an alternative managerial decision would have produced a better result.

---

## Repository Structure

```text
mlb-bullpen-optimization/
├── bullpen_train_quick.py
├── bullpen_predict.py
├── bullpen_game_setup.py
├── simulation.py
├── simulation_game_state.py
├── requirements.txt
├── README.md
└── results/
    ├── bullpen_quick/
    │   ├── xgb_matchup_model.json
    │   ├── model_metadata.json
    │   ├── feature_history.pkl
    │   └── pitcher_daily_workload.parquet
    └── bullpen_demo/
```

### Main Components

| File | Purpose |
|---|---|
| `bullpen_train_quick.py` | XGBoost training pipeline |
| `bullpen_predict.py` | Feature construction and model prediction |
| `bullpen_game_setup.py` | Historical game setup and data preparation |
| `simulation.py` | Original inning-start bullpen simulation |
| `simulation_game_state.py` | Game-state reconstruction, Monte Carlo evaluation, pitcher ranking, observed-outcome comparison, and visualization |
| `requirements.txt` | Python dependencies |
| `results/bullpen_quick/` | Pretrained model and historical feature data |
| `results/bullpen_demo/` | Generated simulation dashboards |

---

## Installation

### Requirements

- Python 3.10 or later, subject to dependency compatibility
- Git
- Internet connection for historical MLB data retrieval
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
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Verify Dependencies

```bash
python -c "import xgboost, pandas, numpy, sklearn, matplotlib, pybaseball; print('Dependencies loaded successfully')"
```

The pretrained model is included in the repository.

Retraining is not required to reproduce the bullpen simulations.

---

## Reproducing Checkpoint #2 Results

### Supported Dates

Automatic historical game lookup currently supports games from September 5, 2025, through December 31, 2025.

The XGBoost training data cutoff is September 4, 2025.

### General Command

```bash
python simulation_game_state.py \
  --date YYYY-MM-DD \
  --team TEAM \
  --opponent OPPONENT \
  --inning N \
  --outs N \
  --sims 100000
```

### Command Arguments

| Argument | Description |
|---|---|
| `--date` | Historical game date |
| `--team` | Pitching team being evaluated |
| `--opponent` | Opposing team |
| `--inning` | Inning to evaluate |
| `--outs` | Number of outs at the selected decision point (0, 1, or 2) |
| `--state-index` | Selects a specific matching state if multiple states exist |
| `--sims` | Number of Monte Carlo simulations per candidate |
| `--game-pk` | Optional MLB game ID for identifying a specific game |
| `--include-starters` | Includes likely starting pitchers in the candidate pool |
| `--exclude-pitchers` | Manually excludes specified pitcher IDs |

**Important:** `--team` refers to the pitching team, not necessarily the home team.

If multiple matching decision states exist, the program displays their indices and requires `--state-index` to identify the desired state.

---

## Checkpoint #2 Results

### Case 1: Los Angeles Dodgers at Baltimore Orioles

**Date:** September 5, 2025

**Pitching team:** Baltimore Orioles

**Batting team:** Los Angeles Dodgers

**Game state:**

- Top of the ninth inning
- 0 outs
- Bases empty
- Score tied 1–1

**Decision objective:** Minimize expected runs allowed.

### Reproduction Command

```bash
python simulation_game_state.py \
  --date 2025-09-05 \
  --team BAL \
  --opponent LAD \
  --inning 9 \
  --outs 0 \
  --sims 100000
```

### Simulation Results

| Pitcher | Expected Runs | P(≥1 Run) | P(≥2 Runs) |
|---|---:|---:|---:|
| Keegan Akin | 0.384 | 20.01% | 10.34% |
| Shawn Dubin | 0.409 | 20.69% | 11.09% |
| Yennier Cano (Actual) | 0.418 | 20.80% | 11.40% |
| Tyler Wells | 0.419 | 21.27% | 11.41% |
| Albert Suárez | 0.431 | 21.34% | 11.82% |

The model recommends **Keegan Akin**, while the actual pitcher selected by the manager was **Yennier Cano**.

### Prediction vs. Reality

| Comparison | Result |
|---|---|
| Model Recommended | Keegan Akin |
| Predicted Expected Runs — Akin | 0.384 |
| Manager Selected | Yennier Cano |
| Predicted Expected Runs — Cano | 0.418 |
| Actual Runs Scored After Decision | 0 |

The model-estimated decision gap is:

**0.418 − 0.384 = 0.034 expected runs.**

The model estimates that Keegan Akin would allow slightly fewer runs on average under the same simulated conditions.

However, the Orioles did not allow a run during the observed remainder of the half-inning.

This does not establish that the manager made an incorrect decision.

The difference between the two pitchers is small, and the actual game represents only one observed outcome.

---

### Case 2: New York Mets at Cincinnati Reds

**Date:** September 5, 2025

**Pitching team:** New York Mets

**Batting team:** Cincinnati Reds

**Game state:**

- Bottom of the ninth inning
- 1 out
- Bases loaded
- Mets leading 5–4

**Decision objective:** Minimize walk-off probability.

### Reproduction Command

```bash
python simulation_game_state.py \
  --date 2025-09-05 \
  --team NYM \
  --opponent CIN \
  --inning 9 \
  --outs 1 \
  --state-index 0 \
  --sims 10000
```

The selected state should be verified against the state information printed by the program.

### Simulation Results

| Pitcher | Estimated Walk-off Probability |
|---|---:|
| Edwin Díaz | 28.95% |
| Gregory Soto | 30.11% |
| Justin Hagenman | 30.46% |
| Ryan Helsley | 32.79% |
| Wander Suero | 32.95% |
| Kevin Herget | 33.12% |

The model recommends **Edwin Díaz**, who was also the pitcher actually selected by the manager.

### Prediction vs. Reality

| Comparison | Result |
|---|---|
| Model Recommended | Edwin Díaz |
| Manager Selected | Edwin Díaz |
| Predicted Walk-off Probability | 28.95% |
| Actual Walk-off | No |

The model estimated a 28.95% probability of a walk-off under the simulated game conditions.

The observed game did not end in a walk-off.

This is consistent with the model assigning a higher probability to avoiding a walk-off than to allowing one.

A single binary outcome does not establish probability calibration.

**Note:** These results were produced using 10,000 simulations per pitcher. The 100,000-simulation result for this case may differ slightly.

---

## Checkpoint #1 Reference Results

The original version evaluated pitching decisions at the start of a half-inning.

The following historical cases were included in Checkpoint #1.

### Case A: Dodgers Pitching Against Orioles

**Date:** September 5, 2025

**Inning:** 8

**Pitching team:** Los Angeles Dodgers

```bash
python simulation.py \
  --date 2025-09-05 \
  --team LAD \
  --opponent BAL \
  --inning 8
```

| Pitcher | Expected Runs Allowed |
|---|---:|
| Justin Wrobleski | 0.317 |
| Michael Kopech (Actual) | 0.330 |
| Tanner Scott | 0.332 |
| Kirby Yates | 0.345 |
| Edgardo Henriquez | 0.377 |

The model ranks Justin Wrobleski first, while Michael Kopech was the actual pitching choice.

The difference is 0.013 expected runs.

### Case B: Pirates Pitching Against Orioles

**Date:** September 9, 2025

**Inning:** 11

**Pitching team:** Pittsburgh Pirates

```bash
python simulation.py \
  --date 2025-09-09 \
  --team PIT \
  --opponent BAL \
  --inning 11
```

| Pitcher | Estimated Walk-off Probability |
|---|---:|
| Yohan Ramírez | 34.52% |
| Evan Sisk | 35.87% |
| Dauri Moreta (Actual) | 36.16% |
| Bubba Chandler | 38.60% |
| Carmen Mlodzinski | 38.90% |
| Colin Holderman | 40.65% |

The model ranks Yohan Ramírez first.

The actual pitcher, Dauri Moreta, has a model-estimated walk-off probability 1.64 percentage points higher.

These are simulated estimates rather than evidence that a different managerial decision would necessarily have changed the result.

---

## Additional Historical Test Cases

The original `simulation.py` also supports the following examples.

### Kansas City Royals vs. Cleveland Guardians

```bash
python simulation.py \
  --date 2025-09-11 \
  --team KC \
  --opponent CLE \
  --inning 8
```

### Arizona Diamondbacks vs. Minnesota Twins

```bash
python simulation.py \
  --date 2025-09-12 \
  --team AZ \
  --opponent MIN \
  --inning 9
```

### Philadelphia Phillies vs. Los Angeles Dodgers

```bash
python simulation.py \
  --date 2025-10-09 \
  --team PHI \
  --opponent LAD \
  --inning 11
```

**Postseason caution:** MLB postseason extra innings do not use the regular-season automatic runner rule. This example requires additional validation before drawing baseball conclusions.

---

## Visualization Outputs

The updated `simulation_game_state.py` generates two visualization files.

### 1. Bullpen Ranking Dashboard

**File:** `bullpen_dashboard.png`

Displays:

- Candidate pitcher rankings
- Expected runs allowed or walk-off probability
- Probability of allowing at least one run
- Probability of allowing at least two runs
- Model-recommended pitcher
- Manager-selected pitcher
- Confidence intervals for simulated walk-off probabilities

### 2. Prediction vs. Reality Dashboard

**File:** `bullpen_comparison.png`

Displays:

- Model-recommended pitcher's predicted performance
- Manager-selected pitcher's predicted performance
- Observed historical game outcome

The comparison is intended to distinguish model-estimated risk from observed game results.

For walk-off situations, the actual outcome is binary, while the model outputs a probability.

These should not be interpreted as equivalent statistical quantities.

### Output Directory

Generated dashboards are saved under:

```text
results/bullpen_demo/
    YYYY-MM-DD/
        OPPONENT-TEAM/
            inning_N/
                outs_N_state_K/
                    bullpen_dashboard.png
                    bullpen_comparison.png
```

The directory structure separates different games, innings, and decision states.

---

## Model Training

The model training pipeline is implemented in `bullpen_train_quick.py`.

Historical Statcast data is stored locally under `data/`.

The full raw training dataset is not included in the repository because of its size.

The repository includes the pretrained model and associated feature-history artifacts needed to reproduce the bullpen simulations.

Reproducing model training from scratch requires obtaining the historical training data and preparing the corresponding features.

**Reproducibility scope:** The project supports running bullpen simulations with the pretrained model. Full model retraining from raw data is not currently provided as a one-command workflow.

---

## Simulation Assumptions and Limitations

The current implementation uses several simplifying assumptions.

### Pitcher Availability

The system identifies potential pitchers using historical roster and pitching information.

However, historical public data cannot fully determine:

- Pitcher health
- Warm-up status
- Bullpen availability
- Rest and recovery
- Managerial restrictions

Therefore, the candidate pool is provisional.

### Simulation Assumptions

- Each candidate pitcher is assumed to pitch the remainder of the evaluated half-inning.
- Pitcher fatigue is not dynamically updated during the simulated inning.
- Baserunning transitions are simplified.
- Double plays, sacrifices, steals, and certain defensive events are not explicitly modeled.
- Same-day pitch counts may be used for candidate filtering but are not direct XGBoost prediction features.
- Future substitutions may not be fully represented in the reconstructed batting order.
- Special game situations require additional validation.

### Statistical Limitations

- Monte Carlo uncertainty does not include all sources of model uncertainty.
- Small differences between pitchers may not be practically meaningful.
- A single historical outcome cannot validate a probabilistic model.
- Observed outcomes do not establish the counterfactual performance of an alternative pitcher.
- Additional historical samples are needed for meaningful model calibration and evaluation.

The system is intended to support managerial decision-making rather than replace baseball judgment.

---

## Project Progress

### Completed

- Historical MLB data collection
- XGBoost plate-appearance outcome prediction
- Monte Carlo half-inning simulation
- Context-dependent pitcher ranking
- Historical game-state reconstruction
- Support for different outs and baserunner configurations
- Prediction vs. Reality comparison
- Automated visualization dashboards
- Retrospective MLB case studies
- Successful 100,000-simulation evaluation for BAL vs. LAD

### Future Work

1. Expand retrospective evaluation across more historical games.
2. Evaluate model calibration using larger samples of observed outcomes.
3. Improve pitcher availability and workload constraints.
4. Incorporate more realistic baserunning and game-state transitions.
5. Improve uncertainty reporting and statistical interpretation.
6. Extend the system toward more interactive decision-support workflows.

---

## Intended Use

This project was developed for academic research and educational purposes.

All reported probabilities and expected-run estimates are model-based predictions, not guaranteed real-world outcomes.

The system provides a framework for comparing pitching alternatives under simulated historical game conditions and investigating how those recommendations relate to actual managerial decisions.
