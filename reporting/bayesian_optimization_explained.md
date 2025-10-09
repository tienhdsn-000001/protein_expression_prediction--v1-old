
# Bayesian Optimization Explained

These plots illustrate the process of hyperparameter tuning for each model.

## How to Interpret the Convergence Plots

- **X-Axis (Optimization Call):** Represents each time the optimizer trained and evaluated a new model with a different set of hyperparameters (like `learning_rate`, `max_depth`, etc.).
- **Y-Axis (Best R² Found):** Shows the best cross-validation R² score found *up to that point*. The line will only ever go up or stay flat, as it tracks the best performance seen so far.
- **What is a "Training Step"?** In this context, a "training step" or "optimization call" is a full cycle of:
    1. The Bayesian optimizer picking a new set of hyperparameters it thinks might be promising.
    2. Training a complete XGBoost model from scratch using those parameters on folds of the training data.
    3. Evaluating the model on a validation fold.
    4. Repeating this across all cross-validation folds to get a robust R² score.
    5. Feeding that score back to the optimizer, which updates its internal "map" of the hyperparameter space.

A steep upward slope indicates the optimizer is quickly finding better parameter combinations. A long plateau means it is struggling to find improvements over the current best-known parameters.
