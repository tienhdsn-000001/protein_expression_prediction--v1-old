
# Understanding the Bayesian Optimization Convergence Plots

The plots titled "Bayesian Optimization Convergence" illustrate how the hyperparameter tuning process found progressively better models.

## How to Read the Graph

-   **X-axis (Optimization Call):** This represents each "step" or "trial" in the optimization process. For each step, the optimizer chooses a new set of hyperparameters to try.
-   **Y-axis (Best R² Found):** This shows the best cross-validated R² score discovered *up to that point*.

An ideal convergence plot shows the R² value rapidly increasing in the early calls and then plateauing. This indicates that the optimizer has successfully explored the hyperparameter space and settled on a high-performing region.

## What are "Training Steps" in Bayesian Optimization?

Unlike neural networks that train over thousands of *epochs* or *steps* with gradient descent, the "training" in this context is different. Each **"call"** or **"step"** on the x-axis represents:

1.  **Selecting Hyperparameters:** The Bayesian optimizer picks a promising set of hyperparameters (e.g., `n_estimators`, `learning_rate`).
2.  **Full Model Training & Cross-Validation:** An XGBoost model is trained from scratch with these hyperparameters on the training data, and its performance is evaluated using 3-fold group cross-validation.
3.  **Recording the Score:** The average R² from the cross-validation is recorded.
4.  **Updating the Belief:** The optimizer uses this new result to update its internal probability model of which hyperparameter sets are most likely to yield the best results.

This cycle repeats for the total number of calls (`N_OPTIMIZATION_CALLS` = 25 in the script). The plot tracks the best score found throughout this entire search process.
