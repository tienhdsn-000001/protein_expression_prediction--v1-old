
# Dimension-Adjusted Feature Importance Analysis (Information Density)

This plot normalizes the total feature importance by the number of dimensions in each embedding. It aims to measure the *efficiency* or *information density* of each embedding type.

## How to Interpret This Graph

This plot answers the question: **"On average, how much predictive power does each dimension of an embedding contribute?"**

- **Y-Axis (Average Importance per Dimension):** This is the total importance (gain) for an embedding type divided by its number of features (e.g., 5120 for ESM-2). A higher bar suggests that the embedding packs more useful information into each of its dimensions.
- **Comparing Embeddings:** This view allows for a fairer comparison of the embeddings' intrinsic quality, independent of their size. A smaller, more efficient embedding might show a higher score here even if its total importance is lower than a much larger embedding.

## Key Observations

- **Efficiency vs. Total Power:** Compare this plot with the absolute importance plot. You might find cases where an embedding has high total importance but low per-dimension importance, suggesting its power comes from its large size rather than its efficiency. Conversely, a high score here indicates a well-structured, information-dense embedding.
- **Model Architecture Insights:** Consistently high scores for a smaller model (like RNA-ERNIE) might suggest that its architecture is particularly effective at capturing the relevant biological signals for these tasks.
