
# Normalized Feature Importance Analysis

This plot shows the *relative* importance of each embedding type, normalized by its own maximum performance across all tasks.

## How to Interpret This Graph

This plot answers a different question than the absolute importance plot. Instead of asking "Which embedding is most useful for this task?", it asks, **"For a given embedding, which task leverages its information the most?"**

- **Y-Axis (Normalized Importance):** For each embedding type (each color), the prediction target where it was most important is given a score of 1. All other bars for that color show how important the embedding was for other tasks *relative to its own peak performance*.
- **Example:** As you noted, the purple bar (ESM-2/Protein) is at 1.0 for `protein_half_life`, indicating this is the task where ESM-2's information is most valuable. The purple bar for `log2_tpm_abundance` is at ~0.6, meaning the protein embedding is about 60% as useful for that task as it is for its best task.

## Key Observations

- **Task Specialization:** This view highlights the "specialization" of each embedding. We can clearly see which tasks are the primary use-case for each modality.
- **Complementary Information:** Comparing this to the absolute importance plot is crucial. An embedding might have a low *absolute* importance for all tasks, but this plot would still show a bar at 1.0 for its "best" task, which could be misleading if viewed in isolation.

This analysis provides a more nuanced understanding of not just *what* information is important, but *where* it is most effectively applied.
