## Model Performance Metrics (100% Data)

### Coefficient of Determination (R²)

| target                   |   baseline |   embeddings_only |
|:-------------------------|-----------:|------------------:|
| mrna_half_life           |    -1.4256 |            0.0210 |
| mrna_max_half_life_delta |    -0.1992 |           -0.0060 |
| mrna_min_half_life_delta |    -1.4332 |            0.0191 |
| log2_tpm_abundance       |    -0.5491 |            0.0321 |
| protein_half_life        |    -1.8046 |            0.1382 |
| log2_ppm_abundance       |    -0.2246 |            0.3527 |

### Mean Squared Error (MSE)

| target                   |    baseline |   embeddings_only |
|:-------------------------|------------:|------------------:|
| mrna_half_life           |     46.5245 |           18.7786 |
| mrna_max_half_life_delta | 599561.6350 |       502961.0941 |
| mrna_min_half_life_delta |     11.2274 |            4.5262 |
| log2_tpm_abundance       |      5.5344 |            3.4578 |
| protein_half_life        |      6.3301 |            1.9452 |
| log2_ppm_abundance       |     23.1810 |           12.2537 |
