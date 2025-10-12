## Comprehensive Performance Metrics (100% Data)

This table provides a detailed evaluation of the final models on the holdout set, including measures of quantitative accuracy (R², MSE, RMSE, MAE), ranking accuracy (Spearman's ρ), and linear correlation (Pearson's r).

| Target                   | Variant         |      R² |         MSE |     RMSE |      MAE |   Spearman's ρ |   Pearson's r |
|:-------------------------|:----------------|--------:|------------:|---------:|---------:|---------------:|--------------:|
| log2_ppm_abundance       | baseline        | -0.2246 |     23.1810 |   4.8147 |   3.8179 |         0.2989 |        0.3195 |
| log2_ppm_abundance       | embeddings_only |  0.3527 |     12.2537 |   3.5005 |   2.8312 |         0.5651 |        0.5951 |
| log2_tpm_abundance       | baseline        | -0.5524 |      5.5472 |   2.3553 |   1.7143 |         0.1259 |        0.1043 |
| log2_tpm_abundance       | embeddings_only |  0.0313 |      3.4614 |   1.8605 |   1.2321 |         0.2219 |        0.2021 |
| mrna_half_life           | baseline        | -1.4256 |     46.5245 |   6.8209 |   5.3768 |         0.0720 |        0.0301 |
| mrna_half_life           | embeddings_only |  0.0210 |     18.7786 |   4.3334 |   3.3967 |         0.1521 |        0.1613 |
| mrna_max_half_life_delta | baseline        | -0.1992 | 599561.6355 | 774.3137 | 285.4510 |         0.1157 |       -0.0567 |
| mrna_max_half_life_delta | embeddings_only | -0.0060 | 502961.0940 | 709.1975 | 100.8859 |         0.1418 |       -0.0313 |
| mrna_min_half_life_delta | baseline        | -1.4332 |     11.2274 |   3.3507 |   2.6225 |         0.0859 |        0.0369 |
| mrna_min_half_life_delta | embeddings_only |  0.0191 |      4.5262 |   2.1275 |   1.6244 |         0.1138 |        0.1557 |
| protein_half_life        | baseline        | -1.8046 |      6.3301 |   2.5160 |   1.9260 |         0.0975 |        0.0356 |
| protein_half_life        | embeddings_only |  0.1382 |      1.9452 |   1.3947 |   1.0463 |         0.3898 |        0.3804 |