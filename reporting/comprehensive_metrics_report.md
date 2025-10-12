## Comprehensive Performance Metrics (100% Data)

This table provides a detailed evaluation of the final models...

| Target                   | Variant         |      R² |         MSE |     RMSE |      MAE |   Spearman's ρ |   Pearson's r |
|:-------------------------|:----------------|--------:|------------:|---------:|---------:|---------------:|--------------:|
| log2_ppm_abundance       | baseline        | -0.1454 |     21.6811 |   4.6563 |   3.6502 |         0.4910 |        0.5022 |
| log2_ppm_abundance       | embeddings_only |  0.4223 |     10.9358 |   3.3069 |   2.6295 |         0.6201 |        0.6541 |
| log2_tpm_abundance       | baseline        | -1.0025 |      7.1555 |   2.6750 |   2.0197 |         0.2696 |        0.2775 |
| log2_tpm_abundance       | embeddings_only |  0.1380 |      3.0800 |   1.7550 |   1.1662 |         0.3896 |        0.3875 |
| mrna_half_life           | baseline        | -0.8877 |     36.2072 |   6.0172 |   4.6743 |         0.1142 |        0.1162 |
| mrna_half_life           | embeddings_only |  0.0166 |     18.8626 |   4.3431 |   3.4081 |         0.1299 |        0.1494 |
| mrna_max_half_life_delta | baseline        | -0.5993 | 180539.7778 | 424.8997 | 238.3458 |        -0.0221 |        0.0015 |
| mrna_max_half_life_delta | embeddings_only | -0.0077 | 113753.2843 | 337.2733 |  67.1853 |         0.0782 |        0.0091 |
| mrna_min_half_life_delta | baseline        | -0.8370 |      8.4760 |   2.9114 |   2.2463 |         0.1390 |        0.1305 |
| mrna_min_half_life_delta | embeddings_only |  0.0272 |      4.4885 |   2.1186 |   1.6111 |         0.1124 |        0.1727 |
| protein_half_life        | baseline        | -0.3409 |      3.0263 |   1.7396 |   1.3391 |         0.2279 |        0.2251 |
| protein_half_life        | embeddings_only |  0.1266 |      1.9713 |   1.4040 |   1.0524 |         0.3281 |        0.3635 |