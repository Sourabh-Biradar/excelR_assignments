| Feature              | `df.corr()`                          | `pps.matrix()`                              |
| -------------------- | ------------------------------------ | ------------------------------------------- |
| Type of relationship | Linear only                          | Linear + Non-linear                         |
| Data types           | Numeric only                         | Numeric + Categorical                       |
| Symmetry             | Symmetric                            | Asymmetric                                  |
| Values               | -1 to 1                              | 0 to 1                                      |
| Use case             | Quick check for linear relationships | More general, predictive feature importance |




Measures how well one feature can predict another, using ML models under the hood.
Range: 0 to 1.
0: no predictive power
1: perfect predictor
