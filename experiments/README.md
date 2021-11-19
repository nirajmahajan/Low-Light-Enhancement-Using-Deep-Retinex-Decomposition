# Experiments for Tuning Hyper parameters

| No     | Description                                                  | Results   |
| ------ | :----------------------------------------------------------- | --------- |
| 1      | Hyper parameters Used by the Paper                           | Failed    |
| 2      | Joint Decomposition Loss (0.001 --> 1)                       | Failed    |
| 3      | SGD lr 1e-3                                                  | Failed    |
| 4      | SGD lr 1e-2                                                  | Failed    |
| 5      | Corrected gradient computation + lr 1e-3                     | Failed    |
| 6      | Adam lr 1e-1                                                 | Failed    |
| 7      | Adam lr 1e-2                                                 | Failed    |
| 8      | Experiment 5 with batch norm + sobel filters for grad        | Good      |
| 9      | Experiment 8 without sobel filters                           | Good      |
| 10     | Experiment 8 without sobel filters; large patch size (96)    | Better    |
| 11     | No patch size - Consider Entire Image                        | Failed    |
| 12     | Patch size 144                                               | Failed    |
| 13     | Patch size 120                                               | Mediocre  |
| **14** | **Larger lambda for Reconstruction loss (S_high)**           | **Best**  |
| 15     | Larger lambda for Smooth loss                                | Poor      |
| 16     | detached backprop for Relight Loss                           | Failed    |
| 17     | Complete connected backprop for Relight Loss                 | Failed    |
| 18     | Experiment 10 with discriminative loss only on R             | Poor      |
| 19     | Experiment 10 with discriminative loss on both R, I          | Failed    |
| 20     | Experiment 18 with a different lamda for discriminative loss | Failed    |
| 21     | Experiment 19 with a different lamda for discriminative loss | Failed    |
| 22     | Experiment 14 with undetached R_low for L_relight            | Failed    |
| 23     | Experiment 14 with lr 1e-4                                   | Mediocre  |
| 24     | Experiment 14 with much more recon loss                      | Failed    |
| 25     | Experiment 14 with different recon loss                      | Mediocre  |
| 26     | Experiment 18 with 14 lambda combination - weaker discriminative loss | Failed    |
| 27     | Experiment 18 with 14 lambda combination - weak discriminative loss | Mediocre  |
| 28     | Experiment 14 with low window size (48)                      | Good      |
| 29     | Experiment 14 with lower window size (32)                    | Mediocre  |
| 30     | Experiment 14 with medium window size (64)                   | Mediocre  |
| 31     | Experiment 14 with Sobel, less smooth                        | Mediocre  |
| 32     | Experiment 14 with larger dataset (Brightness data included) | Near Best |
| 33     | Experiment 14 with more epochs                               | Better    |

