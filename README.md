# Gender and Age Prediction
This is a light and efficient model for gender and age prediction. It takes a face crop as input and predicts whether it's a male or female, as well as their age (less than 19, 20-29, 30-39, 40-49, more than 50). When trained and optimized, it has an average 93% accuracy on gender and 63% on age (cf. results below).

The model as it is now will not give the best results. Indeed, one has to first pretrain the model on the `VGGFace` dataset, then trained it on the `FairFace` dataset (the one with the highest padding). This will significantly improve the accuracy.

To further increase the accuracy, one can also preprocess the two image datasets: the best practice would be to run an efficient face detection on both `VGGFace` and `FairFace` and to align the resulting crops.

# Features
  - Dual task: gender and age classification.
  - Architecture: `(C-MP-BN)^4-F-DO-FC-BN-(O1/O2)`.
  - Convolutions: `64/128/256/512`. Filters: `3x3`. Dropout rate: `0.4`. Fully-connected size: `1024`.
  - Number of parameters: `3,761,638`.
  - Input image size: `64 x 64`.
  - Losses:
    - Gender: binary cross-entropy.
    - Age: scaled MAE.
  - Optimizer: Adam. Learning rate: `1E-05`.
  - Number of epochs: `300`.
  - Data augmentation: horizontal flip + blur.
  - Trained on the `FairFace` dataset.
  - Results on `RTX 3090`:
```
=======================================================================
Epoch 300 | train loss: 0.62440 | val loss: 0.83710 | time: 141.68 sec.
Gender    | train acc : 0.97919 | val acc : 0.93296.
Age       | train acc : 0.78255 | val acc : 0.62619.
=======================================================================
Average time per epoch: 141.74 sec.
``` 
