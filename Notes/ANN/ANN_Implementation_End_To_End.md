# ANN Implementation End To End

Using the CHURN MODELLING dataset: The output of the model is to predict if the Customer will leave the Bank - Y/N

Highlights:

1. Create a sequential model (ANN Architecture)
2. We will perform feature engineering: 
   1. Convert categorical values to numerical values (Label Encoder, One Hot Encoder)
   2. Convert the independent features to scaler values
3. Store all the Encoder and Scaler to pickle file
4. Use Drop Out: To avoid over-fitting
5. Convert the model into a file format to deploy (pickle, h5) -> Deploy in Streamlit cloud
6. Use Activation function for the O/P layer
7. Use the Optimizer to perform back-propagation and update the weights (Gradient Descent)
8. Use the Loss function to reduce the loss (MSE - Regression, Cross Entropy Loss - Classification)
9. Performance Metrics
10. Save the training weights to logs and use Tensorboard

