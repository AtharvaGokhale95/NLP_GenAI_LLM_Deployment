
Difference between ANN, CNN and RNN:

1. ANN:
    a. Solves Classification and Regression Problems -> 1 Dimensional I/P data (Numbers, Matrix, Vectors)
    b. Structure: I/P Layer -> Hidden Layers -> O/P Layer
    c. Hidden layer is associated with weights and neurons -> Every neuron in one layer is connected to every neuron in the next
    d. O/P layer is associated with activation function
    e. We use loss function to determine the error and update the weights to reduce the loss using an Optimizer (Gradient Descent) -> Back Propagation

2. CNN:
   1. Solves Image Classification Problems, Object Detection, Face Recognition, Speech -> 3-D or 4-D I/P data (Images, Videos)
   2. In image processing, convolution (mathematical way of combining 2 functions to produce a third function) is used to extract features (like edges, corners, textures) from images by sliding a small matrix (called a filter or kernel) over the image
   3. Structure: I?P layer → [Convolutional Layer → Dimensionality Reduction Layer] → Fully Connected Layer (Hidden Layers) → Output
   4. O/P layer is associated with activation function
   5. We use loss function to determine the error and update the weights to reduce the loss using an Optimizer (Gradient Descent) -> Back Propagation
