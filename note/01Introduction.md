# Introduction of Machine/Deep Learning
## What is machine learning?
### Machine learning is to make the machine have the ability to find a function.

## Different types of Functions
 - **`Regression`**: The function outputs a scalar
 - **`Classification`**: Given options(classes), the function outputs the correct one
   - **Binary Classification**: Output 0 or 1, Yes or No
   - **Multi-Category Classification**: Output [1, 2, 3, ..., N]
 - **`Structured Learning`**: Create something with structure(image, document)

## How to find the function
 1. Write down a funciton(model) with unknown parameters(feature) based on domain knowledge
 2. Define loss(how good features are) from training data
 3. Solve an optimization problem which minimizes the value of the loss (Gradient Descent)

## How to teach machine learning
 - Supervised Learning: With labels to caculate loss
 - Self-supervised Learning: Without labels, one model value the loss by itself
   - Pre-trained model(Foundation Model)
   - Downstream Tasks
 - Generative Adversarial Network: Without labels, two models to value the loss of each other
 - Reinforcement Learning: Without labels, use dynamic environment to learn

## Cutting-edge research
 - Anomaly Detection
 - Explainable AI
 - Model Attack
 - Domain Adaptation
 - Network Compression
 - Life-long Learning
 - Meta Learning: Learn how to learn