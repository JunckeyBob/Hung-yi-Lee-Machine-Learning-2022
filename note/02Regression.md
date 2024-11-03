# Regression
## Linear Regression
### Linear Regression Model
 - Step1: fuction with unknown
  $$y = w\cdot x + b$$
   - `y` output, $\hat{y}$ means the label
   - `w` weight
   - `b` bias
   - `x` input or object, $x^i$ means a component of object, $x_i$ means an index of object
 - Step2: define loss
  $$L(w, b) = \sum_{n=1}^{N}(\hat{y}-y)^2$$
 - Step3: optimization

### Gradient Descent
 - Step1:(Randomly) Pick an initial value $w^0$
 - Step2: Compute $\frac{\partial L}{\partial w}|_{w = w^0}$
 - Step3: Update `w` iteratively
  $$
  w^1 = w^0 - \eta \frac{\partial L}{\partial w}|_{w = w^0} \\
  \eta: learning\ rate(hyperparameter)
  $$

### Better Learning Rate
 - Adaptive Learning Rate
  As the parameters are updated iteratively, the  learning rate should become smaller and smaller
  $$
  \eta^t = \frac{\eta}{\sqrt{t+1}} \\
  t: number\ of\ iterations
  $$
 - Adaptive Gradient Descent(Adagrad)
  Each parameter has a different learning rate
  $$
  w^{t+1} = w^t - \frac{\eta}{\sqrt{\sum_{i=0}^t(g^i)^2}}g^t \\
  g^i = \frac{\partial L}{\partial w}|_{w = w^i}
  $$

## Sigmoid Function

## How to select model
An art of making trade-off between variance and bias