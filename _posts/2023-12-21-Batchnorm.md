## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, they convert the first two moments to mean 0 and variance one.


## What is batch norm
$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^2 + \varepsilon}} + \beta $$

$$\hat{\mu} = \frac{1}{B} \sum_{i=1}^{B} y_i $$
$$\hat{\sigma}^2 = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{\mu})^2 $$

## What is the internal covariate shift

ICS is closely related to covariate shift, which is referred to as the problem where the input distribution shifts over time. For example, we could use pre-covid's stock data to train a model, however, chances are the model will not be effective in predicting returns in 2020, as the data distribution changed quite substantially.

ICS describes the phenomenon where the data distribution from one layer to the next changes during training.

Recall each layer can be interpreted to be an optimization problem. We are given some input, we are also given some output, we wish to find the best weights that transform the input to the desired output as closely as possible. The ICS occurs when the output of the previous layer (input for current layer) changes drastically at each training step, due to the updates of weight of previous layers.

Let us illustrate this with a simple example
Consider a neural network with 3 neurons with no nonlinearity. Suppose we want to update the weight of layer b. The update rule is:

$$ w_b^{new} \leftarrow w_b^{old} - \alpha \frac{\delta L}{\delta w_b}$$

Taking a look at the gradient term:
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} \frac{\delta z_b}{\delta w_b}$$

Recall
$z_b = w_b*z_a$
hence 
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} z_a$$





<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
