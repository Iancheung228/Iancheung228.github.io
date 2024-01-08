layout: post
title: "Why does Batch Norm work?"
date: 2023-12-21 hh:mm:ss -0000
categories: optimization

## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, they convert the first two moments to mean 0 and variance one.


## what is batch norm
$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^2 + \varepsilon}} + \beta $$

$$\hat{\mu} = \frac{1}{B} \sum_{i=1}^{B} y_i $$
$$\hat{\sigma}^2 = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{\mu})^2 $$
