layout: post
title: "Why does Batch Norm work?"
date: 2023-12-21 hh:mm:ss -0000
categories: optimization

\section{Introduction}
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, they convert the first two moments to mean 0 and variance one

