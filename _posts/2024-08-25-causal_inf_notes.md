---
layout: post
title: Causal Inference Notes
date: 2024-08-25
---


Consistency:
For each individual, one of their counterfactual outcomes is actually factual (one that corresponds to the treatment value that the individual received).
The counterfactual outcome equals the observed outcome and can be expressed as $$Y^A=Y$$ where $$Y^A$$ denotes the counterfactual $$Y^a$$ evaluated at the observed treatment value a.

Problem with I.C.E
ICE is defined as a contrast of the values of counterfactual outcomes, but only one of those outcomes is ever observed for each individual. All other counterfactual outcomes apart from the one 
corresponding to the actual treatment value remain unobserved. Hence individual effects cant be identified

We go to 1.2 average causal effects
Average Causal Effect in the population: An ACE of treatment A on outcome Y is present if 
$$ Pr[Y^{a=1} =1] \neq Pr[Y^{a=0} = 1] $$ in the population of interest (this is only for binary outcomes)

the more general one is $$ E[y^{a=1}] \neq E[y^{a=0}] $$

Aside: The absence of ACE does not imply the absence of ICE

## 1.4 Random variability 
There are 2 main sources of imperfect irl
1) Sampling variability: In practice, we collect data on a sample of the population of interest. Even if we know the counterfactual outcomes in this sample population, we still do not have the exact risk proportion for the superpopulation. That is: we use (consistent estimator) sample proportion $$\hat{Pr}[Y^{a=0} = 1]$$ to estimate the super-population probability $$ Pr[Y^a =1] $$under treatment value a. We call it a consitent estimator because as number of indiv in sample grows, we expect estimate and true value to grow smaller as the error due to sampling variability obeys the law of large number. Since the super population prob cant be computed but only consistently estimated, we need to have a statistical procedure to evaluate the empirical evidence of the causal null hypothesis.

2) The value of an individual's counterfactual outcomes is not fixed either. In previous examples, Zeus would have a 100% chance of dying if treated and would have 0% chance dying if untreated. The value of the counterfactual outcomes is deterministic for each individual. In real life, counterfactual outcomes are often stochastic.

   

## 1.5 Association vs Causation

Association: 2 variables are associated if information about one tells you something about the value of the other
Causation: 2 variables are causally related if manipulating the level of one has the potential to change the level of others (holding all else fixed)

causal effect for an individual: The treatment A has a causal effect on an individual's outcome Y if $$Y^{a=1} \neq Y^{a=0}$$ for the individual

$$ E[Y^{1}_{i}]$$ is generally distinct from $$ E[Y_i|A_i =1] $$
former is defined as a prior and averaged across the entire population (only god knows)
Latter is defined after treatment assignment,  and the average among those who receive treatment A_i = 1

The conditional probability $$ Pr[Y=1 | A=1] $$ is defined as the proportion of individuals that developed the outcome Y among those individuals in the population of interest that happened to receive treatment value a. 

We say treatment A and outcome Y are independent if $$ Pr[Y=1 | A=1] = Pr[Y=1 | A=0] $$, that A is not associated with Y, or that A does not predict Y. Conversely we say that treatment A and outcome Y are dependent or associated

![Screenshot 2024-08-27 at 9 39 23 PM](https://github.com/user-attachments/assets/8700f062-3563-48e1-96c2-81e6c50fc510)

The definition of causation implies a contrast between all individuals treated vs all individuals untreated. Association implies a contrast between the treated and the untreated of the original diamond.

Put another way, inference about causation is concerned with the counterfactual worlds, like "what would be the risk if everybody been treated/untreated?" Inference for association is concerned with questions in the actual world, like "what is the risk in the treated/untreated?"

Association uses a conditional probability, namely being conditioned on having actually received treatment value a (A=a)
causation is unconditional , or marginal probability, the risk of $$Y^a$$ in the entire population.

Therefore, association is defined by a different risk in two disjoint subsets of the population determined
by the individuals’ actual treatment value (A = 1 or A = 0), whereas causation is defined by a different risk in the same population under two different treatment values (a = 1 or a = 0). 



The bottom line is, that causal inference requires unattainable counterfactual data, but all we can ever expect is real world data. The question then becomes under what conditions can real-world data be used for causal inference?


# Section 2: Randomization
Again irl, we don't know both potential outcomes, we only know the observed outcome Y under the treatment value A that the individual happened to receive. Only one of the 2 counterfactual outcomes is known for each individual: the one corresponding to the treatment level that he actually received. The data are missing for the other counterfactual outcomes.

Randomized experiments, still generate data with missing values of the counterfactual outcomes, however, randomization ensures that those missing values occur only by chance and hence causal questions could be consistently estimated.

Suppose we assign an individual to the white group if it is a tail, and to the grey group if it's a head. Then we ask our research assistant to administer the treatment, to give the white group the treatment and the grey group a placebo; we then proceed to collect the data. 


It wasn't until the next day that you found out they had misinterpreted your instructions and gave treatment to the grey group instead. This reversal of treatment DOES NOT affect our conclusions! 


When group membership is randomized, which particular group received the treatment is irrelevant for the value of 
$$ Pr[Y=1 |A=1]$$ or $$Pr[Y=1 |A=0]$$ 
Formally we say the groups are exchangeable.


Exchangeability: 
$$ Pr[Y^a = 1  | A=1] = Pr[Y^a = 1  | A=0] = Pr[Y^a =1]$$ 
We say  the counterfactual risk under treatment value a (both when a=1 and a =0) is the same in both groups A = 1 and A = 0. Phrasing differently, the actual treatment A does not predict the counterfactual outcome, and the actual treatment are independent for all values a. $$ Y^a \perp A$$

In the presence of exchangeability, the observed risk in treated within the white group would equal the counterfactual risk under treatment in the entire population!

Another attempt to explain exchangeability $$ Y^a \perp A$$. The counterfactual outcome $$Y^a$$ is like your genetics, it encodes deterministically how you will react if you took the treatment, before treatment A was randomly assigned. Another point is that we only learn about the value of the counteractual/genetic makeup $$Y^a$$ only after the treatment A is given and only if A=a.

Difference between $$ Y^a \perp A $$ and $$ Y \perp A$$
In a randomized experiment in which exchangeability holds and we find the treatment has a causal effect on the outcome, then 
$$ Y \perp A$$ does not hold.

## section 2.2 Conditional randomization 
In the example in Hernan's book, if you are in critical condition, you will be more likely to be assigned treatment than not. We no longer have marginal exchangeability. However we can conclude the observed risks equal the counterfactual risks in the group of all critical conditions.
1) $$ Pr[Y^{a=1} = 1 |L =1] = Pr[Y=1 |L=1,A=1] $$ 
2) $$ Pr[Y^{a=0} = 1 |L =1] = Pr[Y=1 |L=1,A=0] $$

3) $$ Pr[Y^{a=1} = 1 |L =0] = Pr[Y=1 |L=0,A=1] $$ 
4) $$ Pr[Y^{a=0} = 1 |L =0] = Pr[Y=1 |L=0,A=0] $$

conditionally randomized experiment is simply a combination of 2 separate marginally randomized experiments: one conducted in the subset of individuals in critical condition, the other  in the subset of individuals in critical condition. Within each subset, the treated and the untreated are exchangeable.

Conditional exchangeability: $$Y^a \perp A|L$$ for all a.
Or Equivanetly: 
$$ Pr[Y^a = 1 |A=1,L=1] = Pr[Y^a = 1 |A=0,L=1]$$

conditional randomization does not guarantee unconditional (marginal) exchangeability, it does guarantee conditional exchangeability within levels of the variable L.

## 2.3 Identification under conditional randomization: Standardization
The question is can we write the counterfactual in terms of observed data? (whether it's identifiable)

By cond exch:

1) $$ Pr[Y^{a=1} = 1 |L =1] = Pr[Y=1 |L=1,A=1] $$ 
2) $$ Pr[Y^{a=0} = 1 |L =1] = Pr[Y=1 |L=1,A=0] $$
3) $$ Pr[Y^{a=1} = 1 |L =0] = Pr[Y=1 |L=0,A=1] $$ 
4) $$ Pr[Y^{a=0} = 1 |L =0] = Pr[Y=1 |L=0,A=0] $$

Now suppose we want to calculate the causal risk ratio
$$ \frac{Pr[Y^{a=1} = 1]} {Pr[Y^{a=0} = 1]} $$

The numerator is the risk if all individuals in the population had been treated, we can find this with the weighted average of risk of each group, where weight is proportional to its size L.

Recall in conditional exchangeability
$$
\begin{aligned}
E[Y^1] &= \mathbb{E}_L(\mathbb{E}_{Y|L} [Y^1|L]) \quad (\text{by total expectation}) \\
       &= \sum_{\ell} E[Y^1 \mid L=\ell] \cdot P(L=\ell) \quad  \\
       &= \sum_{\ell} E[Y^1 \mid A=1, L=\ell] \cdot P(L=\ell) \quad (\text{by conditional exchangeability}) \\
       &= \sum_{\ell} E[Y \mid A=1, L=\ell] \cdot P(L=\ell) \quad (\text{by consistency}) \\
\end{aligned}
$$

## 2.4 Identification under conditional randomization: Inverse Probability Weighting

$$ E[Y^a] = E[\frac{I(A=a)}{f(A|L)} Y] $$

We create 2 simulations (2 trees) of what would have happened had all individual in pop been treated , and untreated respectively. These simulations are correct under cond exchange. Then we POOL the 2 trees, to create a hypothestical population of size 2n in which every individual appears as treated AND untreated. This 2n population is called the pseudo population.

Given C.E in origianl population, the treated and untreated are MARGINALLY exchangeable in the pseudo pop because L is independent of A. That is, associational risk ratio in pseudo population equals the causal risk ratio in BOTH the pseudo pop and org pop.


Both standardization and IPW can be viewed as procedures to build a new tree in which all individuals receive treatment a.
They differ by suing different set of the probabilities to build the counterfactual tree; IPW uses conditional prob of treatment A given covariate L, while standarization uses prob of covariate L and the conditional probability of outcome Y given A and L

Both simulate what would have been observed if L had not been used to decide the probability of treatment

## Power Analysis:
definition: Power (1- $$\beta$$) = The probability of rejecting the Null hypothesis when $$H_A$$ is in fact true.
The higher the power, the less likely you will make a false negative error. (type ii error, incorrectly failing to reject null)

definition: Alpha = The probability of rejecting the Null hypothesis when &&H_o$$ is in fact true. (type 1 error, incorrectly rejecting null)

Relationship between $$ /alpha$$, $$/beta$$ and sample size.
Sample size calculation:
For a fixed level of alpha, what sample size would be needed to guarantee a power of P?

Power calculation:
For a given sample size and clinically important treatment effect, what is the power of the study?

Example
Suppose we want to test $$H_0 : \pi_1 = \pi_0 $$ against $$H_A: \pi_1 \neq \pi_0$$ using risk difference. The score test for risk difference would be 

$$ Z = \frac{\hat{\pi_1} - \hat{\pi_0} }{\sqrt{\hat{\pi}(1-\hat{\pi})( \frac{1}{n} + \frac{1}{n})}}  ~ N(0,1)$$

Under a one-sided test, we reject the Null hypothesis when $$ Z < z^* $$ and we want an alpha of $$\alpha$$ means

$$ P_0(Z-z^*) = \alpha $$


To obtain a desired power of 1- $$\beta$$ we want to reject Null 1- $$\beta$$% of the time given Non null is indeed True. That is we want

1- $$\beta$$ $$ \leq P_A (Z<z)$$




## Chapter 11:
What is the difference between the nonparametric estimator in Part I vs the parametric (model-based) estimators in Part II?

We want to consistently estimate the mean of Y among individuals with treatment level A=a in the population, from which we have data on a subset of the population. That is we use $$ \hat{E}[Y|A=a]$$ to estimate $$ \hat{E}[Y|A=a]$$. 

But its possible that A could take on a near continuous range of values, in which case we can't allocate our finite sample across the continuum. In fact if there is no data in a category, the sample average is undefined for that category. We have to make an additional assumption / constraint on the form of data. Let's go over the thought experiment, suppose we have 16 individuals in our sample and they could take on A = {0,1}, or A = {0,1,2,3} . Clearly the number of individuals per category decreases as the number of categories increase. The sample average in each categoy is still an unbiased estimator of the corresponding population mean. But... the probability that the sample average is close to the corresponding population mean decreeases as the number of individuals in each category decreases. That is the distribution of sample averages has a much larger variance.

We need to add restrictions. For example: the outcome of A=1 must be in between that of A=0 and A=2 and linearly proportional.

$$ E[Y|A] = \theta_0 + \theta_1A$$
The above equation is known as the functional form of the conditional mean. This model specifies that all conditional mean functions are straight lines, though their intercepts and slopes may vary.

An exactly unbiased estimator of the parameters can be obtained by ordinary least squares.

This is not a free lunch though, when using a parametric model, we are assuming there is no model misspecification, i.e. the functional form has to be somewhat close to the unknown reality.

### saturated models 11.3
if there are 2 parameters and only 2 treatment levels, we call it  a saturated model, which essentially means the model doesn't impose restrictions on the distribution of the data. generally model is saturated whenever # of parameters in a conditional mean model equals the number of unknown conditional means in the population





<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




