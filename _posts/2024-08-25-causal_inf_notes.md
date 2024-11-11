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
1) Sampling variability: In practice, we collect data on a sample of the population of interest. Even if we know the counterfactual outcomes in this sample population, we still do not have the exact risk proportion for the superpopulation. That is: we use (consistent estimator) sample proportion $$\hat{Pr}[Y^{a=0} = 1]$$ to estimate the super-population probability $$ Pr[Y^a =1] $$under treatment value a. We call it a consistent estimator because as number of indiv in sample grows, we expect estimate and true value to grow smaller as the error due to sampling variability obeys the law of large number. Since the super population prob cant be computed but only consistently estimated, we need to have a statistical procedure to evaluate the empirical evidence of the causal null hypothesis.

2) The value of an individual's counterfactual outcomes is not fixed either. In previous examples, Zeus would have a 100% chance of dying if treated and would have 0% chance dying if untreated. The value of the counterfactual outcomes is deterministic for each individual. In real life, counterfactual outcomes are often stochastic.

   

## 1.5 Association vs Causation

Association: 2 variables are associated if information about one tells you something about the value of the other
Causation: 2 variables are causally related if manipulating the level of one has the potential to change the level of others (holding all else fixed)

causal effect for an individual: The treatment A has a causal effect on an individual's outcome Y if $$Y^{a=1} \neq Y^{a=0}$$ for the individual

$$ E[Y^{1}_{i}]$$ is generally distinct from $$ E[Y_i|A_i =1] $$
former is defined as a prior and averaged across the entire population (only god knows)
Latter is defined after treatment assignment,  and the average among those who receive treatment $$A_i$$ = 1

The conditional probability 
$$ Pr[Y=1 | A=1] $$
is defined as the proportion of individuals that developed the outcome Y among those individuals in the population of interest that happened to receive treatment value a. 

We say treatment A and outcome Y are independent if 
$$ Pr[Y=1 | A=1] = Pr[Y=1 | A=0] $$,
that A is not associated with Y, or that A does not predict Y. Conversely we say that treatment A and outcome Y are dependent or associated

![Screenshot 2024-08-27 at 9 39 23 PM](https://github.com/user-attachments/assets/8700f062-3563-48e1-96c2-81e6c50fc510)

The definition of causation implies a contrast between all individuals treated vs all individuals untreated. Association implies a contrast between the treated and the untreated of the original diamond.

Put another way, inference about causation is concerned with the counterfactual worlds, like "What would be the risk if everybody been treated/untreated?" Inference for association is concerned with questions in the actual world, like "What is the risk in the treated/untreated?"

Association uses a conditional probability, namely being conditioned on having actually received treatment value a (A=a)
causation is unconditional, or marginal probability, the risk of $$Y^a$$ in the entire population.

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

Another attempt to explain exchangeability $$ Y^a \perp A$$. The counterfactual outcome $$Y^a$$ is like your genetics, it encodes deterministically how you will react if you took the treatment, before treatment A was randomly assigned. Another point is that we only learn about the value of the counterfactual/genetic makeup $$Y^a$$ only after the treatment A is given and only if A=a.

Difference between $$ Y^a \perp A $$ and $$ Y \perp A$$
In a randomized experiment in which exchangeability holds and we find the treatment has a causal effect on the outcome, then 
$$ Y \perp A$$ does not hold.

## Section 2.2 Conditional randomization 
In the example in Hernan's book, if you are in critical condition, you will be more likely to be assigned treatment than not. That is, we no longer have marginal exchangeability. The only thing conclusive is the observed risks equal the counterfactual risks WITHIN the group of all critical conditions.

1a. $$ Pr[Y^{a=1} = 1 \mid L =1] = Pr[Y=1 \mid L=1, A=1] $$

1b. $$ Pr[Y^{a=0} = 1 \mid L =1] = Pr[Y=1 \mid L=1, A=0] $$

2a. $$ Pr[Y^{a=1} = 1 \mid L =0] = Pr[Y=1 \mid L=0, A=1] $$

2b. $$ Pr[Y^{a=0} = 1 \mid L =0] = Pr[Y=1 \mid L=0, A=0] $$


We see that a conditionally randomized experiment is simply a combination of 2 separate marginally randomized experiments: one conducted in the subset of individuals in critical condition, the other in the subset of individuals in critical condition. Within each subset, the treated and the untreated are exchangeable.

Definition of Conditional exchangeability: $$Y^a \perp A|L$$ for all a.

Or Equivalently: 
$$ Pr[Y^a = 1 |A=1,L=1] = Pr[Y^a = 1 |A=0,L=1]$$

conditional randomization does not guarantee unconditional (marginal) exchangeability, it does guarantee conditional exchangeability within levels of the variable L.

## 2.3 Identification under conditional randomization: Standardization
The question is under conditional randomization, can we write the counterfactual in terms of observed data? (whether it's identifiable)

Suppose we want to calculate the causal risk ratio
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

We create 2 simulations (2 trees) of what would have happened had all individuals in pop been treated, and untreated respectively. These simulations are correct under cond exchange. Then we POOL the 2 trees, to create a hypothetical population of size 2n in which every individual appears as treated AND untreated. This 2n population is called the pseudo population.

Given C.E in original population, the treated and untreated are MARGINALLY exchangeable in the pseudo pop because L is independent of A. That is, the associational risk ratio in the pseudo population equals the causal risk ratio in BOTH the pseudo pop and org pop.


Both standardization and IPW can be viewed as procedures to build a new tree in which all individuals receive treatment a.
They differ by using different set of probabilities to build the counterfactual tree; IPW uses the conditional prob of treatment A given covariate L, while standarization uses prob of covariate L and the conditional probability of outcome Y given A and L

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


## Chapter 4:
**def** effect modifier: We say V is a modifier of the effect of A on Y when the average causal effect of A on Y varies across levels of V.

Additive effect modification:
$$ E[Y^{a=1} -Y^{a=0} | V= 1] \neq E[Y^{a=1} -Y^{a=0} | V= 0] $$
Note the presence of effect modification depends on the effect measure being used.
We also only consider variables V that are NOT affected by treatment A as effect modifiers.

**def** qualitative effect modification: when the average causal effect in the subsets are in the opposite direction.

properties: with QEM, additive effect modification IFF multiplicative effect modification.
in the absence of QEM, you can find effect modification on one scale but not on the other.

how EM could exist in one scale but not other

We do not need conditional exchangeability to identify effect modification.


one thing to note is even if we found V (nationality) modifies effect of heart transplant A on risk of death Y, we DO NOT know the causal mechanism involved in the effect modification. Thus the term effect modification by V doesnt necessarily imply V plays a causal role in the modification of the effect. In our example, it is possible that nationality is simply a marker for the causal factor that is truly responsible for the modification of effect.

This contrasts with interaction, which DOES attribute a causal role to the variables involved.

## 4.3 Why should we care about effect modification:
1) If we found V is indeed an effect modifier from our study, we need to know the make up of V in the population for the findings of the study to be useful. For example, the average causal effect in the population was harmful in women but beneficial in men (and the proportion of women and men were the same such that they cancel out). Our conclusion would be different had we conducted our study where theres more women. That is the average causal effect in the population depends on the distribution of other effect modifiers in the population. In reality, our discussion thus far is really: the average causal effect of treatment A on outcome Y in a population **with a particular mix** of causal effect modifiers.
2) The extrapolation of causal effects computed in one population to a second population is referred to as transportability of causal inferences across population. Again the finding from one population might not translate to another population with a different distribution of effect modifiers.

In fact we are making a huge assumption that there all effect modifiers (including those unknown or unmeasured) are the same between the 2 populations. these unmeasured effect modifiers are not variables needed to achieve exchangeability, but just risk factors for the outcome. in general the transportabiltiy of effects across population is a more difficult problem than the identification of causal effects in a single population. It is an unverifiable assumption.

fine points:
- additive, but not multiplicative, effect modification is the appropriate scale to identify the groups that will benefit most from intervention. in the absence of additive effect modification, learning that there is multiplicative effect modification may not be very helpful for decision making.


## 4.4 Stratification as a form of adjustment

Stratified analysis is the natural way to identify effect modification. To see if V is a modifier, we compute the average causal effect of A on Y, for each stratum of V. Again, we can do this in both marginally exchangible studies or conditionally exchangible study. 
a) In marginally exchangible you 1) stratify on V 2) calculate sample proportion
b) In conditionally exchangible you 1) stratify on V 2) use Standarization or IPW on L. That is we need to stratify on V to identify effect modification in addition to adjusting for L. But in practice stratification is often used to adjust for L as well, but they only achieve conditional effect measure, instead of IPW which measures marginal effect measure (stratification necessarily results in multiple stratum specific effect measures

## 4.5 Matching as another form of adjustment
goal of matching is to construct a subset of the population which the variables L have the same distribution in both the treated and untreated.

## 4.6 Effect modification and adjustment methods
Standardization, IPW, stratification and matching are all ways to estimate the average causal effect, but they estimate different types of causal effects. The first 2 are capable of measuring both marginal and conditional effects, while the latter 2 can only measure conditional effects within a certain subset of the population. 



## Notes on collapsibility
background: OR is designed for homogeneous patient population, and not when there is substantial outcome heterogeneity even for patients receiving the same treatment. But if a strong risk factor exist (patient comes from a mixture of distribution). It is a good idea to pre-specify important covariates for the primary analysis, otherwise there might be loss in power

Non-collapsibility means the conditional ratio is different from the marginal (unadjusted) ratio even in the complete absence of confounding. It might also mean the marginal ratio is not a weighted average of the conditional ratio.
The marginal ratio is difficult to interpret and does not generalize to other populations with a different covariate distribution than our sample.
in other words, the marginal OR depends on the distribution of the covariate in the sample and doesn't transport to population with a different cov dist.

there is a change in estimate approach to identify if V is a confounder, but this is flawed if the effect measure is non-collapsible


## 5.1 Interaction
There is interaction between 2 treatments A and E if the causal effect of A on Y differs if we SET E = 0 vs E = 1.

When the causal effect is measured on the risk difference scale, the definition of interaction is:
$$ Pr[Y^{a=1,e=1}=1] - Pr[Y^{a=0,e=1}=1] \neq  Pr[Y^{a=1,e=0}=1] - Pr[Y^{a=0,e=0}=1] $$

We can simply rearrange above to show that this inequality the causal risk diff for E when everyone receives a transplant is also less than the causal risk difference for E when nobody receives a transplant. That is we can equivalently define interaction between A and E on the additive scale. and the 2 inequalities show that treatments A and E have equal status in the definition of interaction.

$$ Pr[Y^{a=1,e=1}=1] - Pr[Y^{a=1,e=0}=1]  \neq  Pr[Y^{a=0,e=1}=1] - Pr[Y^{a=0,e=0}=1] $$

**Effect modification vs Interaction**
In effect modification, we do not consider V and A as variables of equal status, because we can only hypothetically intervene on the variable of A, not V. That is the definition of effect modification involves the counterfactual outcomes $$Y^a$$, not the counterfactual outcome $$Y^{a,v}$$


## 5.2 identifying interaction
Interaction is concerned with the join effect of 2 (or more) treatments, to identify we need exchangeability, positivity and consistency for both (or more) treatments.

Case when E is randomly and unconditionally assigned by investigators. Then the treated E = 1 and untreated E = 0 are expected to be exchangeable.
That is:
$$ Pr[Y^{a=1,e=1}=1] = Pr[Y^{a=1} =1 | E=1]$$ As a result we can rewrite the definition of interaction between A and E with something we can observe.

interaction in this case is 
$$ Pr[Y^{a=1}=1 | E = 1] - Pr[Y^{a=0}=1 | E=1] \neq Pr[Y^{a=1}=1 | E = 0] - Pr[Y^{a=0}=1 | E=0]$$
which is the **exactly** the same as the definition of effect modification! When treatment E is randomly assigned, then the concepts of interaction and effect modification coincide.

Case when E is **not** assigned by investigators. To assess the presence of interaction between A and E, we still need to compute the same 4 marginal risks. Without marginal randomization, we can use standardization or IPW. We can view A and E as a combined treatment with 4 levels, instead of A and E as separate treatments with 2 levels each.

When there is conditional exchangeability for treatment A but not for treatment E. Then we can't generally assess the presence of interaction between A and E, but can still assess the presence of effect modification by E. This is because we use notation V (or E here) for variable for which we are not willing to make assumptions about exchangeability, positivity, and consistency. i.e we concluded that effect of transplant A was modified by nationality V, but we never required any identifying assumptions for the effect of V since we are not interested in using our data to compute the causal effect of V on Y.

In section 4.2 we found V is surrogate effect modifier; that is V does not act on the outcome and therefore does not interact with A (no action, no interaction), but V is a modifier of the effect of A on Y because V is correlated with an unidentified variable that actually has an effect on Y and interacts with A.









## Chapter 11:
What is the difference between the nonparametric estimator in Part I vs the parametric (model-based) estimators in Part II?

We want to consistently estimate the mean of Y among individuals with treatment level A=a in the population, from which we have data on a subset of the population. That is we use 
$$ \hat{E}[Y|A=a]$$ to estimate $$ \hat{E}[Y|A=a]$$. 

But it's possible that A could take on a nearly continuous range of values, in which case we can't allocate our finite sample across the continuum. In fact, if there is no data in a category, the sample average is undefined for that category. We have to make an additional assumption/constraint on the form of data. Let's go over the thought experiment, suppose we have 16 individuals in our sample and they could take on A = {0,1}, or A = {0,1,2,3}. Clearly the number of individuals per category decreases as the number of categories increases. The sample average in each category is still an unbiased estimator of the corresponding population mean. But... the probability that the sample average is close to the corresponding population mean decreases as the number of individuals in each category decreases. That is the distribution of sample averages has a much larger variance.

We need to add restrictions. For example: the outcome of A=1 must be in between that of A=0 and A=2 and linearly proportional.

$$ E[Y|A] = \theta_0 + \theta_1A$$
The above equation is known as the functional form of the conditional mean. This model specifies that all conditional mean functions are straight lines, though their intercepts and slopes may vary.

An exactly unbiased estimator of the parameters can be obtained by ordinary least squares.

This is not a free lunch though, when using a parametric model, we are assuming there is no model misspecification, i.e. the functional form has to be somewhat close to the unknown reality.

### Saturated models 11.3
if there are 2 parameters and only 2 treatment levels, we call it  a saturated model, which essentially means the model doesn't impose restrictions on the distribution of the data. generally model is saturated whenever # of parameters in a conditional mean model equals the number of unknown conditional means in the population



### Propensity score



Let L be a vector of baseline covariates and A be the treatment indicator
A balancing score b(L) is any function of the covariates L s.t. $$ A \perp L \mid b(L) $$

**Def Balancing score:** b(L) is a balancing score IFF b(L) is finer than ps(L) in the sense that ps(L) = fb(L) for some function f.

This implies that the balancing function b contains more information than the propensity score. 

Claim: If b(L) is a balancing score (i.e. $$ A \perp L \mid b(L) $$) ,  $$ \exists f s.t. ps(L)  =fb(L) $$

#### Proof (Approach 1)

$$
\begin{aligned}
ps(L) &= P(A = 1 \mid L) \\
&= \sum_{\ell} P(A = 1 \mid L, b(L)) P(b(L) \mid L) \\
&= \sum_{\ell} P(A = 1 \mid b(L)) \quad \text{(since \( b(L) \) is a balancing score, \( P(b(L) \mid L) = 1 \))}
\end{aligned}
$$


Aside: $$ P(X \mid Y)$$ is a random variable with randomness inherited from Y and not X. Hence the expression at the last step is a function of the balancing score.

#### Proof (Approach 2)
Suppose \( b(L) \) is a balancing score, but assume for contradiction that \( b(L) \) is not finer than \( ps(L) \). This implies:

$$ \exists l_1, l_2 \quad \text{such that} \quad b(l_1) = b(l_2) \quad \text{but} \quad ps(l_1) \neq ps(l_2) $$

Since \( b(l_1) = b(l_2) \), we have:

$$ P(A \mid b(l_1)) = P(A \mid b(l_2)) $$

However, since \( ps(l_1) \neq ps(l_2) \), it follows that:

$$ P(A \mid l_1) \neq P(A \mid l_2) $$

Thus, we also have:

$$ P(A \mid l_1, b(l_1)) = P(A \mid l_2, b(l_2)) $$

This leads to a contradiction, because:

$$ P(A \mid L, b(L)) \neq P(A \mid b(L)) $$

This contradicts the assumption that \( b(L) \) is a balancing score.





#### Converse:
Now if $$ \exists f s.t. ps(L)  = fb(L)$$, we want to show b(L) is a balancing score, i.e. $$ A \perp L \mid b(L) $$

$$ P(A =1 \mid b(L),L) =  P(A =1 \mid b(L)) $$

We start from RHS

$$
\begin{aligned}
P(A = 1 \mid b(L), L) &= \mathbb{E}[A \mid b(L)] \\
&= \mathbb{E}[ \mathbb{E}[A \mid b(L), L] \mid b(L)] \\
&= \mathbb{E}[ \mathbb{E}[A \mid L] \mid b(L)] \\
&= \mathbb{E}[ P(A = 1 \mid L) \mid b(L)] \\
&= \mathbb{E}[ f b(L) \mid b(L)] \\
&= f b(L) \\
&= ps(L) \\
&= P(A = 1 \mid L) \\
&= P(A = 1 \mid L, b(L))
\end{aligned}
$$



The result we have shown implies the propensity score is the coarsest balancing score, the finest balancing score would be the identity.


Next, we will prove that if the treatment assignment is strongly ignorable given L, then it is strongly ignorable given any balancing score b(L). Where strongly ignorable means $$ P(Y^{A} \mid A,L) $$

if treatment assignment is strongly ignorable given X, then the difference between treatment and control means at each value of
a balancing score is an unbiased estimate of the treatment effect at that value  


We want to show $$ (Y^0,Y^1) \perp A \mid b(L) $$
it is sufficient to show that 

$$ P(A=1 \mid Y^0,Y^1,b(l)) = P(A=1 \mid b(l)) $$
which by theorem 2 is equivalent to showing that
$$ P(A=1 \mid Y^0,Y^1,b(l)) = ps(l) $$


$$
\begin{aligned}
P(A = 1 \mid Y^0, Y^1, b(l)) &= \mathbb{E}[A \mid (Y^0, Y^1), b(l)] \\
&= \mathbb{E}[\mathbb{E}[A \mid (Y^0, Y^1), l] \mid (Y^0, Y^1), b(l)] \\
&= \mathbb{E}[\mathbb{E}[A \mid l] \mid (Y^0, Y^1), b(l)] \\
&= \mathbb{E}[\text{ps}(l) \mid (Y^0, Y^1), b(l)] \\
&= \mathbb{E}[\text{FB}(l) \mid (Y^0, Y^1), b(l)] \\
&= f b(l) \\
&= \text{ps}(l)
\end{aligned}
$$




Theorem 4) Suppose the treatment assignment is strongly ignorable and b(l) is a balancing score. Then the expected difference in observed responses to the 2 treatments at b(l) is equal to the average treatment effect at b(l)















<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




