---
layout: post
title: Causal Inference Notes
date: 2024-08-25
---


Consistency:
for each individual, one of the counterfactual outcomes is actually factual (one that corresponds to the treatment value the indiv received)
The counterfactual outcome is equal to his observed outcome and can be expressed as $$Y =Y^A$$ where $$Y^A$$ denotes the counterfactual $$Y^a$$ evaluated at the observed treatment value a.

Problem with I.C.E
ICE is defined as a contrast of the values of counterfactual outcomes, but only one of those outcomes is observed for each individual. All counterfactual outcomes other than the one 
corresponding to the treatment value actually experienced remain unobserved. Hence individual effects cant be identified

We go to 1.2 average causal effects
Average Causal Effect in the population: An ACE of treatment A on outcome Y is present if 
$$ Pr[Y^{a=1} =1] \neq Pr[Y^{a=0} = 1] $$ in the population of interest (this is only for binary outcomes)

the more general one is $$ E[y^{a=1}] \neq E[y^{a=0}] $$

Aside: The absence of ACE does not imply the absence of ICE

1.4 Random variability (havnt read)

1.5 Association vs causation

Association: 2 variables are associated if information about one tells you something about the value of other
Causation: 2 variables are causally related if manipulating the level of one has the potential to change the level of others (holding all else fixed)

causal effect for an individual: The treatment A has a causal effect on an individual's outcome Y if $Y^{a=1} \ineq Y^{a=0}$ for the individual

$ E[Y^{1}_{i}]$ is generally distinct from $ E[Y_i|A_i =1] $
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



Bottom line is, causal inference requires unattainable counterfactual data, but all we can ever expect is real world data. The question then becomes under what conditions can real world data be used for causal inference?

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




