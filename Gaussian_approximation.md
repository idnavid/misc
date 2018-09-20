## On the significance of Gaussian distributions
The question is: "Why is the Gaussian distribution so popular in probabilistic estimation?". Although there are MANY reasons that justify using a Gaussian distribution, I find the following, which is the Maximum Lieklihood Taylor Series Expansion most resonating for me as an engineer.     
For more on this, see David Mackay's book "[Information Theory, Inference, and Learning Algorithms - Chapter 28](http://www.inference.org.uk/itila/)" 

Say we want to approximate the posterior probability of a parameter, **w**, given some data, **D**. A reasonable approximation is the Taylor Series expansion around some point of interest. A good candidate for this point is the Maximum Likelihood estimation, **w***. Using the 2nd order Taylor Series expansion of the *log*-probability of P at **w***: 


**log(P(**w**|**D**)) = log(P(**w***|**D**)) + &#8711;log(P(**w***|**D**))(**w**-**w***) - (1/2)(**w**-**w***)^T(-&#8711;&#8711;log(P(**w***|**D**)))(**w**-**w***) +H.O.T




Since the ML is a maxima, &#8711;log(P(**w***|**D**))=0. Defining **&#915;**=(-&#8711;&#8711;log(P(**w***|**D**))), we have:

**log(P(**w**|**D**)) &#8776;	 log(P(**w***|**D**)) - (1/2)(**w**-**w***)^T **&#915;**(***w***-**w***). 

Take the exponent of the additive terms: 

**P(**w**|**D**) &#8776; cte exp(- (1/2)(**w**-**w***)^T **&#915;**(**w**-**w***))

where cte=P(**w***|**D**). So, 

> *The Gaussian N(**w***,**&#x393;**^(-1)) is the second order Taylor Series approximation of any given distribution at its Maximum Likelihood.*

where **w*** is the Maximum Likelihood of the distribution and **&#x393;** is the Hessian of its log-probability at **w***.  


This is closesly related to Laplace approximation for approximating integrations. 

