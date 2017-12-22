
## Coordinate Ascent Update for Mean-Field Variational Bayes
Variational Bayes (Variational Inference) is a functional optimization procedure designed to find the best approximating distribution, $q({\bf Z})$, that results in minimum Kullback-Leibler divergence between itself and the posterior distribution of the latent variables given some data, $p({\bf Z}\big\rvert {\bf X})$. ${\bf Z}$ and ${\bf X}$ respectively represent the collection of latent and observation variables.

### Problem Statement
We follow the interpretation presented in [1]. Let ${\bf X}$ be a set of observations (random variables), ${\bf Z}$ be some unknown random latent variables. From Bayes's theorem: 
$$p({\bf X}) = \frac{p({\bf X,\bf Z})}{p({\bf Z\big\rvert \bf X})}\\
$$
$$\hspace{4cm} \log(p({\bf X})) = \log{p({\bf X, \bf Z})} - \log{p({\bf Z\big\rvert \bf X})}\\
$$
Taking the expectation with respect to some distribution, $q({\bf Z})$, gives:

$$
\log(p({\bf X})) = E_q[\log{p({\bf X,\bf Z})}] - E_q[\log{p({\bf Z\big\rvert \bf X})}],
$$

where $E_q[f({\bf X,\bf Z})] = \int q({\bf Z})f({\bf X ,\bf Z})d{\bf Z}$ and we have used $\log(p({\bf X})) = E_q[\log(p({\bf X}))]$, since $p({\bf X})$ does not depend on ${\bf Z}$. After adding $E_q[\log(q({\bf Z}))]$ to and subtracting it from the r.h.s: 

$$
\log(p({\bf X})) = E_q[\log{\frac{p({\bf X, \bf Z})}{q({\bf Z})}}] - E_q[\log{\frac{p({\bf Z\big\rvert \bf X})}{q({\bf Z})}}].
$$

We know that the marginal distribution of the observations, $p({\bf X})$, is constant. Therefore, minimizing $E_q[\log{\frac{p({\bf Z\big\rvert \bf X})}{q({\bf Z})}}]$, which is the KL divergence between $q({\bf Z})$ and the posterior distribution, is equivalent to maximizing $\mathcal{L}(q) = E_q[\log{\frac{p({\bf X,\bf Z})}{q({\bf Z})}}]$. This maximization forms the **objective of Variational Bayes**. 

The next section describes a special case of VB maximization called Mean-Field approximation (MFA), where $q({\bf Z}) = \prod_{i=1}^{K}q_i({\bf Z}_i)$. The integer $K$ is the number of latent variables. 


### Maximizing the VB objective
We start by inserting the mean-field approximation into the components of $\mathcal{L}(q) =  E_q[\log{{p({\bf X,Z})}]-E_q[\log{q({\bf Z})}}]$, separately. The following approach has been adopted from [2]. 

**Part 1:**    $E_q[\log q({\bf Z})]$

$$
E_q[\log{q({\bf Z})}] = {\bf\displaystyle\int}{q({\bf Z})\log{q({\bf Z})}}d{\bf Z}
$$
 where ${\bf\displaystyle\int}f({\bf Z})d{\bf Z}$ represents $K$ integrations for each ${\bf Z}_i, i = 1\dots K$.
Introduce MFA into the expression of $q({\bf Z})$:
$$
E_q[\log{q({\bf Z})}] =  {\bf\displaystyle\int}{\prod_{j}q_j({\bf Z}_j)\log{\prod_{i}q_i({\bf Z}_i)}}d{\bf Z}_1\dots d{\bf Z}_K\\
$$
$$\hspace{2cm} =  {\bf\displaystyle\int}{\prod_{j}q_j({\bf Z}_j)\sum_{i}\log{q_i({\bf Z}_i)}}d{\bf Z}_1\dots d{\bf Z}_K\\
$$
$$\hspace{2cm} =  \sum_{i}{\bf\displaystyle\int}{\prod_{j}q_j({\bf Z}_j)\log{q_i({\bf Z}_i)}}d{\bf Z}_1\dots d{\bf Z}_K\\
$$
$$\hspace{2.5cm} =  \sum_{i}\displaystyle\int\dots\displaystyle\int{\prod_{j}q_j({\bf Z}_j)\log{q_i({\bf Z}_i)}}d{\bf Z}_1\dots d{\bf Z}_K\\
$$
In the last equation, ${\bf\displaystyle\int}$ has been broken down into $\displaystyle\int\dots\displaystyle\int$. 
 It is straight-forward to show that for each $i$ in the summation: $$\displaystyle\int\dots\displaystyle\int{\prod_{j}q_j({\bf Z}_j)\log{q_i({\bf Z}_i)}}d{\bf Z}_1\dots d{\bf Z}_K = \int{q_i({\bf Z}_i)\log{q_i({\bf Z}_i)}}d{\bf Z}_i$$

, which uses the property of the distribution $\displaystyle\int q_j({\bf Z}_j)d{\bf Z}_j = 1$. Therefore,
$$E_q[\log{q({\bf Z})}]  =  \sum_{i}E_{q_i}[\log{q_i({\bf Z}_i)}].
$$
The term $\sum_iE_{q_i}[\log(q_i({\bf Z}_i))]$ can further be decomposed as:
$$
\sum_iE_{q_i}[\log(q_i({\bf Z}_i))] = E_{q_j}[\log(q_j({\bf Z}_j))] + \sum_{i\ne j}E_{q_i}[\log(q_i({\bf Z}_i))] \hspace{1cm}(1)$$.

**Part 2:**    $E_q[\log p({\bf X,\bf Z})]$

The other reformulation is obtained by applying the chain rule of probabilities to $p({\bf X,\bf Z})$.
for any $1\le j \le K$

$$p({\bf X},{\bf Z}_1 \dots {\bf Z}_K) = p({\bf X}) p({\bf Z}_1\dots{\bf Z}_{j-1},{\bf Z}_{j+1},\dots,{\bf Z}_{K}\big\rvert{\bf X})p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_1\dots{\bf Z}_{j-1},{\bf Z}_{j+1},\dots,{\bf Z}_{K})$$

We can define $\bar{\bf Z}_{j}$ as short for the collection of all latent variables, except ${\bf Z}_j$. 

$$p({\bf X},{\bf Z}_1 \dots {\bf Z}_K) = p({\bf X}) p(\bar{\bf Z}_j\big\rvert{\bf X})p({\bf Z}_j\big\rvert {\bf X},\bar{\bf Z}_j)$$

Therefore, 

$$\log p({\bf X},{\bf Z}) = \log p({\bf X}) + \log p({\bf Z}_j\big\rvert {\bf X},\bar{\bf Z}_j) + \log p(\bar{\bf Z}_j\big\rvert{\bf X})$$

Inserting the resulting expression for $\log p({\bf X},{\bf Z})$ into $E_q[\log p({\bf X,\bf Z})]$: 

$$
E_q[\log p({\bf X,\bf Z})] = \displaystyle\int\dots\displaystyle\int\prod_{i}q_i({\bf Z}_i)\big(\log p({\bf X}) + \log p({\bf Z}_j\big\rvert {\bf X},\bar{\bf Z}_j) + \log p(\bar{\bf Z}_j\big\rvert{\bf X})\big)d{\bf Z}_1\dots d{\bf Z}_K
$$

$$
\hspace{2cm} = \log p({\bf X}) + {\bf\displaystyle\int}\prod_{i\ne j}q_i({\bf Z}_i)\log p(\bar{\bf Z}_j\big\rvert{\bf X})d\bar{\bf Z}_j + 
\displaystyle\int q_j({\bf Z}_j)\Big({\bf\displaystyle\int}\prod_{i\ne j}q_i({\bf Z}_i)\log p({\bf Z}_j\big\rvert{\bf X},\bar{\bf Z}_j)d\bar{\bf Z}_j\Big) d{\bf Z}_j
$$

Similarly, $\bar{q}_j$ is the collection of all p.d.fs, except $q_j$. 

$$
E_q[\log p({\bf X,\bf Z})] = \log p({\bf X}) + E_{\bar{q}_j}[\log p(\bar{\bf Z}_j\big\rvert{\bf X})] + E_{q_j}[E_{\bar{q}_j}[\log p({\bf Z}_j\big\rvert\bar{\bf Z}_j,{\bf X})]]\hspace{1cm}(2)
$$

Notice that the first two terms do not depend on $q({\bf Z}_j)$. 

Using (1) and (2), we can now rewrite $\mathcal{L}(q)$: 

$$
\mathcal{L}(q) = \log p({\bf X}) + E_{\bar{q}_j}[\log p(\bar{\bf Z}_j\big\rvert{\bf X})] + E_{q_j}[E_{\bar{q}_j}[\log p({\bf Z}_j\big\rvert\bar{\bf Z}_j,{\bf X})]] - E_{q_j}[\log(q_j({\bf Z}_j))] - \sum_{i\ne j}E_{q_i}[\log(q_i({\bf Z}_i))]
$$ 

Note that the choice of $j$ is completely arbitrary. 

### Coordinate Ascent
Separating each individual $q_j$ in the expression for $\mathcal{L}(q)$ allows a step-by-step maximization along each $q_j$. This maximization is referred to as $q_j$. To find the maximum point for $q_j$, we must calculate the functional derivative of $\mathcal{L}(q)$ with respect to $q_j$. 

$$
\frac{\partial\mathcal{L}(q)}{\partial q_j} = \frac{\partial}{\partial q_j}\Bigg[\int q_{j}({\bf Z}_j)E_{\bar{q}_j}\big[p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_{\bar{j}})\big]d{\bf Z}_j
                                   - \int q_j({\bf Z}_j)\log(q_j({\bf Z}_j))d{\bf Z}_j\Bigg] = 0 \\
                                   E_{\bar{q}_j}\big[p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_{\bar{j}})\big] - \log(q_j({\bf Z}_j)) - 1 = 0
$$
where we have used the well-known results functional derivatives for entropy (i.e., $\frac{\partial }{\partial p}\int p\log(p)dx = \log(p)+1$). The stationary point from above results in: 

$$
\log(q_j({\bf Z}_j)) + \log(e) = E_{\bar{q}_j}\big[p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_{\bar{j}})\big]\\
\log(q_j({\bf Z}_j)e) = E_{\bar{q}_j}\big[p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_{\bar{j}})\big]\\
q_j({\bf Z}_j) \propto \exp[E_{\bar{q}_j}\big[p({\bf Z}_j\big\rvert {\bf X},{\bf Z}_{\bar{j}})\big]]
$$
The r.h.s does not depend on $q_j$. 

### References
[1] Bishop, C., 2007. Pattern Recognition and Machine Learning (Information Science and Statistics), 1st edn. 2006. corr. 2nd printing edn. Springer, New York.

[2] Neiswanger, W., 2017. Probabilistic Graphical Models, Sprint 2017 lectures, Carnegie Mellon University. 


```python

```
