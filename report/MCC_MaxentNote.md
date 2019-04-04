Motivation
==========

More than half of the US population has at least one chronic condition
and over 30% have two or more chronic conditions. Patients with 2 more
chronic conditions account for 70% of the US healthcare expenditures.
Medical interventions and policy initiatives generally focus on a single
health condition rather than combinations of health conditions. This
approach risks ignoring unique challenges faced by patients with
multiple conditions. The Centers for Disease Control and Prevention
notes that “as the number of chronic conditions in an individual
increases, the risks of the following outcomes also increase: mortality,
poor functional status, unnecessary hospitalizations, adverse drug
events, duplicative tests, and conflicting medical advice.”

While the chronic conditions that are individually most prevalent in the
US population are well known, their *joint prevalence*, that is the
probability estimate of how frequently a particular combination of
conditions co-occurs in the population is harder to estimate. This is
because the number of individuals in a data sample who have a specific
combination of conditions tends to be low. For instance, in the US, it
is estimated that only 5.0% of the population has 4 chronic conditions.
This means that in a dataset of 1000 nationally representative
individuals only 50 individuals (approximately) will have 4 chronic
conditions. If we look at a specific combination of 4 chronic conditions
(say diabetes, hypertension, cholesterol and arthritis), the number of
individuals who have these four conditions will be even smaller. Indeed,
a vast majority of combinations of four will have no representation at
all. Estimating joint prevalence directly through data therefore risks
underestimating the true prevalence and thereby the impact that such
patients have on the health system.

The above discussion is one example of a statistical estimation
challenge related to patients with multiple chronic conditions (MCC).
Other open questions include identifying combinations of chronic
conditions that occur most frequently in a population, workload
estimation for specialty providers seen by MCC patients, and evaluation
of new delivery models that coordinate care more effectively and improve
outcomes. These research questions are summarized below:

1.  Which combination of $k$ chronic conditions is the most prevalent in
    a dataset? \[Market Basket Analysis – we have some preliminary
    results\]

2.  How do we accurately estimate the joint prevalence probabilities for
    a combination of $k$ conditions? Since chronic conditions are highly
    age-dependent, how do we estimate transition probabilities from one
    year to the next? \[Max Ent and Markov transitions with Max Ent – we
    will have some preliminary results\]

3.  How do combinations of chronic conditions differ with regard to
    capacity and resources needed from different speciality providers?
    \[Yet to be figured out.\]

4.  How do we measure the impact of interventions that target a
    combination of conditions rather than individual conditions? \[Yet
    to be figured out.\]


MaxEnt Preliminaries
====================

Consider a disease model in which an individual can simultaneously have
up to $k> 1$ diseases. We represent the state of an individual by a
vector $r=(r_1,\ldots,r_k)$, where, for $j\in[1..k]$, we set $r_j=1$ if
the individual has disease $j$ and $r_j=0$ otherwise. Thus the state
$r=(0,\ldots,0)$ represents complete health. Denote by ${\mathcal{R}}$
the set of possible states. Suppose that a dataset ${\mathcal{D}}$ is
available that contains the disease state for $N$ individuals. We view
${\mathcal{D}}$ as $n$ independent and identically distributed
(i.i.d.) samples from a population distribution $p$, and our goal is to
compute an estimate of $p$.

If there is sufficient data, then for each $r\in{\mathcal{R}}$ we can
estimate $p(r)$ by the maximum likelihood estimator
${\hat{p}}_{\text{ML}}(r)=N_r/N$, where $N_r$ is the number of
individuals in ${\mathcal{D}}$ whose state equals $r$. That is,
${\hat{p}}_{\text{ML}}$ is simply the empirical distribution of the
data. In practice, the number of data points $N$ is typically much less
than the $2^k$ possible disease states. While for some states $r$ there
will be enough data to adequately estimate $p(r)$ by
${\hat{p}}_{\text{ML}}(r)$ as above, for most states there is
insufficient data to form an estimate; indeed, we will erroneously
conclude that many states have probability 0. (One strategy is to simply
augment the data by adding an artificial data point for each missing $r$
value, but we seek a more principled approach.)

To further complicate the problem, it is often desirable to expand the
state definition to include covariates associated with the diseases,
such as age, body mass index, smoker versus non-smoker and so on. The
presence of these additional features can dramatically increase the size
of the state space ${\mathcal{R}}$. Moreover, the estimation problem is
more complicated, in that features can be real-valued, integer, or
categorical.

In the case where all features are discrete (integer or categorical), we
estimate $p$ by computing a small, carefully chosen set of reliable
statistics from the training data ${\mathcal{D}}$ and using these
statistics to develop a parametric model from which an approximate
version of $p(r)$ can be computed. The key idea is to carefully select a
set ${\mathcal{C}}$ of important *constraints* that are satisfied by
$p$, and then to approximate $p$ by the “simplest” distribution that
obeys the constraints in ${\mathcal{C}}$; i.e., we impose no further
implicit or explicit assumptions. Following standard
practice [@jaynes82], we formalize the notion of “simplest distribution”
as the distribution $p$ satisfying the given constraints that has the
maximum *entropy* value $H(p)$, where
$$H(p)=-\sum_{r\in{\mathcal{R}}}p(r)\log p(r).$$ This maximum entropy
(maxEnt) approach is well known in the machine-learning and linguistics
communities—see [@Malouf02]—and can also be viewed as a principled
approach to smoothing the empirical distribution. To deal with
continuous features, we use kernel density estimation, conditioned on
the value of the discrete features.

A desirable feature of the maxEnt framework is that it allows a “pay as
you go” approach that controls the tradeoff between the resolution of
the model and the cost of computing the model. The more data points and
constraints, the greater the accuracy of the smoothed approximation. In
the following sections, we provide the details of the approach.

Basic MaxEnt for Discrete Features
==================================

We first consider the simplest case with no covariates, so that each
state $r$ comprises $k$ binary values. We then extend our methods to
arbitrary discrete features and continuous features.

Binary Features
---------------

In the binary setting, the maxEnt problem can be stated as follows. Let
${\mathcal{P}}$ denote the set of all possible probability distributions
over ${\mathcal{R}}$. For each subset
${\mathcal{R}}_c\subseteq{\mathcal{R}}$, denote by $f_c$ the indicator
function of ${\mathcal{R}}_c$, so that $f_c(r)=1$ if
$r\in{\mathcal{R}}_c$ and $f_c(r)=0$ otherwise. We number these subsets
from 1 to $2^{|{\mathcal{R}}|}$. Set $a_c=M_c/N$, where $M_c$ is the
number of the $N$ data points $r\in{\mathcal{D}}$ that lie in
${\mathcal{R}}_c$ and thus satisfy $f_c(r)=1$. If, as in the previous
section, we define $N_r$ as the number of persons in ${\mathcal{D}}$
having disease state $r$, then we can also compute $a_c$ as
$$\label{eq:defac}
a_c=\frac{1}{N}\sum_{r\in{\mathcal{R}}}N_rf_c(r).$$ The maxEnt problem
is then $$\label{eq:maxent}
\begin{split}
&\operatorname*{maximize}_{p\in\mathcal{P}}H(p)\text{ such that}\\
&\sum_{r\in{\mathcal{R}}}f_c(r)p(r)=a_c\text{ for }c\in{\mathcal{C}},
\end{split}$$ Where ${\mathcal{C}}\subseteq [1..2^{|{\mathcal{R}}|}]$ is
the set of maxEnt constraints. For each $c\in C$, the corresponding
maxEnt constraint requires that $p({\mathcal{R}}_c)=a_c$. That is, the
probability of ${\mathcal{R}}_c$ must match the empirical probability of
${\mathcal{R}}_c$ in ${\mathcal{D}}$, and we seek the simplest
distribution that satisfies the constraints in ${\mathcal{C}}$.

The most basic constraints in ${\mathcal{C}}$ specify the marginal
occurrence probability for each disease. That is, for each disease $i$,
the corresponding constraint function $f_c$ satisfies $f_c(r)=1$ if and
only if $r_i=1$, so that
${\mathcal{R}}_c=\{\,r\in{\mathcal{R}}:r_i=1\,\}$. Equivalently, if $R$
denotes a random sample from $p$, then the constraint can be expressed
as ${\operatorname{Pr}[R_i=1]}=N_i/N$, where $N_i$ is the fraction of
individuals in ${\mathcal{D}}$ having disease $i$. We include such
constraints because of their inherent importance and because we can
obtain a relatively accurate estimate of $a_c$ from the training data;
indeed, most diseases will appear in a non-negligible fraction of
individuals in ${\mathcal{D}}$, so data sparsity is not an issue.

Other constraints can be added to ${\mathcal{C}}$, corresponding to a
small number of “important” statistical features of $p$ as specified by
the user. Typically, these important features specify statistical
dependencies between the various diseases that should be respected in
any approximation to $p$. For example, denoting by $N_{ij}$ the number
of persons in ${\mathcal{D}}$ having both diseases $i$ and $j$, one
might specify a constraint of the form
$${\operatorname{Pr}[R_i=1\wedge R_j=1]}=a_{ij},$$ where
$a_{ij}=N_{ij}/N$ is close to 1, in order to capture a known, strong
positive dependency between the diseases. Similarly, a constraint of the
form $${\operatorname{Pr}[R_i=1\wedge R_j=0]}=a_{i\bar{\jmath}},$$ can
capture a strong negative dependency—persons with disease $i$ tend not
to have disease $j$. As above, these constraints can be expressed in
terms of indicator functions $f_c$. Note that the constraints, being
based on a finite set of patient data, are approximate rather than
exact; we return to this issue below. Also note that, in principle, the
user can directly specify the $a_c$ constants based on expert knowledge
about rule behavior. The user must then ensure, however, that the
resulting set ${\mathcal{C}}$ of constraints is consistent; see, e.g.,
Markl et al. [@Markl2007] for a discussion of this issue. To avoid
situations with insufficient data, to avoid overfitting, and to control
computational costs, we typically include only bivariate probabilities
and only for the most correlated feature pairs.

Discrete Features
-----------------

The foregoing setup can be extended to arbitrary discrete features. The
idea is to encode a given feature $s$ having $m$ possible values as a
binary vector $v$ of length $m$, where $v_i=1$ if and only if the
feature takes on the $i$th possible value. The resulting set of marginal
maxEnt constraints require that the marginal distribution of $s$ matches
the empirical distribution in ${\mathcal{D}}$. (Note that we add an
additional constraint which requires that the sum of $v_i$’s equal 1
with probability 1.) If $m$ is very large, making such an encoding
impractical, one alternative for an integer-valued feature is to replace
the $m$ maxEnt constraints with the single constraint that the expected
value of $s$ under $p$ matches the average value in ${\mathcal{D}}$.
Additional constraints corresponding to other features of the
distribution of $s$, such as higher moments or tail probabilities, can
also be added. A potential drawback to this approach is that such
aggregate constraints can severely underspecify the distribution;
see [@YuDA09]. An alternative approach aggregates feature values into
bins, but in practice the maxEnt fit can be extremely sensitive to the
bin specification. One simple strategy is to approximate the
distribution of $s$ by a continuous distribution and use, e.g., a kernel
density approach as described later.

Computing the maxEnt distribution
---------------------------------

The standard approach to computing the maxEnt distribution converts the
problem into an equivalent formulation, based on duality theory for
convex optimization problems. Specifically, assuming that $a_c$ is
estimated from the training data as discussed previously, it can be
shown—see, e.g., [@PietraPL97]—that the entropy-maximizing distribution
must be of the form $$\label{eq:expmodel}
p(r)=p(r;\theta)=\frac{1}{Z(\theta)}\exp\Bigl\{\sum_{c\in{\mathcal{C}}}\theta_c f_c(r)\Bigr\},$$
where $\theta=\{\,\theta_c:c\in{\mathcal{C}}\,\}$ and
$Z(\theta)=\sum_{r\in{\mathcal{R}}}\exp\bigl\{\sum_{c\in{\mathcal{C}}}\theta_c f_c(r)\bigr\}$
is a normalizing constant which ensures that
$\sum_{r\in{\mathcal{R}}}p(r)=1$. Moreover, solving is equivalent to
solving the problem $$\label{eq:maxlike}
\operatorname*{maximize}_{\theta_c:c\in{\mathcal{C}}}\sum_{r\in{\mathcal{R}}}N_r\sum_{c\in{\mathcal{C}}}\theta_cf_c(r)-N\log Z(\theta),$$
which can be viewed as computing a maximum likelihood estimate of
$\theta$ based on the data in ${\mathcal{D}}$. The optimization problem
typically must be solved numerically; as discussed in [@Malouf02],
standard optimization algorithms such as L-BFGS-B [@ZhuBLN97] tend to
outperform techniques such as iterative scaling, even though the latter
techniques were developed expressly for maxEnt problems.

Avoiding Overfitting
--------------------

Since the data used to fit the population distribution $p$ is based on a
finite sample, some care must be taken to avoid overfitting our model
for $p$. Several approaches to this problem have been proposed in the
literature. Perhaps the simplest approach is to take a Bayesian
viewpoint and introduce a “prior” distribution on the parameters in
$\theta$ and then use a maximum posterior estimate rather than a maximum
likelihood estimate as above. In the case of a gaussian prior, this
reduces to solving a modified version of : $$\label{eq:maxlikegauss}
\operatorname*{maximize}_{\theta_c:c\in{\mathcal{C}}}\sum_{r\in{\mathcal{R}}}N_r\sum_{c\in{\mathcal{C}}}\theta_cf_c(r)-N\log Z(\theta) -\sum_c (\theta^2_c/ 2\sigma^2_c),$$
where for each $c\in{\mathcal{C}}$ the parameter $\sigma^2_c$ is the
variance of the gaussian prior for $\theta_c$; see [@ChenR00]. The
$\sigma^2_c$ values can be selected using cross-validation.
Goodman [@Goodman04] proposes the use of an exponential prior and claims
improved accuracy on certain NLP problems. An alternative
approach [@KazamaT05] replaces the equality constraints in the maxEnt
problem by box-type inequality constraints. I.e., the constraints are
now of the form
$$\sum_{r\in{\mathcal{R}}}f_c(r)p(r)\in[a_c-\eps_c,a_c+\eps_c],$$ where
the half-width $\eps_c$ is chosen to reflect the uncertainty in $a_c$ as
an estimator of $p({\mathcal{R}}_c)$.

Continuous Features {#sec:contf}
-------------------

The discussion so far has centered around discrete-valued features, but
continuous-valued features often appear as disease covariates. The
challenges engendered by discrete features are similar to those
associated with discrete features having a very large domain. As in the
latter case, problems arise in approaches based either on specifying
constraints for summary statistics or on binning of values. Yu et
al. [@YuDA09] propose an approach that, roughly speaking, uses binning
but lets the number of bins go to infinity, yielding a continuous weight
function $\theta$. This continuous functions is approximated via cubic
splines, yielding a standard maxEnt problem but with a set of $K$
transformed features corresponding to each original continuous features.
Besides blowing up the number of features, the derived features are not
directly interpretable, which can hinder understanding of the maxEnt
model.

An alternative, simple approach directly uses kernel density
methods—see, e.g., [@Scott92]—to estimate the conditional distribution
of the continuous features, given the values of the discrete features.
This approach works well only when there are not too many continuous or
discrete features. As discussed in Section \[sec:factor\] below,
however, we typically break up the maxEnt model (either exactly or
approximately) into independent submodels, and each submodel typically
contains a small number of features. For convenience, write $r=(u,v)$,
where $u$ and $v$ comprise the continuous and discrete feature values,
and set ${\mathcal{R}}_d=\{\,v:(u,v)\in{\mathcal{R}}\,\}$. The idea is
to compute, for each $v\in{\mathcal{R}}_d$, a kernel density estimator
of the joint conditional distribution $p_c({\,\cdot\,}\mid v)$ of the
continuous features, given that the subvector of discrete features
equals $v$. For $r=(u,v)\in{\mathcal{R}}$, denote by $l$ the number of
continuous features and set $\pi_c(r)=u$ and $\pi_d(r)=v$, so that
$\pi_c$ and $\pi_d$ are the projections onto the continuous and discrete
components of $r$. Then set
$$p_c(u\mid v)=\frac{1}{|{\mathcal{D}}_v|}\sum_{r\in{\mathcal{D}}_v}|H|^{-1/2}K\Bigl(H^{-1/2}\bigl(u-\pi_c(r)\bigr)\Bigr),$$
for $v\in{\mathcal{R}}_d$, where
${\mathcal{D}}_v=\{\,r\in{\mathcal{D}}:\pi_d(r)=v\,\}$. Here $K$ is a
*kernel* function that smooths out the influence of each data point in
${\mathcal{D}}_v$, and $H$ is an $l\times l$ symmetric and positive
definite *bandwidth matrix* that controls how fast the influence of a
data point decreases with distance from the point. A typical choice of
kernel function is standard multivariate gaussian:
$K(x)=(2\pi)^{-l/2}e^{-\Vert x\Vert^2/2}$. Generally, the choice of
bandwidth is more important than the particular form of kernel. Many
options are available—one common choice is to take $H$ to be a diagonal
matrix and then determine each of the diagonal elements using the data
driven “plug-in” bandwidth formula of Sheather and Jones [@SheatherJ91],
which was subsequently improved in [@LiaoWL10]. See [@WandJ94] for a
treatment of plug-in formulas for general bandwidth matrices.

It may be the case that $|{\mathcal{D}}_v|$ is small for some
$v\in{\mathcal{R}}_d$, so that there are not enough data points on which
to fit the conditional distribution. In this case, we can simply fit a
distribution to the pooled data in
${\mathcal{D}}=\bigcup_v{\mathcal{D}}_v$. A more sophisticated approach
might greedily drop one feature at a time out of $v$ until a sufficient
number of training points are obtained.

Performance Optimizations
=========================

Fitting maxEnt models can be expensive when the number of features $k$,
constraints $|{\mathcal{C}}|$, or number of data points $N$ is large. In
this section we discuss exact and approximate methods for speeding up
maxEnt distribution fitting.

Sampling Approach for Large $N$
-------------------------------

For large $N$, Schofield [@Schofield04] provides a sampling-based method
for quick approximate calculation of $Z(\theta)$ and
$\{\,a_c:c\in{\mathcal{C}}\,\}$ during optimization iterations. These
methods can be used to handle continuous features as well, but only when
the corresponding constraints are on the values of summary statistics.
As mentioned previously, such constraints generally under-specify the
distribution.

Factorization for Large $|{\mathcal{C}}|$ and $k$ {#sec:factor}
-------------------------------------------------

An exact method for accelerating maxEnt computations decomposes the set
of $k$ features into $m>1$ partitions. Specifically, write
$i\leftrightarrow j$ if and only if there exists at least one maxEnt
constraint that involves both feature $i$ and feature $j$, and denote by
$\leftrightsquigarrow$ the transitive closure of $\leftrightarrow$. Then
features $i$ and $j$ belong to the same partitions if and only if
$i\leftrightsquigarrow j$. We correspondingly partition the state as
$r=(r^{(1)},\ldots,r^{(m)})$. Recall that, in the absence of
constraints, features are independently distributed under the maxEnt
distribution. In a similar manner, the features within a given component
$r^{(i)}$ will be statistically independent, but variables in different
partitions will be independent; see, e.g., [@Markl2007 App. B].
Therefore, we can fit a maxEnt model $p^{(i)}$ to the data for each
component $r^{(i)}$ and compute $p$ as
$p(r)=p^{(1)}(r^{(1)})\times\cdots\times p^{(m)}(r^{(m)})$.

As mentioned previously, if we include multi-feature constraints for
only the most important statistical dependencies, then the maxEnt model
will decompose into many small submodels, each involving only a few
features. This decomposition can greatly reduce the computational
burden, since each state space will be relatively small. Indeed, many
partitions might contain exactly one feature, in which case the
corresponding distribution submodel can trivially be computed
analytically. If a partition is too large to be computed in reasonable
time, then approximate methods can be used. One popular method is the
“piecewise likelihood” method in [@Sutton2007]. Roughly speaking, the
sets of features referred to in the various partition-based constraints
are initially treated as if they were independent (even though the sets
are not even disjoint), and then dependence is re-introduced via a
global normalization constant.

Choosing maxEnt Constraints
===========================

As can be seen from the previous section, a judicious choice of maxEnt
constraints can result in good computational efficiency via
factorization of the maxEnt model. The key challenge is to identify the
most important statistical dependencies between features. Typically we
focus on the most important bivariate dependencies, since these lead to
constraints for which the $a_c$ constant can be reliably estimated. We
need a notion of statistical dependency that will encompass integer,
categorical, or continuous data. An appealing candidate is the
information-theoretic “L-measure” defined by Lu [@Lu11] as follows. For
a pair or random variables $(X,Y)$ with joint distribution
$\gamma(x,y)$, let $I(X;Y)$ be the mutual information between $X$ and
$Y$, defined as
$$I(X;Y)=\sum_{x,y}\gamma(x,y)\log\frac{\gamma(x,y)}{\gamma(x)\gamma(y)}.$$
Denote by $A_{X,Y}$ the set of all pairs of random variables $(U,V)$
having the same marginal distributions as $X$ and $Y$, and set
$W(X;Y)=\sup_{(U,V)\in A_{X,Y}} I(U;V)$. We have $$W(X;Y)=
\begin{cases}
\min\bigl(H(X),H(Y)\bigr)&\text{$X$, $Y$ discrete};\\
H(Y)&\text{$X$ continuous, $Y$ discrete};\\
H(X)&\text{$X$ discrete, $Y$ continuous};\\
+\infty&\text{$X$, $Y$ continuous},
\end{cases}$$ where $H$ denotes entropy as before. Then L-measure is
defined as
$$L(X,Y)=\Biggl[1-\exp\Bigl\{\frac{-2I(X;Y)}{1-I(X;Y)/W(X;Y)}\Bigr\}\Biggr]^{1/2}.$$
In the fully continuous case, L-measure reduces to the Linfoot
measure [@Linfoot57]. L-measure has a number of desirable properties. By
definition, it applies to integer, categorical, and continuous features.
It is symmetric in $X$ and $Y$ and takes values in $[0,1]$, where a
value of 0 implies independence between $X$ and $Y$, and a value of 1
implies a strict functional relationship (not necessarily linear)
between the two variables. L-measure is invariant under continuous,
strictly increasing transformations and, in the special case where
$(X,Y)$ is bivariate gaussian, it reduces to the absolute value of the
correlation coefficient.

To compute the needed entropy and mutual information values needed for
L-Distance, available techniques include the “shrinkage” estimator of
Hausser and Strimmer [@HausserS09] for the discrete-discrete case, the
nearest neighbor estimator of Kraskov et al. [@KraskovSG03] for the
continuous-continuous case, and a modified version of this estimator due
to Ross [@Ross14] for the mixed discrete-continuous case. For discrete
features having many distinct values and few duplicated values in the
data set, the Ross estimator fails; in this case, the discrete feature
can be treated as continuous and the estimator of Kraskov et al. can be
applied.

The presence of a discrete feature having many distinct values also
complicates the use of L-measure. Specifically, if a discrete
feature $X$ has very many distinct values, it’s mutual information with
another discrete feature $Y$ may be close to 1 even if features $X$ and
$Y$ are in fact independent. To see why, first observe that if we treat
a continuous feature $X$ as discrete, then its mutual information with
any other discrete feature $Y$ will equal 1. The reason is that, given a
set of $(x,y)$ pairs of feature values, the $x$’s are all distinct, so
that a value $x$ uniquely determines its corresponding $y$. A discrete
feature having many distinct values behaves in roughly the same way,
leading to erroneously high values of mutual information.

A solution to this problem can be found by normalizing the mutual
information values “with respect to chance”, as in Nguyen et
al. [@NguyenEB09]. Given a set ${\mathcal{S}}$ of i.i.d. samples $(x,y)$
from $(X,Y)$,[^1] we first compute $\mu(X;Y)$, the expected value of
$I(X;Y)$ under a random permutation of the $y$ values in the set of
pairs. When $X$ and $Y$ are discrete, this random permutation model can
also be viewed as randomly and uniformly selecting a contingency table
$M$ from the set ${\mathcal{M}}$ of all contingency tables whose
marginal row and column totals are the same as that of the table $M_0$
that corresponds to the original data in ${\mathcal{S}}$. Defining
marginal totals $n_{i\cdot}=|\{\,(x,y)\in{\mathcal{S}}:x=x_i\,\}|$ and
$n_{\cdot j}=|\{\,(x,y)\in{\mathcal{S}}:y=y_j\,\}|$ and setting
$N=|{\mathcal{S}}|$, Nguyen et al. show that $$\label{eq:muxy}
\mu(X;Y)=\sum_{i,j}\sum_{n_{ij}}\frac{n_{ij}}{N}\log\Bigl(\frac{n_{ij}N}{n_{i\cdot}n_{\cdot j}}\Bigr)P(n_{ij}),$$
where the outer sum runs over all values $x_i$ and $y_j$ of $X$ and $Y$,
the inner sum runs from $n_{ij}=\max(0,n_{i\cdot}+n_{\cdot j}-N)$ to
$n_{ij}=\min(n_{i\cdot},n_{\cdot j})$, and
$$P(n_{ij})=\Bigl(\frac{N-n_{\cdot j}}{n_{i\cdot}-n_{ij}}\Bigr)\Bigl(\frac{n_{\cdot j}}{n_{ij}}\Bigr)\bigg/\Bigl(\frac{n}{n_{i\cdot}}\Bigr).$$
When at least one of the features is continuous, we can estimate
$\mu(X;Y)$ via Monte Carlo. We then compute the normalized mutual
information as ${\tilde{I}}(X;Y)=I(X;Y)-\mu(X;Y)$, and define the
normalized L-measure as
$${\tilde{L}}(X,Y)=\Biggl[1-\exp\Bigl\{\frac{-2{\tilde{I}}(X;Y)}{1-{\tilde{I}}(X;Y)/{\tilde{W}}(X;Y)}\Bigr\}\Biggr]^{1/2},$$
where
${\tilde{W}}(X;Y)=\sup_{(U,V)\in A_{X,Y}} {\tilde{I}}(U;V)=W(X;Y)-\mu(X;Y)$.

For a pair of discrete features $(X,Y)$ that is determined to be highly
dependent according to normalized L-measure, we need to identify the
most important $(x_i,y_j)$ value pairs to use in maxEnt constraints.
This can be achieved by ranking the pairs with respect to their absolute
contribution to the empirical mutual information, i.e., rank according
to
$$\delta(x_i,y_j)=\Bigl\vert \frac{n_{ij}}{n}\log\Bigl(\frac{n_{ij}n}{n_{i\cdot}n_{\cdot j}}\Bigr)\Bigr\vert,$$
where $n_{i\cdot}$, $n_{\cdot j}$ and $n$ are as and
$n_{ij}=|\{\,(x,y)\in{\mathcal{S}}:x=x_i\text{ and }y=y_j\,\}|$. For
each $(x_i,y_j)$ pair selected, the corresponding maxEnt constraint
equates the joint probability that $X=x_i$ and $Y=y_j$ to the empirical
joint frequency in the data. The number of such $(x_i,y_j)$ pairs is
chosen so as to balance model accuracy with computational cost.

Fitting the Distribution
========================

Putting the previous steps together, we fit a smoothed distribution to
the data set ${\mathcal{D}}$ as follows.

1.  Compute normalized L-measures between all feature pairs, and select
    the $k$ most dependent pairs, where $k$ is chosen to trade off
    accuracy and computational cost. The value of $k$ may also be chosen
    indirectly using a cutoff value for L-measure, where the cutoff is
    chosen to balance accuracy and cost. For each selected pair of
    discrete features, select specific pairs of feature values to use in
    maxEnt constraints, as described at the end of the previous section.

2.  Partition the features to create one or more submodels. Features are
    partitioned as described in Section \[sec:factor\], except that we
    extend the partitioning scheme to include continuous features by now
    writing $i\leftrightarrow j$ if and only if $(i,j)$ is one of the
    $k$ most-dependent pairs selected in Step 1. Steps 3–5 are now
    performed for each submodel $i$, where $i=1,2,\ldots,m$. For
    submodel $i$, write $r^{(i)}=(u^{(i)},v^{(i)})$, where $u^{(i)}$ an
    $v^{(i)}$ are the sets of continuous and discrete feature values for
    state $r^{(i)}$. Denote by ${\mathcal{R}}_d^{(i)}$ the state space
    of $v^{(i)}$.

3.  Encode all non-binary discrete features as binary vectors (perhaps
    recategorizing features with many distinct values as effectively
    continuous), and rewrite the maxEnt constraints accordingly.

4.  Solve the optimization problem to fit the $\theta_c$ parameters and
    hence compute the joint distribution $p_d^{(i)}(v^{(i)})$ of the
    binary features.

5.  For each $v^{(i)}\in{\mathcal{R}}_d^{(i)}$, compute a conditional
    kernel density estimator $p_c^{(i)}(u^{(i)}\mid v^{(i)})$ of the
    joint distribution of the continuous features. If there are too few
    data points to fit $p_d^{(i)}$ reliably, then fit a density
    estimator to pooled data as discussed in Section \[sec:contf\].

6.  For $r^{(i)}=(u^{(i)},v^{(i)})$, define the submodel $i$ as
    $p^{(i)}(r^{(i)})=p_d^{(i)}(v^{(i)})p_c^{(i)}(u^{(i)}\mid v^{(i)})$.

7.  Define the overall model as
    $p(r)=p^{(1)}(r^{(1)})\times\cdots\times p^{(m)}(r^{(m)})$.

[^1]: We assume throughout that every possible marginal value of $X$ and
    of $Y$ is represented in ${\mathcal{S}}$.
