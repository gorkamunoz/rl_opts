
<p align="center">
<img width="350" src="https://github.com/gorkamunoz/rl_opts/blob/master/nbs/figs/logo_midjourney_scaled.png?raw=true">
</p>
<h1 align="center">
RL-OptS
</h1>
<h4 align="center">
Reinforcement Learning of Optimal Search strategies
</h4>
<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->
<p align="center">
<a href="https://zenodo.org/badge/latestdoi/424986383"><img src="https://zenodo.org/badge/424986383.svg" alt="DOI"></a>
<a href="https://badge.fury.io/py/rl_opts"><img src="https://badge.fury.io/py/rl_opts.svg" alt="PyPI version"></a>
<a href="https://badge.fury.io/py/b"><img src="https://img.shields.io/badge/python-3.9-red" alt="Python version"></a>
</p>

This library builds the necessary tools needed to study, replicate and
develop the results of the paper: [“Optimal foraging strategies can be
learned and outperform Lévy walks”](https://arxiv.org/abs/2303.06050) by
*G. Muñoz-Gil, A. López-Incera, L. J. Fiderer* and *H. J. Briegel*.

### Installation

You can access all these tools installing the python package `rl_opts`
via Pypi:

``` python
pip install rl-opts
```

You can also opt for cloning the [source
repository](https://github.com/gorkamunoz/rl_opts) and executing the
following on the parent folder you just cloned the repo:

``` python
pip install -e rl_opts
```

This will install both the library and the necessary packages.

### Tutorials

We have prepared a series of tutorials to guide you through the most
important functionalities of the package. You can find them in the
[Tutorials
folder](https://github.com/gorkamunoz/rl_opts/tree/master/nbs/tutorials)
of the Github repository or in the Tutorials tab of our
[webpage](https://gorkamunoz.github.io/rl_opts/), with notebooks that
will help you navigate the package as well as reproducing the results of
our paper via minimal examples. In particular, we have three tutorials:

- <a href="tutorials/tutorial_learning.ipynb" style="text-decoration:none">Reinforcement
  learning </a> : shows how to train a RL agent based on Projective
  Simulation agents to search targets in randomly distributed
  environments as the ones considered in our paper.
- <a href="tutorials/tutorial_imitation.ipynb" style="text-decoration:none">Imitation
  learning </a> : shows how to train a RL agent to imitate the policy of
  an expert equipped with a pre-trained policy. The latter is based on
  the benchmark strategies common in the literature.
- <a href="tutorials/tutorial_benchmarks.ipynb" style="text-decoration:none">Benchmarks
  </a> : shows how launch various benchmark strategies with which to
  compare the trained RL agents.

### Package structure

The package contains a set of modules for:

- <a href="lib_nbs/01_rl_framework.ipynb" style="text-decoration:none">Reinforcement
  learning framework (`rl_opts.rl_framework`)</a> : building foraging
  environments as well as the RL agents moving on them.
- <a href="lib_nbs/02_learning_and_benchmark.ipynb" style="text-decoration:none">Learning
  and benchmarking (`rl_opts.learn_and_bench`)</a> : training RL agents
  as well as benchmarking them w.r.t. to known foraging strategies.
- <a href="lib_nbs/04_imitation_learning.ipynb" style="text-decoration:none">Imitation
  learning (`rl_opts.imitation`)</a>: training RL agents in imitation
  schemes via foraging experts.
- <a href="lib_nbs/03_analytics.ipynb" style="text-decoration:none">Analytical
  functions (`rl_opts.analytics)`</a>: builiding analytical functions
  for step length distributions as well as tranforming these to foraging
  policies.
- <a href="lib_nbs/00_utils.ipynb" style="text-decoration:none">Utils
  (`rl_opts.utils)`</a>: helpers used throughout the package.

### Cite

We kindly ask you to cite our paper if any of the previous material was
useful for your work, here is the bibtex info:

``` latex
@article{munoz2023optimal,
  doi = {10.48550/ARXIV.2303.06050},  
  url = {https://arxiv.org/abs/2303.06050},  
  author = {Muñoz-Gil, Gorka and López-Incera, Andrea and Fiderer, Lukas J. and Briegel, Hans J.},  
  title = {Optimal foraging strategies can be learned and outperform Lévy walks},  
  publisher = {arXiv},  
  archivePrefix = {arXiv},
  eprint = {2303.06050},
  primaryClass = {cond-mat.stat-mech},  
  year = {2023},
}
```
