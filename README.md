# biphasic-kinematic-growth

In this repository a biphasic model in the framework of the Theory of Porous Media<sup>1</sup> is combined with a kinematic description according to Rodriguez et al.<sup>2</sup>.
It is the code base for the investigations presented in Suditsch et al.<sup>3</sup>. The script "comparision_Growth_Formulations.py" produces all used 
data and with "create_plots.ipynb" all plots are created. In "TPM_2Phase_MAo_LMo_MAs_Growth.py" and "Rodriguez1994.py" calculation routines for the respective 
theories are written. For convinience, general functions are outsourced into the "helper.py" file. In this README, a rough summary of the basic 
equations of the coupled kinetic-multiphasic growth model are shown. It is followed by an installation guide. For deeper insights into the theory 
and the motivation the reader is referred to the mentioned papers.

# Basic Theory

A biphasic growth model is modified about a split of the deformation gradient $\textbf{F}$ into an elastic and a growth part

$$ \textbf{F} = \textbf{F}_e \textbf{F}_g\ .$$

With this assumption of a three dimensional volumetric growth $\textbf{F}_g$ simplifies to

$$ \textbf{F}_g = \mathrm{J}_g^{1/3}\ \mathbf{I}$$

with the determinant of the deformation gradient for growth $\mathrm{J}_g$ and the identity $\mathbf{I}$. The authors propose an assumption
for the constitutive relation of the growth Jacobian to

$$ \mathrm{J}_g = \exp [\alpha_g\ \hat{n}^S\ (1-\frac{\rho^{SR}}{\rho^{FR}})] \quad \forall\ \alpha_g \in 0,...,1\ .$$

The $\alpha_g$ activates the stressless and irreversible growth. Following the Theory of Porous Media, the system of equation consists of the linear 
momentum balance of the total aggregate, the volume balance of the total aggregate and the volume balance of the solid constituent with

$$\textrm{div}\ \mathbf{T} = \mathbf{0}$$

$$\textrm{div}\ \overset{'}{\textbf{x}}_S + \textrm{div} (n^F \mathbf{w}_F) = 0$$

$$(n^S)^{'}_S + n^S \mathrm{div}\ \overset{'}{\textbf{x}}_S - \hat{n}^S = 0$$

where $\textrm{div}$ is the divergence operator, $\mathbf{T}$ is the Cauchy stress, $\overset{'}{\textbf{x}}_S$ is the velocity of solid body,
$n^F$ is the volume fraction of the fluid. The seepage velocity $\mathbf{w}_F$ is a referential velocity to the solid body and therefore, underlines
the Lagrange-Eulerian setup. A supply term of the volume fraction of the solid $\hat{n}^S$ is indicated by the hat. To close the mechanical problem
constitutive equations are assumed to be neo Hookean for the stress via

$$\mathbf{T} = \mu^S \mathbf{B}_e + \lambda^S \ln \mathrm{J}_e - \mathrm{p} \mathbf{I}$$

and Darcy-like for the seepage velocity by

$$n^F \mathbf{w}_F = - \mathrm{k}_F\ \mathrm{grad}\ \mathrm{p}.$$

Herein, $\lambda^S$ and $\mu^S$ are the Lam\'e parameters, $\mathbf{B}_e = \textbf{F}_e \textbf{F}_e^{\mathrm{T}}$ is the right elastic Cauchy-Green strain,
$\mathrm{p}$ is the fluid pressure and $\mathrm{k}_F$ is the Darcy permeability. These equations are formulated into their strong form and solved with the 
Finite Element framework of FEniCS<sup>4</sup>.

# Quick Installation and Run

The presented code can simply be run in an [Anaconda environment](https://anaconda.org/), which therefore needs to be installed. With the following command
all necessary libraries will be installed.
```
conda create -n biphasic-kinematic-growth -c conda-forge fenics meshio matplotlib jupyterlab gmsh
```
This way not always lead to success and an alternative workaround is to set up an environment just with the FEniCS library and install all other one by one.
```
conda create -n biphasic-kinematic-growth -c conda-forge fenics
conda install -c conda-forge meshio
conda install -c conda-forge matplotlib
conda install -c conda-forge jupyterlab
conda install -c conda-forge gmsh
```
To activate the environment run
```
conda activate biphasic-kinematic-growth
```
The simulations are run with
```
python3 comparision_Growth_Formulations.py
```

# Licence

The repository is released under GPL 3.0


# How to cite

If you are using code from this repository please also cite the related publication

**Growth in biphasic tissue**. Marlon Suditsch, Franziska S. Egli, Lena Lambers, Tim Ricken, International Journal of Engineering Science (2025). [10.1016/j.ijengsci.2024.104183](https://doi.org/10.1016/j.ijengsci.2024.104183) 
```bib
@article{SUDITSCH2025104183,
title = {Growth in biphasic tissue},
journal = {International Journal of Engineering Science},
volume = {208},
pages = {104183},
year = {2025},
issn = {0020-7225},
doi = {https://doi.org/10.1016/j.ijengsci.2024.104183},
url = {https://www.sciencedirect.com/science/article/pii/S0020722524001678},
author = {Marlon Suditsch and Franziska S. Egli and Lena Lambers and Tim Ricken},
}
```

# Literature

<sup>1</sup> Ehlers, Foundations of multiphasic and porous materials. In: Porous Media. Springer, 2002

<sup>2</sup> Rodriguez, Hoger, McCulloch, Stress-dependent finite growth in soft elastic tissues. Journal of Biomechanics, 1994

<sup>3</sup> Suditsch, Egli, Lambers, Wagner, Ricken, Growth in biphasic tissue, Journal of Engineering Science, 2025

<sup>4</sup> Anders Logg et al., Automated Solution of Differential Equations by the Finite Element Method, Springer Berlin Heidelberg, 2012, DOI: 10.1007/978-3-642-23099-8


# About

The repository is written by Marlon Suditsch
