# z-rad v1.0.

## Build Status

Tested functionality for Windows 7. 

## Known issues
- 

## Introduction

This is an in-house implementation of radiomics software from University Hospital Zurich (based on Python 2.7.14). Feature definition standardization according to [Image biomarker standardisation initiative](https://arxiv.org/abs/1612.07003).

## Installation
z-rad is OS dependent and compatible with Python 2.7. only.
With the [deprecation of Python 2.7.](https://pythonclock.org/) beginning of Jan 2020, z-rad is planning to move to Python 3.6. 

### Installation of Python 2.7.
Easiest to install Python 2.7. is via [Anaconda distribution](https://www.anaconda.com/distribution/)

### Required packages
Required packages for z-rad are the following
- python >= 2.7.14.
- numpy > 14.0.
- matplotlob >= 2.0.
- wxPython >=
- vtk-python
- python-opencv

#### with root: 
`conda install <package-name>`

#### without root: Installation on USZ Windows computer

Due to access right restrictions from USZ site, installation is possible either using compiled binary wheels or via adjustment of python's in-built package manager (pip). 
1. **wheel**: Download the corresponding python packages via the [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
and then redirect to the folder where the wheel was downloaded. Use the folling command to install
`pip install <name of the wheel.whl>`

2. pip adjustment

## Documentation

Please find the documentation in the documentation folder. 

## Contact
Please contact first your responsible research physicist from the USZ radiomics team when problems occur. Problems will be redirected by us to the whole group if they are of general interest.

- [Marta Bogowicz](marta.bogowicz@usz.ch): Distributed Learning, Head and Neck
- [Hubert Gabrys](hubert.gabrys@usz.ch): Melanoma, Lung Fibrosis, Artefacts
- [Diem Vuong](diem.vuong@usz.ch): Lung Radiomics (SAKK 16/00)
- [Stephanie Tanadini-Lang](stephanie.tanadini-lang@usz.ch): Administrative
