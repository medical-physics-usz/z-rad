Z-Rad
=====

This is an in-house implementation of radiomics software from University Hospital Zurich. Feature definition standardization according to image biomarker standardisation initiative (https://arxiv.org/abs/1612.07003).

Installation
------------

Current version of Z-Rad supports Python 3.6-3.9. All releases are tested in MS Windows, MacOS, and Linux. The legacy version of Z-Rad supporting Python 2.7 is available in **python2** branch.

Dependencies
~~~~~~~~~~~~ 

- joblib
- matplotlib
- numpy
- pandas
- pydicom
- pywavelets
- scikit-learn
- scipy
- tqdm
- vtk
- wxpython
- xlsxwriter
- pytorch
- nibabel
- simpleitk


Installing Python
~~~~~~~~~~~~~~~~~

Probably the easiest way to install Python 3 is from https://www.anaconda.com/distribution/. **Important**: Please download the 64-bit version, otherwise Z-Rad may crash when calculating large structures.

Installing Python packages
~~~~~~~~~~~~~~~~~~~~~~~~~~

**With GPU support:** ::

    conda create -n zrad python numpy scipy matplotlib pandas scikit-learn scikit-image tqdm joblib wxpython vtk pydicom pywavelets opencv nibabel pytorch simpleitk xlsxwriter cudatoolkit=11.1
    conda activate zrad
    
**Without GPU support:** ::

    conda create -n zrad python numpy scipy matplotlib pandas scikit-learn scikit-image tqdm joblib wxpython vtk pydicom pywavelets opencv nibabel pytorch simpleitk xlsxwriter
    conda activate zrad

**On USZ PC without admin rights:**

Due to access right restrictions from USZ site, installation is possible either using compiled binary wheels or via ``pip`` through a proxy.

1. **Wheel**: Download the packages from the Unofficial Windows Binaries for Python Extension Packages: https://www.lfd.uci.edu/~gohlke/pythonlibs/ and then go to the directory where the wheel was downloaded. Use the following command to install them::

    pip install <name_of_the_wheel.whl>

2. **PIP through a proxy**: Create *pip.ini* file in ``%APPDATA%\pip\pip.ini`` with the following contents::

    [global]
    proxy = http://username:password@proxy.usz.ch:8080
    trusted-host = pypi.python.org
                   pypi.org
                   files.pythonhosted.org

   This configuration file routes the network traffic related to ``pip``. Do not forget to change *username* and *password* to your credentials.



Configure HD-BET for skull stripping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source code was copied from https://github.com/MIC-DKFZ/HD-BET. If you want to use it for segmentation, you have to copy the model paremeters from ``K:\RAO_Physik\Research\1_FUNCTIONAL IMAGING\F_Software\radiomics documentation\hd-bet_params`` to ``zrad\hdbet\hd-bet_params``.

Executable
----------

Creating an executable requires PyInstaller (https://pyinstaller.readthedocs.io). To generate an executable, run ::

    python generate_executable.py
    
The executable is going to be saved in *exec/* directory. The compilation takes around 15 minutes and the resulting file size is ca. 700 MB.

Documentation
-------------

Please find the documentation in the documentation folder and on https://medical-physics-usz.github.io.

Known issues
------------

List of the known issues related to the software or hardware:

- Python environment without admin rights or IT with conda. It used to work, but somehow now it does not anymore ;)


Contact
-------
Please contact first your responsible research physicist from the USZ radiomics team when problems occur. Problems will be redirected by us to the whole group if they are of general interest.

- Marta Bogowicz (marta.bogowicz@usz.ch): Distributed Learning, Head and Neck
- Hubert Gabryś (hubert.gabrys@usz.ch): Melanoma, Lung Fibrosis, Artefacts
- Diem Vuong (diem.vuong@usz.ch): Lung Radiomics (SAKK 16/00)
- Stephanie Tanadini-Lang (stephanie.tanadini-lang@usz.ch): Administrative
