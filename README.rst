Z-Rad
=====

Z-Rad is an in-house implementation of radiomics software from University Hospital Zurich. Feature definition standardization according to image biomarker standardisation initiative (https://arxiv.org/abs/1612.07003).

Installation
------------

Current version of Z-Rad was tested in Python 3.11. Other versions of Python may work but were not tested.

Supported platforms: MS Windows, MacOS, and Linux.

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

Install Python 3.11 from https://www.python.org or https://www.anaconda.com/distribution/.

Installing Python packages
~~~~~~~~~~~~~~~~~~~~~~~~~~
    pip install -r requirements.txt

**On USZ PC without admin rights**

Due to access right restrictions from USZ site, a proxy needs to be set up for ``pip``. To do it, create *pip.ini* file in ``%APPDATA%\pip\pip.ini`` with the following contents::

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

Contact
-------
Please contact first your responsible research physicist from the USZ radiomics team when problems occur. Problems will be redirected by us to the whole group if they are of general interest.

- Hubert Gabryś (hubert.gabrys@usz.ch): Development
- Stephanie Tanadini-Lang (stephanie.tanadini-lang@usz.ch): Administrative
