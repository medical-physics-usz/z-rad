from setuptools import setup, find_packages
from os import path

# Get the absolute path of the directory containing the script
working_directory = path.abspath(path.dirname(__file__))

# Read the contents of the README.md file
with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements.txt file
with open(path.join(working_directory, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# Set up the package
setup(
    name='zrad',
    version='24.09',
    url='https://github.com/radiomics-usz/z-rad',
    author='University Hospital Zurich, Department of Radiation Oncology',
    author_email='zrad@usz.ch',
    description='Z-RAD: THE SWISS POCKET KNIFE FOR RADIOMICS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,  # Use combined dependencies
)
