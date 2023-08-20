from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Larynx recognition system',
    version='0.1',
    packages=find_packages(),
    description='Code for an active radar for drone recognition',
    install_requires=required,
    python_requires='>=3.7',
)
