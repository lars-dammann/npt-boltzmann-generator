from setuptools import setup, find_packages

setup(
    name='npt-fff',
    description='NpT Free-form flows',
    version='0.1dev',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'torch', 'lightning', 'wandb', 'ray[tune,train], flatdict'],
    license='MIT',
    author='Lars Dammann',
    author_email='lars.dammann@tuhh.de',
)
