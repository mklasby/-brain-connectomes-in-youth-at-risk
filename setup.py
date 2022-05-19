from setuptools import setup, find_packages

setup(
    name='procan_connectome',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'requests',
        'importlib; python_version == "3.7"',
    ],
)