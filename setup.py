from setuptools import setup, find_packages

setup(
    name="asapml",
    version="0.1.0",
    author="Abhishek Thakur",
    description="AI/ML framework for digital pathology — modular and stain-robust",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
)