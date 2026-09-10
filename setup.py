from setuptools import setup, find_packages

setup(
    name="neuroworld_lm",
    version="1.0.0",
    description="NeuroWorld-LM: Constant-Memory Language Modeling with Latent World State Duality",
    author="Eun-Lab, Virginia Tech",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "tokenizers>=0.19.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
)
