from setuptools import setup, find_packages

setup(
    name="narecli",
    version="0.3.5",
    description="Neuro-Adaptive Reasoning Engine (NARE) - A Skill-Based Cognitive Architecture",
    author="NARE Contributors",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu",
        "numpy",
        "python-dotenv",
        "requests"
    ],
    entry_points={
        "console_scripts": [
            "nare=nare.cli:main",
        ],
    },
)
