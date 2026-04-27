from setuptools import setup, find_packages

setup(
    name="nare",
    version="1.0.0",
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
