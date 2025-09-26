"""
Setup script for DeepReconXploreAi
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepreconxploreai",
    version="1.0.0",
    author="DeepReconXploreAi Team",
    author_email="contact@deepreconxploreai.com",
    description="Deep Learning Framework for Reconstruction Tasks - Xplore AI 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/priyankaa2006/DeepReconXploreAi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deeprecon-train=deeprecon.scripts.train:main",
            "deeprecon-infer=deeprecon.scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)