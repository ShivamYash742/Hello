from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="thermal-sr",
    version="1.0.0",
    author="ISRO Thermal SR Lab",
    description="Optics-Guided, Physics-Grounded Thermal Super-Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isro/thermal-sr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "thermal-sr-train=scripts.train:main",
            "thermal-sr-infer=scripts.tile_infer:main",
            "thermal-sr-eval=scripts.eval:main",
            "thermal-sr-export=scripts.export_onnx:main",
        ],
    },
)