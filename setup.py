# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
try:
    import fusionlab
    VERSION = fusionlab.__version__
except:
    VERSION = "0.1.1"  
    
# Package metadata
DISTNAME = "fusionlab"
DESCRIPTION = "Next-Gen Temporal Fusion Architectures for Time-Series Forecasting"
LONG_DESCRIPTION = open('README.md', 'r', encoding='utf8').read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/earthai-tech/fusionlab"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation": "https://fusionlab.readthedocs.io/en/latest/api_references.html",
    "Home page": "https://fusionlab.readthedocs.io",
    "Bugs tracker": "https://github.com/earthai-tech/fusionlab/issues",
    "Installation guide": "https://fusionlab.readthedocs.io/en/latest/installation.html",
    "User guide": "https://fusionlab.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS = "time-series forecasting, machine learning, temporal fusion, deep learning"

# Ensure dependencies are installed
_required_dependencies = [
    "numpy<2",
    "pandas",
    "scipy",
    "matplotlib",
    "tqdm",
    "scikit-learn",
    "statsmodels", 
    # "torch",  # PyTorch is a core dependency in next realease
    "tensorflow",  
    # "jax",  # If using JAX for some operations
    "seaborn", 
    "joblib"
]

# Package data specification
PACKAGE_DATA = {
    'fusionlab': [
        # 'data/*.json', 
        # 'assets/*.txt'
    ],
}
setup_kwargs = {
    'entry_points': {
        'console_scripts': [
            'fusionlab=fusionlab.cli:main',
        ]
    },
    'packages': find_packages(),
    'install_requires': _required_dependencies,
    'extras_require': {
        "dev": [
            "pytest",
            "sphinx",
            "flake8",
        ]
    },
    'python_requires': '>=3.9',
}

setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
    keywords=KEYWORDS,
    package_data=PACKAGE_DATA,
    **setup_kwargs
)
