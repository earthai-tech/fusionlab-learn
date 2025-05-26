# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
try:
    import fusionlab
    VERSION = fusionlab.__version__
except:
    VERSION = "0.2.1" 

# Package metadata
DISTNAME = "fusionlab-learn"
DESCRIPTION = "Next-Gen Temporal Fusion Architectures for Time-Series Forecasting"
LONG_DESCRIPTION = open('README.md', 'r', encoding='utf8').read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/earthai-tech/fusionlab-learn"
LICENSE = "BSD-3-Clause" 
PROJECT_URLS = {
    "API Documentation": "https://fusion-lab.readthedocs.io/en/latest/api.html",
    "Home page": "https://fusion-lab.readthedocs.io",
    "Bugs tracker": "https://github.com/earthai-tech/fusionlab-learn/issues",
    "Installation guide": "https://fusion-lab.readthedocs.io/en/latest/installation.html",
    "User guide": "https://fusion-lab.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS = "time-series forecasting, machine learning, temporal fusion, deep learning"

# Core dependencies
_required_dependencies = [
    "numpy<2", 
    "pandas",
    "scipy",
    "matplotlib",
    "tqdm",
    "scikit-learn",
    "statsmodels",
    "tensorflow>=2.15", 
    "joblib",
    "pyyaml"
]

# Optional dependencies
_extras_require = {
    "dev": [
        "pytest",
        "sphinx",
        "flake8",
        # other dev tools like black, isort, mypy can be added
    ],
    "k-diagram": [ 
        "k-diagram>=1.0.3", 
    ],
    "full": [ 
        "k-diagram>=1.0.3",
    ]
}
# "full" to include all optional dependencies by default if desired
# or keep them separate. Let's make "full" install kdiagram.
_extras_require["full"] = list(
    set(dep for group in _extras_require.values() for dep in group)
    )

# Package data specification
PACKAGE_DATA = {
    'fusionlab': [
        'datasets/data/*.csv', 
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
    'packages': find_packages(exclude=['docs', 'tests', 'examples']), 
    'install_requires': _required_dependencies,
    'extras_require': _extras_require, 
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
    ],
    keywords=KEYWORDS,
    package_data=PACKAGE_DATA,
    **setup_kwargs
)