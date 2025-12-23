from setuptools import find_packages, setup

setup(
    name='ai-ml-linear-regression',
    version='0.1.0',
    author='Kiattipong Kamonrat',
    author_email='kiattipong.kamonrat@gmail.com',
    description='A project for linear regression analysis with simulated data.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
