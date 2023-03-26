Generalised Random Forest Methods
=================================



[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Krish26-2000/generalised_random_forest_methods/main.svg)](https://results.pre-commit.ci/latest/github/Krish26-2000/generalised_random_forest_methods/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This project is an attempt to use Machine learning methods like Generalized Random Forest for asserting the hypothesis that
    "after the removal of ban on same-sex marriages in the USA in the span of 2004--2015, the same-sex married couples have increased".
    The extention of this study is to understand whether such couples are encouraged to start a family.
    The motivation behind this
    study is a research paper by _Susan Athey, Julie Tibshirani and Stefan Wager_ on [Generalized Random Forest](https://arxiv.org/pdf/1610.01271)
    I have used the methods discussed in the paper and tried to implement the methods on a different dataset.
    The aim of both the studies is to study the heterogeneous treatment effects on the dependent variable.
    I have used the [EconML](https://github.com/py-why/econml) package by Microsoft to implement Causal Forest Double Machine Learning method on the dataset.
    With this approach, I calculate the treatment effects on the outcome and ultimately get the conditional average treatment effect.

## Usage

To get started, create and activate the environment with

    $ conda/mamba env create --file=environment.yml
    $ conda activate gen_rfm_2

Now you can build the project using

    $ pytask

For the final paper pdf to generate, please make sure that pytask-latex and all the latex related packages 
are properly installed.

### Testing 

Testing functions are located in tests folder. These functions are written to test and assert the original 
functions on a small representative data set.

Run the test file using

    $ pytest

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
