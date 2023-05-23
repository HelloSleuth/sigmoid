# Sigmoidal 

Sigmoidal is intended to work like the [Numpy Polynomial](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html) class where it makes sense. Specifically it supports:
* Using the sigmoid function in a very natural way like `y = sig(x)` including when x is a numpy array. (Once you've created a Sigmoid instance `sig = Sigmoid(...)`)
* Fitting a Sigmoid to data just like Polynomial with `Sigmoid.fit(x, y)`.
* Taking the first or second derivative with `deriv()`.
* Finding the roots of the sigmoid or it's first or second derivitive with `.roots()`.
* The convenience method `.linspace()` which can get you an array of dependent values with only the range of independent values.
* `.copy()`
* Operations `==`, `!=`, `str()`, `repr()`

## Running Tests
* `python -m unittest discover tests`

## Setup for Deployment
* `pip install twine`

## Building the Package
* `python setup.py sdist bdist_wheel`

## Deploying
* `twine upload --skip-existing --repository-url https://upload.pypi.org/legacy/ dist/*`
