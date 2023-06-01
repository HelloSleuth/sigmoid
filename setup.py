from setuptools import setup

setup(
    name='Sigmoidal',
    version='0.3.0',
    author='Jade Glaze, Sleuth',
    author_email='jade@hellosleuth.com',
    url='https://github.com/HelloSleuth/sigmoid',
    license='MIT',
    description='Sigmoidal is a small library to allow you to fit and evaluate sigmoid functions in a way that works like the Numpy Polynomial class.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'numpy>=1.24.3',
        'scipy>=1.10.1',
    ],
    packages=[
        'sigmoidal',
    ],
)
