from setuptools import setup

setup(
    name='sigmoid',
    version='0.1.0',
    author='Jade Glaze',
    author_email='jade@hellosleuth.com',
    url='https://github.com/HelloSleuth/sigmoid',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.24.3',
        'scipy>=1.10.1',
    ],
    packages=[
        'sigmoid',
    ],
)
