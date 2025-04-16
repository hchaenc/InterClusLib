from setuptools import setup, find_packages

setup(
    name = 'interClusLib',
    version = '0.1.0',
    description='A Python library for interval data clustering', 
    long_description=open('README.md').read(),
    author = 'Jiashu CHEN',
    author_email = 'jiashuchen758@gmail.com',
    license = 'MIT',
    install_requires = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
    ],
    packages = find_packages(),
)