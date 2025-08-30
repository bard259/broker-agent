from setuptools import setup, find_packages

setup(
    name='stock_consultant',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'praw',
        'nltk',
        'psaw',
        'pandas',
        'numpy',
    ],
    author='Your Name', # Replace with your name
    description='A simple stock consultant package',
    url='https://github.com/yourusername/stock_consultant', # Replace with your GitHub URL
)
