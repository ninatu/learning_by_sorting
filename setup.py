from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


setup(name='learning_by_sorting',
    version='0.1',
    description='Learning by Sorting',
    long_description=long_description,
    url='',
    packages=find_packages(),
    dependency_links=[],
    zip_safe=False)
