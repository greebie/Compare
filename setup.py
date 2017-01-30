from setuptools import setup
from setuptools import find_packages

setup(name='Compare',
      version='0.1',
      description='A way of Comparing large collections in libraries.',
      url='https://github.com/greebie/Compare',
      author='Ryan Deschamps',
      author_email='ryan.deschamps@gmail.com',
      license='Apache 2.0',
      packages=find_packages(where='compare'),
      package_dir={'': 'compare'},
      install_requires=[
                        'matplotlib-venn',
                        'mca',
                        'pandas',
                        'numpy',
                        'matplotlib',
                        'adjustText'
                        ],
      zip_safe=False,
      include_package_data=True)
