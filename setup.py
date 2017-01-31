from setuptools import setup
from setuptools import find_packages

setup(name='walkcompare',
      version='0.1.1',
      description='A way of Comparing large collections in libraries.',
      url='https://github.com/greebie/Compare',
      author='Ryan Deschamps',
      author_email='ryan.deschamps@gmail.com',
      license='Apache 2.0',
      packages=find_packages(where='walkcompare'),
      package_dir={'': 'walkcompare'},
      install_requires=[
                        'matplotlib-venn>=0.11.3',
                        'mca==1.0',
                        'pandas>=0.18.0',
                        'numpy>=1.10.4',
                        'matplotlib>=1.5.1',
                        'adjustText>=0.6.0'
                        ],
      zip_safe=False,
      include_package_data=True)
