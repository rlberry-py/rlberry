from setuptools import setup, find_packages

packages_ = find_packages()

setup(name='rlberry',
      version='0.0.1',
      description='Reinforcement Learning Library',
      url='https://github.com/rlberry-py',
      license='MIT',
      packages=packages,
      zip_safe=False)
