from setuptools import setup, find_packages

packages = find_packages()

install_requires = [
    'numpy>=1.17',
    'pytest',
    'numba',
    'PyOpenGL',
    'PyOpenGL_accelerate',
    'pygame'
]

setup(name='rlberry',
      version='0.0.1',
      description='Reinforcement Learning Library',
      url='https://github.com/rlberry-py',
      license='MIT',
      packages=packages,
      install_requires=install_requires,
      zip_safe=False)
