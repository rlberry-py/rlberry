from setuptools import setup, find_packages

packages = find_packages(exclude=['docs', 'notebooks', 'logo'])

install_requires = [
    'numpy>=1.17',
    'numba',
    'PyOpenGL',
    'PyOpenGL_accelerate',
    'pygame',
    'torch>=1.6.0',
    'joblib',
    'matplotlib',
    'seaborn',
    'pandas'
]

tests_require = [
    'pytest', 
    'pytest-cov',
    'numpy',
    'numba',
    'joblib',
    'matplotlib',
    'pandas',
    'seaborn'
]

extras_require = {
    'test': tests_require,
    'deploy': ['sphinx', 'sphinx_rtd_theme']
}

setup(
    name='rlberry',
    version='0.0.1',
    description='Reinforcement Learning Library',
    url='https://github.com/rlberry-py',
    license='MIT',
    packages=packages,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    zip_safe=False,
)

