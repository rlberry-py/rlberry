from setuptools import setup, find_packages

packages = find_packages(exclude=['docs', 'notebooks', 'logo'])

install_requires = [
    'numpy>=1.17',
    'numba',
    'pygame',
    'joblib',
    'matplotlib',
    'seaborn',
    'pandas'
]

tests_require = [
    'pytest',
    'pytest-cov',
    'numpy>=1.17',
    'numba',
    'joblib',
    'matplotlib',
    'pandas',
    'seaborn',
    'optuna',
]

full_requires = [
    'torch>=1.6.0',
    'optuna',
    'ffmpeg-python',
    'PyOpenGL',
    'PyOpenGL_accelerate',
]

extras_require = {
    'full': full_requires,
    'test': tests_require,
    'deploy': ['sphinx', 'sphinx_rtd_theme'],
    'opengl_rendering': ['PyOpenGL', 'PyOpenGL_accelerate'],
    'torch_agents': ['torch>=1.6.0'],
    'hyperparam_optimization': ['optuna'],
    'save_video': ['ffmpeg-python'],
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
