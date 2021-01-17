from setuptools import setup, find_packages

packages = find_packages(exclude=['docs', 'notebooks', 'assets'])

install_requires = [
    'numpy>=1.17',
    'pygame',
    'joblib',
    'matplotlib',
    'seaborn',
    'pandas',
    'gym',
    'dill',
    'docopt',
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
    'pyvirtualdisplay',
    'gym',
]

full_requires = [
    'numba',
    'torch>=1.6.0',
    'tensorboard',
    'optuna',
    'ffmpeg-python',
    'PyOpenGL',
    'PyOpenGL_accelerate',
    'pyvirtualdisplay',
    'sacred',
]

extras_require = {
    'full': full_requires,
    'test': tests_require,
    'deploy': ['sphinx', 'sphinx_rtd_theme'],
    'opengl_rendering': ['PyOpenGL', 'PyOpenGL_accelerate'],
    'torch_agents': ['torch>=1.6.0', 'tensorboard'],
    'hyperparam_optimization': ['optuna'],
    'save_video': ['ffmpeg-python'],
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rlberry',
    version='0.0.3.1',
    description='An easy-to-use reinforcement learning library for research and education',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Omar Darwiche Domingues, Yannis Flet-Berliac, Edouard Leurent, Pierre Menard, Xuedong Shang',
    url='https://github.com/rlberry-py',
    license='MIT',
    packages=packages,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    zip_safe=False,
)
