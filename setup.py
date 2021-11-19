from setuptools import setup, find_packages

packages = find_packages(exclude=['docs', 'notebooks', 'assets'])

#
# Base installation (interface only)
#
install_requires = [
    'numpy>=1.17',
    'pygame',
    'matplotlib',
    'seaborn',
    'pandas',
    'gym',
    'dill',
    'docopt',
    'pyyaml',
]

#
# Extras
#

# default installation
default_requires = [
    'numba',
    'optuna',
    'ffmpeg-python',
    'PyOpenGL',
    'PyOpenGL_accelerate',
    'pyvirtualdisplay',
]

# tensorboard must be installed manually, due to conflicts with
# dm-reverb-nightly[tensorflow] in jax_agents_requires
torch_agents_requires = default_requires + [
    'torch>=1.6.0',
    # 'tensorboard'
]

jax_agents_requires = default_requires + [
    'jax[cpu]',
    'chex',
    'dm-haiku',
    'optax',
    'dm-reverb[tensorflow]==0.5.0',
    'dm-tree',
    'rlax'
]

extras_require = {
    'default': default_requires,
    'jax_agents': jax_agents_requires,
    'torch_agents': torch_agents_requires,
    'deploy': ['sphinx', 'sphinx_rtd_theme'],
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rlberry',
    version='0.2.1',
    description='An easy-to-use reinforcement learning library for research and education',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Omar Darwiche Domingues, Yannis Flet-Berliac, Edouard Leurent, Pierre Menard, Xuedong Shang',
    url='https://github.com/rlberry-py',
    license='MIT',
    packages=packages,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
)
