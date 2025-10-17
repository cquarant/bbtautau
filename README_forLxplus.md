First, create a virtual environment (`micromamba` is recommended):

```bash
# Clone the repository
git clone --recursive https://github.com/jinwang137/bbtautau.git
cd bbtautau
# Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
# Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
micromamba env create -f environment.yaml
# If it’s convenient, it would be preferable to install the hh environment on EOS, as it’s quite large.
micromamba activate hh


### Installing package

**Remember to install this in your mamba environment**.

```bash
# Clone the repsitory as above if you haven't already
# Perform an editable installation
pip install -e .
# for committing to the repository
pip install pre-commit
pre-commit install
# Install as well the common HH utilities
cd boostedhh
pip install -e .
cd ..
```