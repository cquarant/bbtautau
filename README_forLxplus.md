First, create a virtual environment (`micromamba` is recommended):
Before running, please fork this repository to your own GitHub account. Subsequent Condor submissions require the code to stay synchronized with your fork.
For convenience, I will continue to use my own path as an example in this document.

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
## Running coffea processors

### Setup

For submitting to condor, all you need is python >= 3.7.

For running locally, follow the same virtual environment setup instructions
above and activate the environment.

```bash
micromamba activate hh
```



## Running locally

For testing, e.g.:

in bbtautau/

```bash
python src/run.py --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00 --starti 0 --endi 1 --year 2022 --processor skimmer
```

### Error

If you encounter issues with missing or incompatible Python packages, try the following: (in hh env)

```bash
pip install coffea==0.7.22
pip install 'setuptools<66.0.0’
```

### Condor jobs

A single sample / subsample:

1.Obtain grid credentials and store them as a persistent file.
Please modify the filename accordingly, based on the output of voms-proxy-info -all.

```bash
voms-proxy-init --voms cms --valid 168:00
voms-proxy-info -all
cd ..
export X509_USER_PROXY=/tmp/x509up_u154433 
cp /tmp/x509up_u154433 ～/x509up_u154433
```
Then in bbtautau:

This piece of code was, in my view, the most problematic in the entire analysis. So I used AI to write a new EOS-based workflow that generates scripts tailored for running on EOS.
Notes for the command:
subsamples: edit as needed (refer to the sample names in the data folder).
    year: edit as needed.
    git-branch / git-user: set according to the branch you actually run; ensure your working copy is synchronized with Git before execution.
    x509: this is your grid proxy file; it will be injected into the .sub file—update it as described in the previous steps.
    eos-user-path: choose the EOS path that corresponds to your user space.

```bash
python src/condor/submit_eos.py   --git-branch main   --processor skimmer   --samples HHbbtt   --subsamples VBFHHto2B2Tau_CV_1_C2V_0_C3_1   --files-per-job 5   --tag 24Nov7Signal   --year 2022   --git-user jinwang137   --x509 /afs/cern.ch/user/j/jinwa/x509up_u154433   --eos-user-path j/jinwa
```

You now have the .sub and .sh files needed to submit Condor jobs. Below is a sample submission.

```bash
condor_submit condor/skimmer/24Nov7Signal_v12_private_signal/2022_VBFHHto2B2Tau_CV_1_C2V_0_C3_1_0_eos.sub
```

If everything runs correctly, the output files will be saved under your --eos-user-path, for example: /eos/user/j/jinwa/bbtautau/skimmer.


### Error

