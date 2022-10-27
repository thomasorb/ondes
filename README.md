
## installation instructions with Anaconda (should work on Linux, Mac OSX, Windows)

### 1. download Miniconda for your OS and python 3.7.

**If you already have [Anaconda](https://www.anaconda.com/) installed go to step 2**

instructions are here: [Miniconda — Conda](https://conda.io/miniconda.html)
1. place the downloaded file on your home directory
2. install it (use the real file name instead of `Miniconda*.sh`)
```bash
bash Miniconda*.sh
```
You may have to run
```bash
conda init bash
```

### 2. install `conda-build` tools
```bash
conda install conda-build
```

## Install

```bash
conda create -n music python=3.7
conda install -n music numpy scipy matplotlib 
conda install -n music -c conda-forge python-sounddevice
conda install -n music -c conda-forge pysoundfile
```
