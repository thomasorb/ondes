
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

### 2. install `conda-build` tools (Ubuntu 20.04+)
```bash
conda install conda-build
```

## Install

```bash
conda create -n music python=3.7
conda install -n music numpy scipy matplotlib 
conda install -n music -c conda-forge python-sounddevice
conda install -n music -c conda-forge pysoundfile mido
```

```bash
sudo apt install build-essential
```

## Install rtmidi
On Ubuntu 20.04+:
```bash
python3-rtmidi
```

On MacOS Darwin
```bash
conda activate music
pip install python-rtmidi
```

## Compile core library


## Run

Once in the folder where ondes is installed
```bash
conda activate music # activate the music environment
python run.py # run the program
```

Program can be stopped with CTRL+C, no need to reactivate the music environment each time you run the program
