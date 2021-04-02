This repo contains a collection of tools for assessing the performance of 
pose estimation algorithms, in both single-view and multi-view setups.

## Installation

First you'll have to install the `git` package in order to access the code on github. 
Follow the directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your specific OS.
Then, in the command line, navigate to where you'd like to install the package and move into that directory:
```
$: git clone https://github.com/paninski-lab/tracking-diagnostics
$: cd tracking-diagnostics
```

### Running in an existing conda environment
The dependencies for this package are relatively minimal, and quite standard - 
numpy, scipy, matplotlib, etc. It may be possible to run this code in an already
existing environment, with few additional package installations. Please stay 
tuned for more updates.

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `tracking-diagnostics` directory after you have
activated an existing conda environment:

```
(existing_environment) $: pip install -e .
```

### Creating a new conda environment
To create a new environment specifically for this package, follow the directions 
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 
to install the `conda` package for managing development environments. 
Then, create a conda environment:

```
$: conda create --name=diagnostics python=3.7
$: conda activate diagnostics
(diagnostics) $: pip install -r requirements.txt 
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `tracking-diagnostics` directory:

```
(diagnostics) $: pip install -e .
```

To be able to use this environment for jupyter notebooks:

```
(diagnostics) $: python -m ipykernel install --user --name diagnostics
```

