# propagation

Simulate Twitter message propagation

See [setup.txt](setup.txt) for installation instructions, and [simulation.ipynb](simulation.ipynb) for general
information about the library.

Run `bin/run --help` for usage information.

![Tests](https://github.com/sarming/propagation/actions/workflows/tests.yml/badge.svg)

### Development

Install development environment:

```sh
conda env create -f environment-dev.yml
conda activate snsim
conda config --env --add channels conda-forge
flit install --symlink
```

Build Package in `dist/`:

```sh
flit build --setup-py
```

Update Conda environment files after changing requirements:

```sh
beni --deps=production pyproject.toml >environment.yml
beni pyproject.toml >environment-dev.yml  
beni pyproject.toml --extras test # For test environments
```

### Testing

```shell
pytest tests/ # run tests
tox # run tests for all supported Python and MPI versions
```