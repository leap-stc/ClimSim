# Developer Guide

## Jupter Book based website

Our [website](https://climsim.github.io) is built with [Jupyter Book](https://jupyterbook.org/en/stable/intro.html)

### Local development

Create a local conda environment

```bash
cd website
conda env create -f environment.yml
```

activate the environment

```bash
conda activate climsim-docs-env
```

and run the following command to build the website locally

```bash
jupyter-book build .
```

the HTML content will be built in a folder called `_build` and can be viewed by opening the file `_build/html/index.html` in a browser.
