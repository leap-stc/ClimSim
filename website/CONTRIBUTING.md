# Contributor Guidelines

## Feedback and future development

Additional issues from community researchers wishing to build off ClimSim or make use of our tools are welcome and can be raised using the GitHub issues page (preferred) or by directly emailing ClimSim maintainers.

## Developer Guide for the website

Our [website](https://leap-stc.github.io/ClimSim/) is built with [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

### How to build the website locally

To mimic what [GitHub Actions](https://github.com/leap-stc/ClimSim/blob/main/.github/workflows/publish-website.yml) does on GitHub, first copy the README and demo notebooks to the website directory.

```bash
cd website
cp -r ../README.md ../ARCHITECTURE.md ../figures ../demo_notebooks ../evaluation .
```

Create a local conda environment

```bash
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

The HTML content will be built in a folder called `_build` and can be viewed by opening the file `_build/html/index.html` in a browser.
