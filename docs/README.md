Documentation is built using sphinx.

```
pip install sphinx
pip install sphinx-rtd-theme # Theme for "Read the Docs".
pip install ghp-import # For publishing to github pages.
pip install m2r2 # For importing markdown files (i.e. README.md).
```

Build documentation:

```
sphinx-build -b html docs/source/ docs/build/html/documentation
```

Deploy to github pages:

```
cd docs && make gh-deploy
```

Additional notes:

- It's recommended to write docstrings in Google style. https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy
