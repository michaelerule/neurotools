#!/usr/bin/env bash
echo running autodoc
sphinx-apidoc -fe -o ./source .
make clean
make html


