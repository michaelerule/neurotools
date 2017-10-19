#!/usr/bin/env bash

shopt -s extglob

echo running autodoc
cd ./docs
sphinx-apidoc -fe -o . ../
make clean html
cd ../


