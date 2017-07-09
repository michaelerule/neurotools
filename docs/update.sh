#!/usr/bin/env bash

echo "This will overwrite any customization of .rst files!"

# https://stackoverflow.com/questions/1885525/how-do-i-prompt-a-user-for-confirmation-in-bash-script
read -p "Are you sure? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # do dangerous stuff
    sphinx-apidoc -fe -o . ../
    #make clean
    #make html
    #make SPHINXOPTS='-W' clean html
    make clean html
fi
