#!/usr/bin/env bash


# It seems to be a litte tricky to get the Sphinx automatic documentation
# to host as a web page on git pages.
#
# What we will do intead is manually synchronize the documentation
# builds with the branches of these projects used for the web pages
#
# Run this script from it's local directory
#

shopt -s extglob

#!/usr/bin/env bash
## echo running autodoc
## sphinx-apidoc -fe -o ./source .
## make clean
#ISSUES
#make html 2>&1 | grep "ImportError: No module named " | awk '{ print $5 }' | sort | uniq | tee ./source/missing_modules
## grep -rh --include="*.py" "^from" ../ | grep "import" | awk '{ print $2 }' | sort -d | uniq | tee -a ./source/missing_modules_
## grep -rh --include="*.py" "^import" ../ | grep -v ',' | awk '{ print $2 }' | sort -d | uniq | tee -a ./source/missing_modules_
## grep -rh --include="*.py" "^\s" ../ | grep "from" | grep "import" | awk '{ print $2 }' | sort -d | uniq | tee -a ./source/missing_modules_
## cat ./source/missing_modules_ | sort -d | uniq >> ./source/missing_modules
## make clean html

# hopefully removes everything that shouldn't be being tracked as per
# .gitignore. Possibly dangerous. 
cat .gitignore | awk "/^[.\*]/" | sed 's/"/"\\""/g;s/.*/"&"/' |  xargs -E '' -I{} git rm -rf --cached {}
git rm -rf --cached *.pyc
git add . 
git add -u :/
git commit -m "$1"
git push origin master
