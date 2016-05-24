#!/usr/bin/env bash
echo running autodoc
sphinx-apidoc -fe -o ./source .
make clean

#ISSUES
#make html 2>&1 | grep "ImportError: No module named " | awk '{ print $5 }' | sort | uniq | tee ./source/missing_modules
grep -rh --include="*.py" "^from" ../ | grep "import" | awk '{ print $2 }' | sort -d | uniq | tee ./source/missing_modules_
grep -rh --include="*.py" "^import" ../ | grep -v ',' | awk '{ print $2 }' | sort -d | uniq | tee -a ./source/missing_modules_
grep -rh --include="*.py" "^\s" ../ | grep "from" | grep "import" | awk '{ print $2 }' | sort -d | uniq | tee -a ./source/missing_modules_
cat ./source/missing_modules_ | sort -d | uniq >> ./source/missing_modules

make html


