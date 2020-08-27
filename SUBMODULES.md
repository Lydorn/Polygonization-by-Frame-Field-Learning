This project uses various git submodules. You have to pull all for the code to work.

See this tutorial on git submodules used with Python modules in dev mode: https://shunsvineyard.info/2019/12/23/using-git-submodule-and-develop-mode-to-manage-python-projects/

Further useful git submodules commands:

Clone a repository including its submodules:
```
git clone --recursive --jobs 8 <URL to Git repo>
```

If you already have cloned a repository and now want to load it’s submodules:
```
git submodule update --init --recursive --jobs 8
OR
git submodule update --recursive
```

Pull everything, including submodules:
```
git pull --recurse-submodules
```

Add a sudmodule:
```
git submodule add -b <branch_name> <URL to Git repo>
git submodule init
```

Update your submodule --remote fetches new commits in the submodules and updates the working tree to the commit described by the branch:
```
git submodule update --remote
```

The following example shows how to update a submodule to its latest commit in its master branch:
```
# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
```

Get the update by pulling in the changes and running the submodules update command:
```
# another developer wants to get the changes
git pull

# this updates the submodule to the latest
# commit in master as set in the last example
git submodule update
```

Remove submodule:
```
git rm the_submodule
rm -rf .git/modules/the_submodule
```
