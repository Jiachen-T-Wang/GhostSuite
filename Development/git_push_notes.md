# Switch to your public-release branch
```
git checkout public
```

# Merge changes from your main development branch
```
git merge master
```

# Remove any development-related files
```
git rm any-new-sensitive-files
git commit -m "Update public release"
```

# Push to public repo's master branch
git push public public:master