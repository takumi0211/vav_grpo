git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change to normalize"
git push origin main