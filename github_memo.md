git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "delete old dataset"
git push origin main