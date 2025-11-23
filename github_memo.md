git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "update step.md"
git push origin main