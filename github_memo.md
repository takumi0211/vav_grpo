git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "before well done eval_model"
git push origin main