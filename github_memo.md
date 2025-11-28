git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "before eval_model"
git push origin main