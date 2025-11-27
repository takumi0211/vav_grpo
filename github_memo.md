git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add q_min_ideal"
git push origin main