git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "正規化"
git push origin main