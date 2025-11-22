git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "well done setting and normalization with codex"
git push origin main