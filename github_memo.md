git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "before do task by codex"
git push origin main