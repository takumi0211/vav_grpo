git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "fix error train_support/step_stream.py"
git push origin main