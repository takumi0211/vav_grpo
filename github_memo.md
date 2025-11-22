git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "fix error train_support/data_reward.py"
git push origin main