git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change train_support/td3_reward.py"
git push origin main