git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "update data/vav_prompts_harmony.csv"
git push origin main