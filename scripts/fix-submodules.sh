#!/bin/bash
# fix-submodules.sh — update submodules whose pinned commit was lost by upstream force-push
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

for sub in $(git config --file .gitmodules --get-regexp path | awk '{print $2}'); do
    pinned=$(git ls-tree HEAD "$sub" | awk '{print $3}')
    cd "$sub"
    if ! git branch -r --contains "$pinned" 2>/dev/null | grep -q .; then
        echo "fixing $sub — commit $pinned lost, updating to latest"
        cd ..
        git submodule update --remote "$sub"
        git add "$sub"
        git commit -m "fix: update submodule $sub (upstream force-pushed)"
        continue
    fi
    cd ..
done
