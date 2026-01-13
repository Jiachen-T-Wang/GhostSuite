```
git clone --no-local /scratch/gpfs/PMITTAL/tianhao/GhostSuite GhostSuite-public-v0.33
cd GhostSuite-public-v0.33
git filter-repo --path codereview --path tests --path AGENTS.md --invert-paths
git remote add public https://github.com/Jiachen-T-Wang/GhostSuite.git
git push -f public v0.33:v0.33
```