# Publish or update this project on GitHub

1. Create an empty repository on GitHub (for example, `medhop-agentic-rag`) or use an existing repository. Do not initialize an existing project with an unrelated history.
2. On AutoDL, unpack this project and enter its directory.
3. Confirm what will be committed: `git status` and `git diff --cached --stat`.

## For a new, empty repository

```bash
git init
git branch -M main
git add .gitignore README.md GITHUB_UPLOAD.md medhop_pipeline.py run_full_experiment.sh requirements.txt demo.json
git commit -m "Add reproducible MedHop retrieval baselines"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## For an existing repository

**Recommended:** first clone your repository, then copy these project files into it so you preserve existing history. Review the diff before committing. Never use `git push --force` unless you specifically intend to replace the remote history.

```bash
git status
git add .gitignore README.md GITHUB_UPLOAD.md medhop_pipeline.py run_full_experiment.sh requirements.txt demo.json
git commit -m "Update MedHop baseline experiment pipeline"
git push origin HEAD
```

GitHub authentication requires your own configured SSH key, Git Credential Manager or suitable access token. Do not paste personal access tokens into chat or commit them to the repo.
