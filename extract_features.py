from pydriller import Repository
import pandas as pd

repo_path = "../commons-lang"

data = []

print("Collecting first 2000 commits...")

for i, commit in enumerate(Repository(repo_path).traverse_commits()):
    if i >= 2000:
        break

    data.append({
        "hash": commit.hash,
        "msg_len": len(commit.msg),
        "files_changed": len(commit.modified_files),
        "insertions": commit.insertions,
        "deletions": commit.deletions
    })

df = pd.DataFrame(data)
df.to_csv("commit_data.csv", index=False)

print("Done. Total commits saved:", len(df))