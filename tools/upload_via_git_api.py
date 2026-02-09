import os
import sys
import base64
import json
import requests
from pathlib import Path

ROOT = Path("g:/Moments2")

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".trae", ".vscode", ".idea", ".venv", "venv", "env",
    "pretrained_models", "static/uploads", "static/images", "示例html"
}
EXCLUDE_FILES = {
    "energy_log.json", "moments_store.json", "profile_store.json",
    "word_cards_store.json", "config.ini"
}
EXCLUDE_SUFFIXES = {".onnx", ".zip"}

def should_exclude(root: Path, p: Path) -> bool:
    rel = p.relative_to(root).as_posix()
    if any(part in EXCLUDE_DIRS for part in rel.split("/")):
        return True
    if p.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if p.name.lower() in EXCLUDE_FILES:
        return True
    return False

def collect_files(root: Path):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            full = Path(dirpath) / fn
            if should_exclude(root, full):
                continue
            files.append(full)
    return files

def api(headers, method, url, **kwargs):
    r = requests.request(method, url, headers=headers, **kwargs)
    return r

def ensure_repo(headers, login, repo, private: bool):
    r = api(headers, "GET", f"https://api.github.com/repos/{login}/{repo}")
    if r.status_code == 404:
        cr = api(headers, "POST", "https://api.github.com/user/repos",
                 json={"name": repo, "private": private})
        if cr.status_code not in (201, 202):
            raise RuntimeError(f"create repo failed: {cr.status_code} {cr.text}")
    elif r.status_code != 200:
        raise RuntimeError(f"get repo failed: {r.status_code} {r.text}")

def get_head_commit(headers, login, repo, branch):
    r = api(headers, "GET", f"https://api.github.com/repos/{login}/{repo}/git/ref/heads/{branch}")
    if r.status_code == 200:
        ref = r.json()
        commit_sha = ref["object"]["sha"]
        cr = api(headers, "GET", f"https://api.github.com/repos/{login}/{repo}/git/commits/{commit_sha}")
        if cr.status_code != 200:
            raise RuntimeError(f"get commit failed: {cr.status_code} {cr.text}")
        tree_sha = cr.json()["tree"]["sha"]
        return commit_sha, tree_sha
    elif r.status_code == 404:
        return None, None
    else:
        raise RuntimeError(f"get ref failed: {r.status_code} {r.text}")

def create_blobs(headers, login, repo, files):
    blob_map = {}
    for fp in files:
        content = fp.read_bytes()
        b64 = base64.b64encode(content).decode("ascii")
        br = api(headers, "POST",
                 f"https://api.github.com/repos/{login}/{repo}/git/blobs",
                 json={"content": b64, "encoding": "base64"})
        if br.status_code != 201:
            raise RuntimeError(f"blob {fp} failed: {br.status_code} {br.text[:200]}")
        blob_map[fp] = br.json()["sha"]
    return blob_map

def create_tree(headers, login, repo, root, blob_map, base_tree_sha=None):
    tree_items = []
    for fp, sha in blob_map.items():
        rel = fp.relative_to(root).as_posix()
        tree_items.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
    payload = {"tree": tree_items}
    if base_tree_sha:
        payload["base_tree"] = base_tree_sha
    tr = api(headers, "POST",
             f"https://api.github.com/repos/{login}/{repo}/git/trees",
             json=payload)
    if tr.status_code != 201:
        raise RuntimeError(f"create tree failed: {tr.status_code} {tr.text[:200]}")
    return tr.json()["sha"]

def create_commit(headers, login, repo, message, tree_sha, parent_sha=None):
    payload = {"message": message, "tree": tree_sha}
    if parent_sha:
        payload["parents"] = [parent_sha]
    cr = api(headers, "POST",
             f"https://api.github.com/repos/{login}/{repo}/git/commits",
             json=payload)
    if cr.status_code != 201:
        raise RuntimeError(f"create commit failed: {cr.status_code} {cr.text[:200]}")
    return cr.json()["sha"]

def update_or_create_ref(headers, login, repo, branch, commit_sha, create=False):
    if create:
        rr = api(headers, "POST",
                 f"https://api.github.com/repos/{login}/{repo}/git/refs",
                 json={"ref": f"refs/heads/{branch}", "sha": commit_sha})
        if rr.status_code != 201:
            raise RuntimeError(f"create ref failed: {rr.status_code} {rr.text[:200]}")
    else:
        rr = api(headers, "PATCH",
                 f"https://api.github.com/repos/{login}/{repo}/git/refs/heads/{branch}",
                 json={"sha": commit_sha, "force": True})
        if rr.status_code != 200:
            raise RuntimeError(f"update ref failed: {rr.status_code} {rr.text[:200]}")

def main():
    token = None
    repo = "Moments"
    visibility = "private"
    branch = "main"
    # parse args
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--token":
            i += 1; token = argv[i]
        elif a == "--repo":
            i += 1; repo = argv[i]
        elif a == "--visibility":
            i += 1; visibility = argv[i]
        elif a == "--branch":
            i += 1; branch = argv[i]
        i += 1
    if not token:
        token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: no token")
        sys.exit(1)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "TraeUpload"
    }
    # get user
    u = requests.get("https://api.github.com/user", headers=headers)
    if u.status_code != 200:
        print("error: get user failed", u.status_code, u.text[:200]); sys.exit(1)
    login = u.json()["login"]
    # ensure repo exists
    ensure_repo(headers, login, repo, private=(visibility == "private"))
    # get current head
    parent_sha, base_tree = get_head_commit(headers, login, repo, branch)
    files = collect_files(ROOT)
    blob_map = create_blobs(headers, login, repo, files)
    tree_sha = create_tree(headers, login, repo, ROOT, blob_map, base_tree_sha=base_tree)
    commit_sha = create_commit(headers, login, repo, "Initial import", tree_sha, parent_sha=parent_sha)
    update_or_create_ref(headers, login, repo, branch, commit_sha, create=(parent_sha is None))
    print(f"https://github.com/{login}/{repo}")

if __name__ == "__main__":
    main()
