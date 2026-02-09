import os, sys, base64, json, requests
from pathlib import Path

def should_exclude(root, path):
    p = Path(path)
    rel = p.relative_to(root).as_posix()
    parts = rel.split('/')
    if parts[0] in {'.git', '__pycache__', '.trae', '.vscode', '.idea', '.venv', 'venv', 'env'}:
        return True
    if rel.startswith('pretrained_models/'):
        return True
    if rel.startswith('static/uploads/'):
        return True
    if p.suffix.lower() in {'.onnx', '.zip'}:
        return True
    bn = p.name.lower()
    if bn in {'energy_log.json','moments_store.json','profile_store.json','word_cards_store.json','config.ini'}:
        return True
    return False

def collect_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {'.git','__pycache__','.trae','.vscode','.idea','.venv','venv','env','pretrained_models','static/uploads'}]
        for fn in filenames:
            full = Path(dirpath) / fn
            if should_exclude(root, full):
                continue
            files.append(full)
    return files

def main():
    token = None
    repo = 'Moments'
    visibility = 'private'
    for i, a in enumerate(sys.argv[1:]):
        if a == '--token':
            token = sys.argv[1+ i + 1]
        elif a == '--repo':
            repo = sys.argv[1+ i + 1]
        elif a == '--visibility':
            visibility = sys.argv[1+ i + 1]
    if not token:
        token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print('error: no token')
        sys.exit(1)
    root = Path('g:/Moments2')
    headers = {'Authorization': f'Bearer {token}','Accept':'application/vnd.github+json','User-Agent':'TraeUpload'}
    u = requests.get('https://api.github.com/user', headers=headers)
    if u.status_code != 200:
        print('error: get user failed', u.status_code, u.text[:200])
        sys.exit(1)
    login = u.json()['login']
    r = requests.get(f'https://api.github.com/repos/{login}/{repo}', headers=headers)
    if r.status_code == 404:
        cr = requests.post('https://api.github.com/user/repos', headers=headers, json={'name': repo, 'private': visibility=='private'})
        if cr.status_code not in (201,202):
            print('error: create repo failed', cr.status_code, cr.text[:200])
            sys.exit(1)
    elif r.status_code != 200:
        print('error: get repo failed', r.status_code, r.text[:200])
        sys.exit(1)
    files = collect_files(root)
    branch = 'main'
    for fp in files:
        rel = fp.relative_to(root).as_posix()
        data = fp.read_bytes()
        b64 = base64.b64encode(data).decode('ascii')
        url = f'https://api.github.com/repos/{login}/{repo}/contents/{rel}'
        payload = {'message': f'Add {rel}','content': b64,'branch': branch}
        ex = requests.get(url, headers=headers, params={'ref': branch})
        if ex.status_code == 200:
            js = ex.json()
            sha = js.get('sha')
            if sha:
                payload['sha'] = sha
                payload['message'] = f'Update {rel}'
        put = requests.put(url, headers=headers, data=json.dumps(payload))
        if put.status_code not in (201,200):
            print('error: put', rel, put.status_code, put.text[:200])
            sys.exit(1)
    print(f'https://github.com/{login}/{repo}')

if __name__ == '__main__':
    main()
