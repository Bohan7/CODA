"""Download the released CODA checkpoints from the Hugging Face Hub.

    python scripts/download_checkpoints.py        # -> generation/checkpoints/<split>/{score,classifier}.pth
"""
import argparse
import glob
import os

from huggingface_hub import snapshot_download

REPO_ID = 'BobbbWang/CODA'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=REPO_ID, help='Hugging Face model repository')
    parser.add_argument('--out', default='generation', help='directory that receives checkpoints/')
    args = parser.parse_args()

    snapshot_download(repo_id=args.repo, allow_patterns=['checkpoints/*'], local_dir=args.out)
    target = os.path.join(args.out, 'checkpoints')
    n = sum(os.path.isfile(p) for p in glob.glob(os.path.join(target, '**'), recursive=True))
    if n == 0:
        raise SystemExit(f'{args.repo} holds no files under checkpoints/')
    print(f'{n} files in {target}')
