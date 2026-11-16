"""Offline Kaggle zipapp for the previously measured shared-K4 TACS-X run.

Includes original checkpoints, DINOv3 source/weights, CIFAR files, exact warm
start, and code. Evaluate fixed models or retrain both arms from the warm start.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/tacs_shared_backbone_k4_v1"

LAUNCHER = r'''"""Offline shared-K4 TACS-X: default validation evaluation; optional matched retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/tacsx_shared_k4'))
    p.add_argument('--mode',choices=['evaluate','train'],default='evaluate')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    root=(args.output/'project').resolve()
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(sys.argv[0]) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name,expected in manifest['files_sha256'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Invalid bundle path')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:
                raise ValueError('Corrupt input '+name)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Existing input changed; choose a fresh --output: '+name)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
    print('TACS-X shared trainable backbone, K4, CLEAN training and Gaussian-NOISE evaluation. Not a clean-accuracy win.',flush=True)
    print('Reference result is exploratory, one seed, previously used test; fresh GPU training may differ.',flush=True)
    subprocess.run([sys.executable,str(root/'scripts/run_shared_k4.py'),'--mode',args.mode,
                    '--split',args.split,'--workers',str(args.workers)],cwd=root,check=True)

if __name__=='__main__':
    main()
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/kaggle/tacsx_shared_k4.pyz')
    args=p.parse_args()
    files={}
    for folder in ('src','scripts'):
        files.update({p.relative_to(ROOT).as_posix():p for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts})
    files['scripts/train_tacs_shared_k4_snapshot.py']=RUN/'execution/train_tacs_shared_k4_snapshot.py'
    for p in (ROOT/'third_party/dinov3/dinov3').rglob('*.py'):
        files[p.relative_to(ROOT).as_posix()]=p
    for name in ('LICENSE','LICENSE.txt','README.md'):
        p=ROOT/'third_party/dinov3'/name
        if p.exists():
            files[p.relative_to(ROOT).as_posix()]=p
    for name in ('train','test'):
        files[f'data/cifar-100-python/{name}']=ROOT/'data/cifar-100-python'/name
    weight='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    files[weight]=ROOT/weight
    for name in ('plan.json','train_shortlist.pt','validation_shortlist.pt','no_context/checkpoint_best.pt','full_tacs/checkpoint_best.pt'):
        files['outputs/reference_shared_k4/'+name]=RUN/name
    for name in ('test_result.json','prospective_protocol.json','checkpoint_lock.json'):
        files['reference/'+name]=RUN/'evaluation/gaussian_noise_followup'/name
    files['reference/RESULT_SCOPE.md']=ROOT/'docs/TACS_SHARED_BACKBONE_K4_RESULT.md'
    # Preserve exact checkpoint bytes and its CRC while omitting unrelated runs.
    with zipfile.ZipFile(ROOT/'outputs.zip') as archive:
        entry=next(n for n in archive.namelist() if 'cifar100_no_context_c4_e5_unf4v3/' in n and n.endswith('checkpoint_best.pt'))
        initial=archive.read(entry)
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr(entry,initial)
    warm_archive=memory.getvalue()
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    hashes['outputs.zip']=hashlib.sha256(warm_archive).hexdigest()
    lock=json.loads((RUN/'evaluation/gaussian_noise_followup/checkpoint_lock.json').read_text())
    for arm,expected in lock['checkpoint_sha256'].items():
        if hashes[f'outputs/reference_shared_k4/{arm}/checkpoint_best.pt']!=expected:
            raise ValueError('Original measured checkpoint changed')
    manifest={'method':'TACS-X shared-backbone K4, ten clean continuation epochs, seed42',
              'files_sha256':hashes,'warmstart_entry':entry,'warmstart_sha256':hashlib.sha256(initial).hexdigest(),
              'reference_test_endpoint':json.loads((RUN/'evaluation/gaussian_noise_followup/test_result.json').read_text())['primary_endpoint_mean_noise_accuracy_difference_pp'],
              'scope':'one-seed exploratory Gaussian-noise result; not clean accuracy or an exact paper reproduction'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('__main__.py',LAUNCHER)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        for name,path in files.items():
            z.write(path,name)
        z.writestr('outputs.zip',warm_archive)
    temp.replace(args.output)
    print(json.dumps({'bundle':str(args.output),'size_mb':args.output.stat().st_size/1e6,
                      'reference_endpoint':manifest['reference_test_endpoint']},indent=2),flush=True)


if __name__=='__main__':
    main()
