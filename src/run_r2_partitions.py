"""Run independent partitions with bounded resource use, then merge results."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import subprocess
import sys
import json

R2=Path(__file__).resolve().parents[1]
SCRIPT=R2/'src/run_r2_experiments.py'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='fever')
    parser.add_argument('--pool',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    seeds=json.loads((R2/'data/r2_protocol.json').read_text())['seeds']
    workers=min(max(1,args.workers),len(seeds),max(1,(os.cpu_count() or 2)//4))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def one(seed):
        log=R2/f'results/{args.dataset}_k{args.pool}_seed{seed}.log'
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,str(SCRIPT),'--datasets',args.dataset,'--mode','full','--pool',str(args.pool),'--seed',str(seed)],check=True,stdout=stream,stderr=subprocess.STDOUT,env=env)
        print('Completed partition',seed,flush=True)
    with ThreadPoolExecutor(workers) as executor:
        list(executor.map(one,seeds))
    subprocess.run([sys.executable,str(SCRIPT),'--datasets',args.dataset,'--mode','full','--pool',str(args.pool),'--merge-only'],check=True,env=env)
    print('Merged four partitions and query-cluster intervals',flush=True)


if __name__=='__main__':main()
