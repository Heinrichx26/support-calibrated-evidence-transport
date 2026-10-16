"""Create a new computation directory while preserving released reference results."""
from pathlib import Path
import argparse
import shutil

R=Path(__file__).resolve().parents[1]


def run(output,cached,training_base):
    output=output.resolve()
    assert not output.exists(),'Use a new output directory.'
    output.mkdir(parents=True)
    def copy(source):
        destination=output/source.relative_to(R)
        destination.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,destination)
    for source in (R/'src').glob('*.py'):copy(source)
    copy(R/'data/r2_protocol.json')
    for source in (R/'data/independent_confirmation').glob('*'):
        if source.is_file():copy(source)
    directories=[R/'results/r5_context_gain/development']
    if training_base is None:directories.append(R/'results/evidence_set_confirmation/balanced/frozen')
    for directory in directories:
        for source in directory.glob('*'):
            if source.is_file():copy(source)
    for source in R.glob('requirements-independent-confirmation*.txt'):copy(source)
    if cached:
        for source in (R/'data/independent_confirmation/scores').rglob('*'):
            if source.is_file() and source.name in ['scored.parquet','frozen_predictions.parquet']:copy(source)
        for source in (R/'results/independent_confirmation').glob('scoring_*_audit.json'):copy(source)
    if training_base is not None:
        training_base=training_base.resolve()
        relatives=['data/scored/fever_full_k100.parquet','data/processed/fever_full_queries.json']
        relatives += [f'results/experiments/fever_full_k100/model_seed{seed}.json' for seed in [13,17,29,7]]
        for relative in relatives:
            source=training_base/relative;assert source.is_file(),str(source)
            destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    print(str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cached-scores',action='store_true');parser.add_argument('--training-base',type=Path);args=parser.parse_args()
    run(args.output,args.cached_scores,args.training_base)
