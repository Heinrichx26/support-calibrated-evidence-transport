"""Bounded admission-operator evaluation with matched controls and audits."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

import r2_admission as admission
from r2_statistics import query_statistics
from strengthen_r2_analysis import budget_rule
from run_multivers_rationale_extension import load_frame
from run_multivers_extension import paired_intervals

R2 = Path(__file__).resolve().parents[1]
SPEC_PATH = R2/'data/multivers_admission_protocol.json'
SPEC = json.loads(SPEC_PATH.read_text())
OUT = R2/'results/multivers_extension/admission_operators'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Operators:
    def __init__(self):
        corpus = [json.loads(line) for line in (R2/'data/raw/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines()]
        ids, sentences = [], []
        for document in corpus:
            for i, sentence in enumerate(document['abstract']):
                ids.append(f"{document['doc_id']}::{i}")
                sentences.append(sentence)
        self.positions = {cid: i for i, cid in enumerate(ids)}
        self.vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', dtype=np.float32).fit_transform(sentences)
        records = []
        for line in (R2/'results/strengthening/multivers/document_scores.jsonl').read_text().splitlines():
            row = json.loads(line)
            records.append({'query_id': row['query_id'], 'doc_id': row['doc_id'],
                            'full_mass': float(sum(row['rationale_probabilities']))})
        self.mass = pd.DataFrame(records).set_index(['query_id', 'doc_id']).full_mass.to_dict()

    def family(self, frame):
        p = frame.raw_multivers_rationale.to_numpy(float)
        qr = admission.ranks(frame, p)
        grouped = frame[['query_id', 'doc_id', 'tie_key']].copy()
        grouped['p'] = p
        ordered = grouped.sort_values(['query_id', 'doc_id', 'p', 'tie_key'], ascending=[True, True, False, True])
        dr = np.empty(len(frame), dtype=int)
        dr[ordered.index] = (ordered.groupby(['query_id', 'doc_id']).cumcount()+1).to_numpy()
        other = np.maximum(0, np.array([self.mass[(q, d)] for q, d in zip(frame.query_id, frame.doc_id)])-p)
        out = {'rationale': p, 'product': frame.raw_multivers.to_numpy(float)}
        for key, rank in [('document_rank_discount', dr), ('query_rank_discount', qr)]:
            for lam in SPEC['operators'][key]:
                out[f'{key}:{lam:g}'] = p/(1+lam*(rank-1))
        for lam in SPEC['operators']['document_other_mass_discount']:
            out[f'document_other_mass_discount:{lam:g}'] = p/(1+lam*other)
        for lam in SPEC['operators']['maximal_marginal_relevance']:
            output = np.full(len(frame), -2., dtype=float)
            for _, ix in frame.groupby('query_id', sort=False).indices.items():
                ix = np.asarray(ix)
                positions = [self.positions[c] for c in frame.iloc[ix].candidate_id]
                vectors = self.vectors[positions]
                similarity = (vectors@vectors.T).toarray()
                redundancy = np.zeros(len(ix))
                used = np.zeros(len(ix), dtype=bool)
                previous = float('inf')
                for _ in range(min(5, len(ix))):
                    value = p[ix]-lam*redundancy
                    value[used] = -np.inf
                    choice = np.lexsort((frame.iloc[ix].tie_key.to_numpy(str), -value))[0]
                    assert value[choice] <= previous+1e-12
                    output[ix[choice]] = value[choice]
                    previous = value[choice]
                    used[choice] = True
                    redundancy = np.maximum(redundancy, similarity[:, choice])
            out[f'maximal_marginal_relevance:{lam:g}'] = output
        out['one_candidate_per_document'] = np.where(dr == 1, p, -1.-dr*1e-6)
        out['one_candidate_per_query'] = np.where(qr == 1, p, -1.-qr*1e-6)
        return out


def inner_select(fitting, family, seed):
    ids = sorted(fitting.query_id.unique(), key=lambda q: hashlib.sha256(f'{seed}:admission:{q}'.encode()).hexdigest())
    mapping = {q: i % 4 for i, q in enumerate(ids)}
    fold = fitting.query_id.map(mapping).to_numpy()
    rows = []
    for f in range(4):
        valid = fitting.loc[fold == f].reset_index(drop=True)
        for name, scores in family.items():
            s = scores[fold == f]
            selected, _ = admission.apply_threshold(valid, s, budget_rule(valid, s, valid.query_id.nunique()))
            rows.append({'seed': seed, 'fold': f, 'method': name, 'precision': float(valid.gold_evidence.to_numpy(int)[selected].mean())})
    cv = pd.DataFrame(rows)
    means = cv.groupby('method').precision.mean().to_dict()
    names = list(family)
    choice = max(names, key=lambda n: (round(means[n], 12), -names.index(n)))
    return choice, cv


def run(smoke):
    start = time.monotonic()
    code_sha = digest(Path(__file__))
    if not smoke:
        audit = json.loads((OUT/'smoke/audit.json').read_text())
        assert audit['passed'] and audit['protocol_sha256'] == digest(SPEC_PATH) and audit['code_sha256'] == code_sha
    out = OUT/('smoke' if smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    frame = load_frame().reset_index(drop=True)
    operators = Operators()
    # Scores depend only on each query and frozen corpus/model probabilities.
    all_scores = operators.family(frame)
    if smoke:
        probe = frame[frame.query_id.isin(sorted(frame.query_id.unique())[:3])].reset_index(drop=True)
        normal = operators.family(probe)
        changed = probe.copy()
        changed['gold_evidence'] = 1-changed.gold_evidence
        changed['claim_verdict'] = 'unused'
        changed = changed.iloc[::-1].reset_index(drop=True)
        reverse = operators.family(changed)
        for name in normal:
            assert np.allclose(normal[name], reverse[name][::-1], rtol=0, atol=1e-10), name
    query_info = json.loads((R2/'data/processed/scifact_full_queries.json').read_text(encoding='utf-8'))
    summaries, choices, cvs = [], [], []
    collections = {}
    for seed in [13] if smoke else SPEC['seeds']:
        old = json.loads((R2/f'results/experiments/scifact_full_k100/model_seed{seed}.json').read_text())
        fitids, calids, testids = [old[k] for k in ['fitting_query_ids', 'threshold_query_ids', 'evaluation_query_ids']]
        fm = frame.query_id.isin(fitids).to_numpy()
        fitting = frame.loc[fm].reset_index(drop=True)
        choice, cv = inner_select(fitting, {k: s[fm] for k, s in all_scores.items()}, seed)
        choices.append({'seed': seed, 'choice': choice})
        cvs.append(cv)
        (out/f'choice_seed{seed}.json').write_text(json.dumps(choices[-1], indent=2))
        if smoke:
            testids = sorted(testids)[:12]
        tm, cm = frame.query_id.isin(testids).to_numpy(), frame.query_id.isin(calids).to_numpy()
        test, cal = frame.loc[tm].reset_index(drop=True), frame.loc[cm].reset_index(drop=True)
        queries = {q: query_info[q] for q in testids}
        for label, name in [('rationale', 'rationale'), ('product', 'product'), ('selected', choice),
                            ('document_quota', 'one_candidate_per_document'), ('query_quota', 'one_candidate_per_query')]:
            s, cs = all_scores[name][tm], all_scores[name][cm]
            rules = {'exact_budget': budget_rule(test, s, len(testids)),
                     'prospective_100': admission.fit_threshold(cal, cs, 100)}
            for analysis, rule in rules.items():
                stat = query_statistics(test, s, rule, np.zeros(len(test), dtype=bool), queries)
                if analysis == 'exact_budget':
                    assert stat.admitted.sum() == len(testids)
                stat.to_csv(out/f'query_seed{seed}_{label}_{analysis}.csv', index=False)
                summaries.append({'seed': seed, 'method': label, 'chosen': name, 'analysis': analysis, **admission.summarize(stat)})
                collections.setdefault(analysis, {}).setdefault(seed, {})[label] = stat
        print('seed', seed, 'choice', choice, 'elapsed', round(time.monotonic()-start, 1), flush=True)
    data = pd.DataFrame(summaries)
    data.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(choices).to_csv(out/'selected.csv', index=False)
    pd.concat(cvs).to_csv(out/'inner_cv.csv', index=False)
    data.groupby(['analysis', 'method'])[['evidence_precision', 'coverage', 'evidence_recall', 'all_facts_recall', 'volume_per_100']].agg(['mean', 'std']).to_csv(out/'mean_sd.csv')
    if not smoke:
        rows = []
        for analysis, stats in collections.items():
            for reference in ['product', 'rationale', 'document_quota', 'query_quota']:
                for row in paired_intervals(stats, 'selected', reference, ['evidence_precision', 'coverage', 'evidence_recall', 'all_facts_recall', 'volume_per_100']):
                    rows.append({'analysis': analysis, **row})
        pd.DataFrame(rows).to_csv(out/'paired_intervals.csv', index=False)
    audit = {'passed': True, 'smoke': smoke, 'protocol_sha256': digest(SPEC_PATH), 'code_sha256': code_sha,
             'fitting_query_selection_only': True, 'independent_threshold_queries': True,
             'outcome_and_row_order_invariance': smoke, 'exact_budget_verified': True,
             'exploratory': True, 'elapsed_seconds': time.monotonic()-start}
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    print(data[data.analysis == 'exact_budget'][['seed', 'method', 'chosen', 'evidence_precision', 'coverage', 'all_facts_recall']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke)
