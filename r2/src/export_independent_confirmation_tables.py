from pathlib import Path
import csv, json, hashlib
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
R=Path(__file__).resolve().parents[1]

def specs():
    m=pd.read_csv(R/'results/independent_confirmation/evaluation_full/mean_sd.csv').set_index(['method','protocol'])
    def v(method,prot,col,digits=3):
        row=m.loc[(method,prot)]
        fmt=lambda x: str(Decimal(str(round(float(x),12))).quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP))
        return fmt(row[col+'_mean'])+'\n('+fmt(row[col+'_std'])+')'
    rows=[]
    names={'list_all':'SCET, list features','list_ce':'Cross-encoder features','set_all':'SCET, evidence-set utility','set_ce':'Cross-encoder, evidence-set utility'}
    for prot,label in [('fixed_batch','Fixed batch'),('prospective','Calibrated threshold')]:
        for method in ['list_all','list_ce','set_all','set_ce']:
            rows.append([names[method]+'\n'+label,v(method,prot,'volume_per_100',1),v(method,prot,'evidence_precision'),v(method,prot,'all_facts_recall'),v(method,prot,'multiple_set_recall'),v(method,prot,'false_admission_no_evidence')])
    s36={'title':'Independent FEVER confirmation','headers':['Method and rule','Sentences /100','Precision','Set recall','Multiple-set recall','No-evidence admission'],
         'rows':rows,'widths':[.29,.12,.14,.14,.15,.16], 'bold':[[i,j] for i in [0,4] for j in [2,5]]+[[i,j] for i in [2,6] for j in [3,4]],
         'note':'Means (sample SD) over four fixed models on 3,000 claims. FEVER sentences use the June 2017 Wikipedia snapshot. Set recall uses 2,000 annotated claims; multiple-set recall uses 63 claims. Higher precision and recall and lower no-evidence admission are preferable. Bold identifies the best value within each rule. The cross-encoder evidence-set form selects beta zero on the original fitting folds.'}
    d=pd.read_csv(R/'results/independent_confirmation/evaluation_full/primary_intervals.csv'); rows=[]
    labels=[('List features minus cross-encoder features','Calibrated threshold','Precision'),('List features minus cross-encoder features','Fixed batch','Precision'),('Evidence-set utility minus cross-encoder evidence-set utility','Fixed batch','Precision'),('Evidence-set utility minus list features','Fixed batch','Multiple-set recall')]
    for (_,x),(contrast,rule,metric) in zip(d.iterrows(),labels):
        rows.append([contrast,rule,metric,str(Decimal(str(round(float(x.difference),12))).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)),f'[{x.four_primary_simultaneous_low:.4f}, {x.four_primary_simultaneous_high:.4f}]'])
    s37={'title':'Prespecified comparisons on independent FEVER claims','headers':['Comparison','Workload rule','Outcome','Difference','Simultaneous interval'],
         'rows':rows,'widths':[.31,.17,.16,.12,.24],'bold':[[i,j] for i in range(4) for j in [3,4]],
         'note':'Paired source-component bootstrap with 2,000 draws and verdict normalization, shared across the four fixed models. Bonferroni intervals jointly cover the four comparisons at the nominal 95% level, conditional on fitted models and admission decisions. Positive differences favour the first method. Bold differences and intervals are above zero.'}
    return {'S36':s36,'S37':s37}


def main():
    out=R/'results/tables/independent_confirmation'; out.mkdir(parents=True,exist_ok=True)
    for key,spec in specs().items():
        with (out/(key+'.csv')).open('w',encoding='utf-8',newline='') as h:
            csv.writer(h).writerows([spec['headers']]+spec['rows'])
    print('Recreated S36-S37 from saved numerical results.')

if __name__=='__main__':main()
