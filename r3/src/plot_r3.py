"""Generate publication figures from saved analysis tables."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
METHODS=['Admission','Cross-encoder','Cross-encoder top-one','Cross-encoder list-cutoff']
SHORT=['Admission','CE','CE top-one','CE list-cutoff']
COLORS=['#253f67','#b45b35','#55896a','#7e6799']
MARKS=['o','s','^','D']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,
    'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'lines.linewidth':1.5,'savefig.facecolor':'white','axes.axisbelow':True})

def save(fig,out,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def main():
    default_out=ROOT/'figures'
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'results/analysis');p.add_argument('--out',type=Path,default=default_out);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data/'stratified_mean_sd.csv')
    f=pd.read_csv(args.data/'frontiers_mean_sd.csv')
    fig,axs=plt.subplots(1,3,figsize=(8.4,3.1),sharex=True)
    fills=['#55799e','#c3c9d0','#e3cdb6']
    for ax,(ds,name) in zip(axs,[('scifact','(a) SciFact'),('fever','(b) FEVER'),('hover','(c) HoVer')]):
        sub=d[(d.dataset==ds)&(d.verdict=='ALL')&(d.protocol=='fixed_batch')].set_index('method')
        y=list(range(4));gold=[sub.loc[m,'precision_mean'] for m in METHODS];nei=[sub.loc[m,'nei_item_share_mean'] for m in METHODS];ung=[1-a-b for a,b in zip(gold,nei)]
        ax.barh(y,gold,color=fills[0],height=.65)
        ax.barh(y,ung,left=gold,color=fills[1],height=.65)
        ax.barh(y,nei,left=[a+b for a,b in zip(gold,ung)],color=fills[2],height=.65)
        ax.set_yticks(y,SHORT);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Share of admitted sentences');ax.set_title(name);ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(fills,['Annotated evidence','Unannotated, evidence-bearing claim','No annotated evidence'])],loc='lower center',ncol=1,bbox_to_anchor=(.5,-.1),frameon=False)
    fig.subplots_adjust(left=.10,right=.99,top=.88,bottom=.30,wspace=.75)
    save(fig,args.out,'figure2_admission')
    fig,axs=plt.subplots(1,3,figsize=(8.4,3.2))
    for ax,(ds,name) in zip(axs,[('scifact','(a) SciFact'),('fever','(b) FEVER'),('hover','(c) HoVer')]):
        for m,c,mark in zip(METHODS,COLORS,MARKS):
            g=f[(f.dataset==ds)&(f.method==m)].sort_values('target')
            # Repeated top-one targets above 100 are the same output and need one marker.
            g=g.drop_duplicates(['all_facts_recall_mean','evidence_precision_mean'])
            ax.plot(g.all_facts_recall_mean,g.evidence_precision_mean,color=c,marker=mark,markersize=4,label=m)
            r=g[g.target==100]
            if len(r):ax.scatter(r.all_facts_recall_mean,r.evidence_precision_mean,s=50,facecolors='none',edgecolors=c,linewidths=1.1,zorder=4)
        ax.set_title(name);ax.set_xlabel('Complete-set recall');ax.set_ylabel('Evidence precision');ax.set_xlim(-.02,1.02);ax.set_ylim(0,1);ax.grid(alpha=.20)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.subplots_adjust(left=.07,right=.99,top=.88,bottom=.29,wspace=.43)
    save(fig,args.out,'figure3_risk_coverage')
    fig,axs=plt.subplots(1,3,figsize=(8.4,3.2))
    for ax,(ds,name) in zip(axs,[('scifact','(a) SciFact'),('fever','(b) FEVER'),('hover','(c) HoVer')]):
        metric='zero_admission_rate_mean' if ds=='hover' else 'false_admission_no_evidence_mean'
        for m,c,mark in zip(METHODS,COLORS,MARKS):
            g=f[(f.dataset==ds)&(f.method==m)].sort_values('target').drop_duplicates(['volume_per_100_mean',metric])
            ax.plot(g.volume_per_100_mean,g[metric],color=c,marker=mark,markersize=4,label=m)
        ax.set_title(name);ax.set_xlabel('Admitted sentences / 100 queries');ax.set_ylabel('Zero output rate' if ds=='hover' else 'No-evidence output rate');ax.set_ylim(0,1.03);ax.set_xlim(0,510);ax.set_xticks([0,100,300,500]);ax.grid(alpha=.20)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.subplots_adjust(left=.07,right=.99,top=.88,bottom=.29,wspace=.45)
    save(fig,args.out,'figure4_precision_volume')
    print('Wrote three editable SVG and vector PDF plots from saved CSV data.')

if __name__=='__main__':main()
