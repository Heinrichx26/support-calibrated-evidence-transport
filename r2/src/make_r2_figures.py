"""Generate editable vector figures exclusively from completed result files."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import subprocess
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R2=Path(__file__).resolve().parents[1]
OUT=R2/'results/figures'
OUT.mkdir(exist_ok=True)
COLORS={'scet':'#1f77b4','cal_single':'#7f7f7f','ce':'#d62728','e5':'#9467bd','bge':'#2ca02c','nli_relevance':'#ff7f0e'}
LABELS={'scet':'SCET','cal_single':'Calibration-selected single','ce':'Cross-encoder','e5':'E5','bge':'BGE-M3','nli_relevance':'NLI relevance'}
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    for extension in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{extension}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def selected(dataset):
    root=R2/f'results/experiments/{dataset}_full_k100'
    dat=pd.read_csv(root/'primary.csv')
    rows=[]
    for seed in [13,17,29,7]:
        model=json.loads((root/f'model_seed{seed}.json').read_text())
        rows.append(dat[(dat.seed==seed)&(dat.method==model['primary_single'])].iloc[0])
    return pd.DataFrame(rows)


def admission():
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.35))
    methods=['scet','cal_single','ce','e5']
    for ax,(metric,title) in zip(axes,[('evidence_precision','(a) Evidence precision'),('evidence_recall','(b) Retained evidence recall'),('zero_admission_rate','(c) Zero-admission rate')]):
        for j,method in enumerate(methods):
            means,sds=[],[]
            for ds in ['scifact','fever','hover']:
                dat=selected(ds) if method=='cal_single' else pd.read_csv(R2/f'results/experiments/{ds}_full_k100/primary.csv')
                if method!='cal_single': dat=dat[dat.method==method]
                means.append(dat[metric].mean());sds.append(dat[metric].std(ddof=1))
            ax.bar(np.arange(3)+(j-1.5)*.19,means,.18,yerr=sds,capsize=2,color=COLORS[method],label=LABELS[method],linewidth=.6)
        ax.set_xticks(range(3),['SciFact','FEVER','HoVer']);ax.set_ylim(0,1);ax.set_title(title,loc='left');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'figure2_admission')


def curves(kind):
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.35))
    for ax,ds,name in zip(axes,['scifact','fever','hover'],['SciFact','FEVER','HoVer']):
        dat=pd.read_csv(R2/f'results/experiments/{ds}_full_k100/volume_mean_sd.csv')
        for method in ['scet','ce','e5','bge','nli_relevance']:
            row=dat[dat.method==method].sort_values('calibration_target')
            x=row['coverage_mean'] if kind=='risk' else row['volume_per_100_mean']
            y=1-row['evidence_precision_mean'] if kind=='risk' else row['evidence_precision_mean']
            ax.plot(x,y,'o-',color=COLORS[method],label=LABELS[method],markersize=3,linewidth=1.2)
        ax.set_title(name,loc='left');ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.set_xlabel('Query coverage' if kind=='risk' else 'Admitted items per 100 queries')
        if kind=='risk': ax.set_xlim(0,1)
    axes[0].set_ylabel('Unannotated fraction of admitted items' if kind=='risk' else 'Evidence precision')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.02),fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'figure3_risk_coverage' if kind=='risk' else 'figure4_precision_volume')


def workflow():
    # diagrams.net is the editable source and export authority for the framework.
    root=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
    dia=ET.SubElement(root,'diagram',id='r2-framework',name='SCET admission')
    graph=ET.SubElement(dia,'mxGraphModel',dx='1040',dy='390',grid='1',gridSize='10',page='1',pageScale='1',pageWidth='1040',pageHeight='390')
    base=ET.SubElement(graph,'root');ET.SubElement(base,'mxCell',id='0');ET.SubElement(base,'mxCell',id='1',parent='0')
    entries=[('a','Original corpus\nSentence identities\nVerified text',20,40,205,80),('b','Fixed candidate pool\nRandom tie keys',275,40,205,80),('c','Evidence views\nRaw cross-query scores',530,40,205,80),('d','Held-out admission\nFrozen form and threshold',785,40,225,80),('e','Fitting queries\nView transforms and models\nScoring-form selection',275,200,205,100),('f','Threshold calibration\nWorkload order statistic\nOptional risk testing',530,200,205,100),('g','Admission record\nSource, score, rank\nYield and coverage',785,200,225,100)]
    # Plain labels export as native SVG text and remain selectable in PDF.
    style='rounded=0;whiteSpace=wrap;html=0;fillColor=#ffffff;strokeColor=#333333;fontFamily=Times New Roman;fontSize=17;fontColor=#000000;align=center;verticalAlign=middle;'
    for key,label,x,y,w,h in entries:
        c=ET.SubElement(base,'mxCell',id=key,value=label,style=style,vertex='1',parent='1');ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    for j,(s,t) in enumerate([('a','b'),('b','c'),('c','d'),('b','e'),('e','f'),('f','d'),('d','g')]):
        anchor='exitX=1;exitY=0.5;entryX=0;entryY=0.75;' if (s,t)==('f','d') else ''
        c=ET.SubElement(base,'mxCell',id='edge'+str(j),style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#333333;strokeWidth=1.5;'+anchor,edge='1',parent='1',source=s,target=t)
        ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    drawio=OUT/'figure1_workflow.drawio';ET.ElementTree(root).write(drawio,encoding='utf-8',xml_declaration=True)
    exe=os.environ.get('DRAWIO_BINARY') or shutil.which('draw.io') or shutil.which('drawio')
    if not exe:raise RuntimeError('Set DRAWIO_BINARY to the diagrams.net desktop executable.')
    subprocess.run([str(exe),'--export','--format','svg','--output',str(OUT/'figure1_workflow.svg'),str(drawio)],check=True,timeout=90,capture_output=True)
    # Desktop exports can rasterize foreignObject labels. Retain the exported
    # shapes and routes and replace those label groups with native SVG text.
    namespace='http://www.w3.org/2000/svg'
    ET.register_namespace('',namespace)
    svg=ET.parse(OUT/'figure1_workflow.svg')
    svg_root=svg.getroot()
    for key,label,x,y,w,h in entries:
        cell=next(g for g in svg_root.iter('{'+namespace+'}g') if g.get('data-cell-id')==key)
        for parent in cell.iter():
            for child in list(parent):
                if child.tag=='{'+namespace+'}switch':parent.remove(child)
        lines=label.split('\n')
        center_x=x-20+w/2
        first_y=y-40+h/2-(len(lines)-1)*20.4/2+5.7
        for j,line in enumerate(lines):
            text_node=ET.SubElement(cell,'{'+namespace+'}text',{
                'x':str(center_x),'y':str(first_y+j*20.4),'text-anchor':'middle',
                'font-family':'Times New Roman','font-size':'17','fill':'#000000'})
            text_node.text=line
    for child in list(svg_root):
        if child.tag=='{'+namespace+'}switch':svg_root.remove(child)
    svg.write(OUT/'figure1_workflow.svg',encoding='utf-8',xml_declaration=True)
    inkscape=os.environ.get('INKSCAPE_BINARY') or shutil.which('inkscape')
    if not inkscape:raise RuntimeError('Set INKSCAPE_BINARY to the Inkscape executable.')
    subprocess.run([str(inkscape),str(OUT/'figure1_workflow.svg'),'--export-type=pdf','--export-filename='+str(OUT/'figure1_workflow.pdf')],check=True,capture_output=True)
    subprocess.run([str(inkscape),str(OUT/'figure1_workflow.svg'),'--export-type=pdf','--export-latex','--export-filename='+str(OUT/'figure1_workflow_tex.pdf')],check=True,capture_output=True)
    subprocess.run([str(inkscape),str(OUT/'figure1_workflow.svg'),'--export-type=png','--export-dpi=220','--export-filename='+str(OUT/'figure1_workflow.png')],check=True,capture_output=True)
    latex=r'''\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx,graphicx,xcolor}
\begin{document}
\fontsize{12.75}{15.3}\selectfont
\input{figure1_workflow_tex.pdf_tex}
\end{document}
'''
    (OUT/'figure1_workflow_latex.tex').write_text(latex,encoding='utf-8')
    compiler=os.environ.get('PDFLATEX_BINARY') or shutil.which('pdflatex')
    if not compiler:raise RuntimeError('Set PDFLATEX_BINARY to the pdfLaTeX executable.')
    subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','figure1_workflow_latex.tex'],cwd=OUT,check=True,capture_output=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workflow-only',action='store_true');p.add_argument('--plots-only',action='store_true');args=p.parse_args()
    if not args.plots_only:workflow()
    if not args.workflow_only:
        admission();curves('risk');curves('volume')
