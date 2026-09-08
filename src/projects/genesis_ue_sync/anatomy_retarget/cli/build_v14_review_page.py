"""Index the actual V14 Genesis comparison images and continuous videos."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/anatomy_retarget/v14_review_20260908_001')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'outputs/anatomy_retarget'; entries=[]; videos=[]
    for manifest in sorted(base.glob('v14*/manifest.json')):
        data=json.loads(manifest.read_text())
        if 'cells' not in data or 'renderer' not in data:continue
        for cell in data['cells']:
            folder=manifest.parent/cell/'comparison'
            for path in sorted(folder.glob('*.png')):
                entries.append(dict(version=manifest.parent.name,cell=cell,view=path.stem,
                    path='../'+str(path.relative_to(base))))
    for path in sorted(base.glob('v14*/genesis_motion.mp4')):
        videos.append(dict(name=path.parent.name,path='../'+str(path.relative_to(base))))
    # Start on the current fully reviewed six-cell candidate, keeping every
    # earlier failure available in the same selector.
    preferred = 'v14_driver_axes_genesis_20260908_001'
    entries.sort(key=lambda e: (e['version'] != preferred, e['version'], e['cell'], e['view']))
    videos.sort(key=lambda v: (not v['name'].startswith('v14_driver_axes_'), v['name']))
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V14 Genesis 三维复核</title>
<style>body{margin:0;background:#12181e;color:#eef2f6;font:16px system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:24px}p{line-height:1.6;color:#bac5cf}.status{border-left:4px solid #eeab61;padding:8px 16px;background:#28231e}label{display:inline-block;margin:16px 18px 16px 0}select{display:block;margin-top:7px;max-width:420px;padding:9px;background:#26323d;color:white;border:1px solid #5a6b79}img,video{display:block;width:100%;background:#09131c}a{color:#8bcaff}#caption{word-break:break-all}details{margin-top:24px}summary{cursor:pointer}small{color:#9dabb8}</style>
<main><h1>V14：真实 Genesis 三维复核</h1>
<p class="status">研究候选，尚未通过解剖验收。静态形状、逐顶点权重或回放一致，不等于动作中无穿入。失败结果保留用于比较。</p>
<p>左右图使用同一皮肤、姿态和相机；内部骨骼、血管、神经均为不透明材质，皮肤透明。局部视角的前后裁剪各为 160 mm，画面中的平切端可能来自相机裁剪。图中自带左右版本名称与 10 mm 标尺。</p>
<label>版本 / 实验<select id="version"></select></label><label>采集体型 / 姿态<select id="cell"></select></label><label>视角<select id="view"></select></label>
<img id="picture" alt="真实几何的 Genesis 三维对比"><p id="caption"></p>
<details open><summary>连续动作视频</summary><div id="videos"></div></details>
<p><a href="../../../MD/anatomy_v14_implementation_20260908.md">实现与验收记录</a> · <a href="../../../MD/anatomy_v14_driver_axes_independent_review_20260908.md">当前两体型六格独立图审</a> · <a href="../../../MD/anatomy_v14_continuous_driver_axes_review_20260908.md">当前连续动作独立图审</a> · <a href="../../../MD/anatomy_v14_lower_chain_audit_20260908.md">髋膝踝独立骨面审计</a> · <a href="../../../MD/anatomy_v14_baseline_review_20260908.md">独立 V7 对照图审</a></p>
<small>该页只索引已有渲染，不修改几何或重新 retarget。</small></main>
<script>const entries=__ENTRIES__;const videos=__VIDEOS__;
const version=document.querySelector('#version'),cell=document.querySelector('#cell'),view=document.querySelector('#view');
function fill(el,values){const old=el.value;el.replaceChildren(...[...new Set(values)].map(v=>new Option(v,v)));if(values.includes(old))el.value=old;}
function updateCells(){fill(cell,entries.filter(e=>e.version===version.value).map(e=>e.cell));updateViews();}
function updateViews(){const previous=view.value;fill(view,entries.filter(e=>e.version===version.value&&e.cell===cell.value).map(e=>e.view));if(!previous&&[...view.options].some(o=>o.value==='whole_ap'))view.value='whole_ap';show();}
function show(){const e=entries.find(e=>e.version===version.value&&e.cell===cell.value&&e.view===view.value);if(!e)return;document.querySelector('#picture').src=e.path;const a=document.createElement('a');a.href=e.path;a.textContent='打开原始对比图：'+e.version+' / '+e.cell+' / '+e.view;document.querySelector('#caption').replaceChildren(a);}
fill(version,entries.map(e=>e.version));version.onchange=updateCells;cell.onchange=updateViews;view.onchange=show;updateCells();
for(const v of videos){const p=document.createElement('p');p.textContent=v.name+'（诊断候选）';const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=v.path;document.querySelector('#videos').append(p,video);}
if(!videos.length)document.querySelector('#videos').textContent='连续渲染尚未加入本页。';</script></html>'''
    content=content.replace('__ENTRIES__',json.dumps(entries,ensure_ascii=False)).replace('__VIDEOS__',json.dumps(videos,ensure_ascii=False))
    (args.output/'index.html').write_text(content,encoding='utf-8')
    (args.output/'gallery.json').write_text(json.dumps(dict(images=entries,videos=videos),indent=2)+'\n')
    print(len(entries),'comparison images;',len(videos),'videos;',args.output/'index.html')


if __name__=='__main__':main()
