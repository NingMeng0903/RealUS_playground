"""Index saved Genesis renders; never synthesize images or change geometry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/anatomy_retarget/v15_review_20260908_001')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'outputs/anatomy_retarget'
    entries, videos = [], []
    for manifest in sorted(base.glob('v15*/manifest.json')):
        data = json.loads(manifest.read_text())
        if not isinstance(data.get('cells'), dict) or 'renderer' not in data:
            continue
        for cell in data['cells']:
            for path in sorted((manifest.parent / cell / 'comparison').glob('*.png')):
                entries.append(dict(experiment=manifest.parent.name, cell=cell,
                                    view=path.stem, path=os.path.relpath(path, output)))
    for path in sorted(base.glob('v15*/genesis_motion.mp4')):
        videos.append(dict(name=path.parent.name, path=os.path.relpath(path, output)))
    entries.sort(key=lambda item: (
        'pair_genesis' not in item['experiment'],
        item['experiment'], item['cell'], item['view']))
    gallery = dict(images=entries, videos=videos, anatomical_passed=False,
                   image_source='saved Genesis RGB from hashed geometry NPZs')
    (output / 'gallery.json').write_text(json.dumps(gallery, indent=2) + '\n')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V15 双侧联动修正：Genesis 二次检查</title>
<style>body{background:#111a22;color:#ecf0f3;font:16px system-ui;margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:25px}p{line-height:1.7}a{color:#91c9ff}.status{padding:14px 18px;border-left:4px solid #e4ac68;background:#2c2821}label{display:inline-block;vertical-align:top;margin:12px 16px 12px 0}select{display:block;max-width:430px;padding:10px;background:#22303b;color:white;border:1px solid #658095;margin-top:7px}img,video{width:100%;display:block;background:#09131c}details{margin:24px 0}summary{cursor:pointer}small{color:#bfccd6}#caption{overflow-wrap:anywhere}</style>
<main><h1>V15：双侧联动修正后的真实三维检查</h1>
<p class="status"><b>尚未完成全部要求，解剖验收未通过。</b>已保存的两个体型可重复用同一个包由 θ 驱动；这不代表任意 β、任意动作中的内部结构都合理。当前候选修正双侧锁骨响应，还未完成全身骨长与关节接触适配、骨—皮双边界软组织运输。</p>
<p>修正前后使用同一 β、θ、SMPL-X 皮肤和相机。左图为原左侧锁骨修正版，右图为新双侧响应候选。两份采集动作变化小；走路、喝水的右臂及相连血管有明显位置变化，局部关节和软组织仍需修正。</p>
<p><small>骨骼、血管与神经为不透明材质，皮肤透明。局部相机在关节前后各裁剪 160 mm，平切端可能来自裁剪。投影重叠不能单独认定为穿入或无穿入；结合不同视角与表面测量判断。骨骼轻微出皮按部位和合理性图审，不使用统一骨出皮否决阈值。</small></p>
<label>体型 / 实验<select id="experiment"></select></label>
<label>动作<select id="cell"></select></label><label>视角<select id="view"></select></label>
<img id="picture" alt="已保存几何的 Genesis 对照图"><p id="caption"></p>
<details open><summary>连续动作视频</summary><p>视频使用固定编译包逐帧回放，没有重新 bake。步行视频为原生 100 fps 动作每 8 帧取一帧，33 帧按 12.5 fps 播放；它用于视觉检查，不是全原生帧验收。</p><div id="videos"></div></details>
<p><a href="../../../MD/anatomy_v15_execution_review_20260908.md">本轮修改、图审与未完成项</a> · <a href="../../../MD/anatomy_v15_independent_recheck_20260908.md">旧包独立二次检查</a> · <a href="../../../MD/anatomy_v15_bilateral_independent_review_20260908.md">新候选独立检查</a> · <a href="../../../MD/anatomy_v15_general_beta_plan_20260908.md">完整 β / θ 方案</a> · <a href="../v14_review_20260908_001/index.html">V14 历史图审</a></p>
<p id="count"></p></main>
<script>const entries=__ENTRIES__,videos=__VIDEOS__;
const experiment=document.querySelector('#experiment'),cell=document.querySelector('#cell'),view=document.querySelector('#view');
function fill(el,values){const old=el.value;el.replaceChildren(...[...new Set(values)].map(v=>new Option(v,v)));if(values.includes(old))el.value=old;}
function updateCells(){fill(cell,entries.filter(e=>e.experiment===experiment.value).map(e=>e.cell));updateViews();}
function updateViews(){const old=view.value;fill(view,entries.filter(e=>e.experiment===experiment.value&&e.cell===cell.value).map(e=>e.view));if(!old&&[...view.options].some(o=>o.value==='whole_ap'))view.value='whole_ap';show();}
function show(){const e=entries.find(e=>e.experiment===experiment.value&&e.cell===cell.value&&e.view===view.value);if(!e)return;document.querySelector('#picture').src=e.path;const a=document.createElement('a');a.href=e.path;a.textContent='打开原图：'+e.experiment+' / '+e.cell+' / '+e.view;document.querySelector('#caption').replaceChildren(a);}
fill(experiment,entries.map(e=>e.experiment));experiment.onchange=updateCells;cell.onchange=updateViews;view.onchange=show;updateCells();
for(const item of videos){const p=document.createElement('p');p.textContent=item.name;const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=item.path;document.querySelector('#videos').append(p,video);}
document.querySelector('#count').textContent=entries.length+' 张真实对照图，'+videos.length+' 段视频。';</script></html>'''
    html = html.replace('__ENTRIES__', json.dumps(entries, ensure_ascii=False))
    html = html.replace('__VIDEOS__', json.dumps(videos, ensure_ascii=False))
    (output / 'index.html').write_text(html, encoding='utf-8')
    print(json.dumps(dict(output=str(output), images=len(entries), videos=len(videos))))


if __name__ == '__main__':
    main()
