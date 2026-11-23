# 时序读图器训练缓存

在实验根目录设置 README.md 中的仿真输入变量并准备历史回放记录。为每个数据分区生成独立的缓存。

```bash
cache="$PWD/sequence_cache"
data=work/leg_reader_teacher_20261123/sequence/data.py
for split in train develop archive fresh public; do
    python "$data" "$split" "$cache" --workers 8
done
study=work/leg_registration_teacher_20261123
python "$study/group_fit.py" "$cache" --threads 6 --epochs 10
python "$study/group_eval.py" --cache "$cache" --workers 8
```
