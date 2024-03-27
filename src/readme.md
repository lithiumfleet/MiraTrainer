# MiraTrainer src

目前以Qwen1.5-1.8B/Chat为例, 跑通三段式后兼容其他模型.
在这里做一些规范, 脚本间的规定可以起到接口的作用

## Config

每个脚本都有其对应的{filename}_config.json


## Dataset

### 1.pt

对于预训练, 假设所有文件都是txt格式文件(doc), 都存放在src/data/pt下, 依此获得一个文件里的文本(text). 在处理后根据模型输入长度拼接成文字块(chunck).
load_dataset的输出是一个Dataset, 格式如下
```json
{
    "train": {
        "input_ids":
        "attention_mask":
        "target_ids":
    }
}
```

## Logs

任何log用logging输出.

## Check

尽可能多加assert, 保证error及时被反馈.
