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
        "input_ids": "tokens",
        "attention_mask": "ones",
        "labels": "copy of input_ids"
    }
}
```

### 2.sft

数据格式采用 Vicuna格式, 后续可能会添加"tools"等字段.
```json
"conversations": [
    [
        {
            "from": "human",
            "value": "Who are you?"
        },
        {
            "from": "gpt",
            "value": "I am Vicuna, ..."
        }
    ],
    [
        {
            "from": "human",
            "value": "What can you do?"
        },
        {
            "from": "gpt",
            "value": "I can chat with you."
        }
    ]
]
```
load_dataset的输出同样是Dataset, 格式如下:
```json
{
    "train": {
        "input_ids": "tokens",
        "attention_mask": "ones",
        "target_ids": "[IGNORE_INDEX]*n + tokens"
    }
}
```
针对长对话, 这里会用最简单的方式: 参考LLaMA-Factory, 先模板化(ChatML或其他), 每次response都会连带上文当成一个example.

## Logs

任何log用logging输出.

## Check

尽可能多加assert, 保证error及时被反馈.
