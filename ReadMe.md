# MiraTrainer: for better mirabot!

这是mirabot的子项目, 目的是搭建一个训练模型的框架, 方便复用和快速实验.
框架绝大部分将参考[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), 感谢开源框架!

目前的进展主要是学习和复现各个框架的训练方法:
+ [x] pt: gpt2 -> huggingface script clm_run.py
+ [x] sft: Qwen -> zero_nlp script train_Qwen2_sft.py & Qwen official script
+ [x] dpo: from zero_nlp and LLaMA-Factory -> (no script...)
+ [ ] rlhf: ?
+ [ ] p-tuning: ?

其他任务:
+ [ ] 一个造数据,清洗加质检的Pipeline.可能会参考[distilabel](https://github.com/argilla-io/distilabel)
