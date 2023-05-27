# qlora-chinese-example
使用qlora对中文大语言模型进行微调。

# 依赖

```python
numpy==1.24.2
pandas==2.0.0
nltk==3.7
transformers==4.30.0.dev0
accelerate==0.20.0.dev0
deepspeed==0.9.2
peft==0.4.0.dev0
datasets==2.12.0
evaluate==0.2.2
sentencepiece==0.1.97
scipy==1.10.1
icetk
cpm_kernels
mpi4py==3.1.4
```

# 目录结构

```python
--output #训练保存lora权重
----chatglm
----alpaca
----bloom
--data
----msra
------instruct_data
--------train.json  #指令数据
--model_hub
----BELLE-7B-2M #bloom权重
----chatglm-6b  #chatGLM权重
----7B：#英文LLaMA原始权重
----7B-hf：#英文权重转换为hugging face格式权重
----chinese-llama-plus-lora-7b：#中文llama-7b的lora权重
----chinese-alpaca-plus-lora-7b：#中文alpaca-7b的lora权重
----chinese-alpaca-7b：#合并lora后的最终的模型
----tokenizer.model：#7B文件
----convert_llama_weights_to_hf.py  #llama转换为hugging face格式
----merge_llama_with_chinese_lora.py  #合并lora到预训练模型
--tools
----get_version.py  #获取python包版本
----get_used_gpus.py  #循环打印使用的GPU显卡
--chat.py  # 闲聊
--qlora.py  # 4bit训练
--process.py  # 测试处理数据
```

# ChatGLM

```python
python qlora.py --model_name="chatglm" --model_name_or_path="./model_hub/chatglm-6b" --trust_remote_code=True --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="left" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/chatglm/" --lora_r=8 --lora_alpha=32
```

# Alpaca

```python
python qlora.py --model_name="chinese_alpaca" --model_name_or_path="./model_hub/chinese-alpaca-7b" --trust_remote_code=False --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="right" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/alpaca/" --lora_r=8 --lora_alpha=32
```

# BLOOM

```python
python qlora.py --model_name="chinese_bloom" --model_name_or_path="./model_hub/BELLE-7B-2M" --trust_remote_code=False --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="left" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/bloom/" --lora_r=8 --lora_alpha=32
```

# 闲聊

```python
python chat.py --model_name "chatglm" --base_model "./model_hub/chatglm-6b" --tokenizer_path "./model_hub/chatglm-6b" --lora_model "./output/chatglm/adapter_model" --with_prompt --interactive
```

