# qlora-chinese-example
使用qlora对中文大语言模型进行微调。

使用qlora对baichuan-7b进行微调，代码更加简洁：https://github.com/taishan1994/baichuan-Qlora-Tuning

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
----tokenizer.model：#原始llama的7B文件
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

ChatGLM-6B下载地址：[清华大学云盘 (tsinghua.edu.cn)](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/) 

```python
python qlora.py --model_name="chatglm" --model_name_or_path="./model_hub/chatglm-6b" --trust_remote_code=True --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="left" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/chatglm/" --lora_r=8 --lora_alpha=32
```

# Alpaca

Facebook官方发布的[LLaMA模型禁止商用](https://github.com/facebookresearch/llama)，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。自行搜索下载地址。

## 转换步骤

- 1、下载好7B、[llama-lora](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b)、[alpaca-lora](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b)到model_hub下。进入到model_hub目录下。
- 2、将llama转换为hugging face支持的格式：`python convert_llama_weights_to_hf.py --input_dir ./ --model_size 7B --output_dir ./7B-hf`。如果报错：`If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0`则可以`pip install --upgrade protobuf==3.20.1`，然后：`python convert_llama_weights_to_hf.py --input_dir ./ --model_size tokenizer_only --output_dir ./7B-hf`。最终我们可以得到7B-hf。
- 3、合并lora到llama上：`python merge_llama_with_chinese_lora.py --base_model "./7B-hf" --lora_model "./chinese-llama-plus-lora-7b,chinese-alpaca-plus-lora-7b" --output_type "huggingface" --output_dir "./chinese-alpaca-7b" `。最终我们可以得到chinese-alpaca-7b。

```python
python qlora.py --model_name="chinese_alpaca" --model_name_or_path="./model_hub/chinese-alpaca-7b" --trust_remote_code=False --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="right" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/alpaca/" --lora_r=8 --lora_alpha=32
```

# BLOOM

BELLE-7B-2M下载地址：[BelleGroup/BELLE-7B-2M at main (huggingface.co)](https://huggingface.co/BelleGroup/BELLE-7B-2M/tree/main)

```python
python qlora.py --model_name="chinese_bloom" --model_name_or_path="./model_hub/BELLE-7B-2M" --trust_remote_code=False --dataset="msra" --source_max_len=128 --target_max_len=64 --do_train --save_total_limit=1 --padding_side="left" --per_device_train_batch_size=8 --do_eval --bits=4 --save_steps=10 --gradient_accumulation_steps=1 --learning_rate=1e-5 --output_dir="./output/bloom/" --lora_r=8 --lora_alpha=32
```

# 闲聊

```python
python chat.py --model_name "chatglm" --base_model "./model_hub/chatglm-6b" --tokenizer_path "./model_hub/chatglm-6b" --lora_model "./output/chatglm/adapter_model" --with_prompt --interactive
```

# 补充

- **怎么训练自己的数据？** 数据格式为：

	```python
	{
	    "data": [
	        {"instruction":"", "input":"", "output":""},
	        {"instruction":"", "input":"", "output":""},
	        ...
	    ]
	}
	```

	然后在qlora.py里面定义数据的地方加上自己的数据集即可。最后运行指令的时候自己定义相关的参数。

# 参考

> [liucongg/ChatGLM-Finetuning: 基于ChatGLM-6B模型，进行下游具体任务微调，涉及Freeze、Lora、P-tuning等 (github.com)](https://github.com/liucongg/ChatGLM-Finetuning)
>
> [THUDM/ChatGLM-6B: ChatGLM-6B: An Open Bilingual Dialogue Language Model | 开源双语对话语言模型 (github.com)](https://github.com/THUDM/ChatGLM-6B/projects?query=is%3Aopen)
>
> [huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. (github.com)](https://github.com/huggingface/peft)
>
> [ymcui/Chinese-LLaMA-Alpaca: 中文LLaMA&Alpaca大语言模型+本地CPU/GPU训练部署 (Chinese LLaMA & Alpaca LLMs) (github.com)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
>
> [LianjiaTech/BELLE: BELLE: Be Everyone's Large Language model Engine（开源中文对话大模型） (github.com)](https://github.com/LianjiaTech/BELLE/)
>
> [artidoro/qlora: QLoRA: Efficient Finetuning of Quantized LLMs (github.com)](https://github.com/artidoro/qlora)

**哪位好心人士卡多的试试看最终效果吧，租AutoDL入不敷出呀**
