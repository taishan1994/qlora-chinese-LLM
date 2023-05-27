# coding=utf-8
import copy
import datasets
import json
import torch
import transformers
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def process_msra():
    with open("./data/msra/instruct_data/train.txt", "r", encoding="utf-8") as fp:
        data = fp.read().strip().split("\n")
    res = []
    tmp = {}
    tmp["version"] = "0.1.0"
    tmp["data"] = []
    for d in data:
        d = eval(d)
        d_tmp = {}
        d_tmp["instruction"] = d["instruct"]
        d_tmp["input"] = d["query"]
        d_tmp["output"] = d["answer"]
        tmp["data"].append(d_tmp)
    with open("./data/msra/instruct_data/train.json", "w", encoding="utf-8") as fp:
        json.dump(tmp, fp, ensure_ascii=False, indent=2)


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
    "chatglm_input": ("{instruction}{input}"),
    "alpaca_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}{input}\n\n### Response: "
    ),
    "bloom_input": ("Human: \n{instruction}{input}\n\nAssistant: \n"),
}


def extract_alpaca_dataset(example, model_name="chatglm"):
    if example.get("input", "") != "":
        if model_name == "chatglm":
            prompt_format = PROMPT_DICT["chatglm_input"]
        elif model_name == "chinese_alpaca":
            prompt_format = PROMPT_DICT["alpaca_input"]
        elif model_name == "chinese_bloom":
            prompt_format = PROMPT_DICT["bloom_input"]
        else:
            prompt_format = PROMPT_DICT["prompt_input"]
    else:
        prompt_format = PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def test():
    data = datasets.load_dataset("json", data_files=["./data/msra/instruct_data/train.json"], field="data")
    print(data)
    data = data.map(lambda x: extract_alpaca_dataset(x, "chinese_bloom"), remove_columns=["instruction"])
    for i in range(3):
        print(data["train"][i])


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    model_name: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        print(instances)
        sources = [example['input'] for example in instances]
        targets = [example['output'] for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len - 1 if self.model_name == "chatglm" else self.source_max_len,
            add_special_tokens=False,
            truncation=True,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len - 2,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        attention_mask = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):

            if not self.predict_with_generate:
                if self.model_name == "chatglm":
                    tokenized_source = tokenized_source + [self.tokenizer.convert_tokens_to_ids("[gMASK]")]
                    tokenized_target = [self.tokenizer.convert_tokens_to_ids("<sop>")] + tokenized_target + [
                        self.tokenizer.convert_tokens_to_ids("<eop>")]
                else:
                    tokenized_target = [self.tokenizer.bos_token_id] + tokenized_target + [self.tokenizer.eos_token_id]
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                attention_mask.append([1] * len(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    tmp_label = [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target)
                    labels.append(
                        tmp_label if self.model_name in ["chatglm", "chinese_bloom"] else torch.tensor(tmp_label))
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.tokenizer.padding_side == "right" and self.model_name != 'chinese_bloom':
            labels = pad_sequence(labels, batch_first=True,
                                  padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        else:
            if self.predict_with_generate:
                labels = None
            else:
                max_length = max([len(label) for label in labels])
                labels = [[IGNORE_INDEX] * (max_length - len(label)) + label for label in labels]
                labels = torch.tensor(labels)
                attention_mask = [[0] * (max_length - len(mask)) + mask for mask in attention_mask]
                attention_mask = torch.tensor(attention_mask)

        if self.model_name == "chatglm":
            data_dict = {
                'input_ids': input_ids,
            }
        elif self.model_name == "chinese_bloom":
            data_dict = {
                'input_ids': input_ids,
                "attention_mask": attention_mask,
            }
        elif self.model_name == "chinese_alpaca":
            data_dict = {
                'input_ids': input_ids,
            }
        else:
            data_dict = {
                'input_ids': input_ids,
                'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            }
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


def test_collator():
    data = datasets.load_dataset("json", data_files=["./data/msra/instruct_data/train.json"], field="data")
    data = data.map(lambda x: extract_alpaca_dataset(x, "chinese_alpaca"), remove_columns=["instruction"])
    data = data["train"]
    data = data[:1]
    input = data["input"]
    output = data["output"]
    data = [{"input": inp, "output": out} for inp, out in zip(input, output)]
    tokenizer = AutoTokenizer.from_pretrained("./model_hub/BELLE-7B-2M", trust_remote_code=True)
    collator = DataCollatorForCausalLM(tokenizer=tokenizer,
                                       source_max_len=128,
                                       target_max_len=64,
                                       train_on_source=False,
                                       predict_with_generate=False,
                                       model_name="chinese_alpaca")
    collator(data)


if __name__ == "__main__":
    # process_msra()
    # test()
    test_collator()