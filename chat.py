import argparse
import json, os
import time
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None, type=str, required=True)
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str,
                    help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
args = parser.parse_args()

pprint(vars(args))
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
from peft import PeftModel

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400
)

# The prompt template below is taken from llama.cpp
# and is slightly different from the one used in training.
# But we find it gives better results

model_dict = {
    "chatglm": (AutoModel, AutoTokenizer),
    "alpaca": (LlamaForCausalLM, LlamaTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
}

prompt_dict = {
    "llama": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    ),
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    ),
    "chatglm": "{instruction}",
    "bloom": "Human: \n{instruction}\n\nAssistant: \n",
}

prompt_input = prompt_dict[args.model_name]

sample_data = ["为什么要减少污染，保护环境？"]


def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


if __name__ == '__main__':
    # load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model

    model_class, tokenizer_class = model_dict[args.model_name]

    start = time.time()
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        # torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        load_in_4bit=True,
        trust_remote_code=True if args.model_name == "chatglm" else False
    )
    end = time.time()
    print("加载模型耗时：{}分钟".format((end - start) / 60))

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path,
                                                trust_remote_code=True if args.model_name == "chatglm" else False)

    # model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    # tokenzier_vocab_size = len(tokenizer)
    # print(f"Vocab of the base model: {model_vocab_size}")
    # print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    # if model_vocab_size != tokenzier_vocab_size:
    #     assert tokenzier_vocab_size > model_vocab_size
    #     print("Resize model embeddings to fit tokenizer")
    #     base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model is not None:
        print("loading peft model")
        # , torch_dtype=load_type
        model = PeftModel.from_pretrained(base_model, args.lora_model, device_map='auto', )
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            print('=' * 85)
            print("+ 当前使用的模型是：{}".format(args.model_name))
            print('-' * 85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。")
            print('=' * 85)

            while True:
                raw_input_text = input("Input:")
                raw_input_text = str(raw_input_text)
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device) if args.model_name != "chatglm" else None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                if args.model_name == "chatglm":
                    s = s.cpu().numpy().tolist()
                    ind = len(s)
                    if tokenizer.bos_token_id in s:
                        ind = s.index(tokenizer.bos_token_id)
                    length = ind + 1
                else:
                    attention_mask = inputs["attention_mask"][0]
                    length = sum(attention_mask)
                output = tokenizer.decode(s[length:], skip_special_tokens=True)
                response = output
                print("Response: ", response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=example)
                else:
                    input_text = example
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device) if args.model_name != "chatglm" else None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                if args.model_name == "chatglm":
                    s = s.cpu().numpy().tolist()
                    ind = len(s)
                    if tokenizer.bos_token_id in s:
                        ind = s.index(tokenizer.bos_token_id)
                    length = ind + 1
                else:
                    attention_mask = inputs["attention_mask"][0]
                    length = sum(attention_mask)
                output = tokenizer.decode(s[length:], skip_special_tokens=True)
                response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")

                results.append({"Input": input_text, "Output": response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.predictions_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            with open(dirname + '/generation_config.json', 'w') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=2)