from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig

generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,    
)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(instruction, input=None, generation_config = None):
    prompts = generate_prompt(instruction, input)
    inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    results = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )

    for i,s in enumerate(results.sequences):
        gen = tokenizer.decode(s)
        
        given = prompts
        gen = gen.split(given)[1]

        print('\033[34m' + prompts+ '\033[0m' + '\033[32m' + gen+ '\033[0m')
        print("\n==================================\n")


if __name__=="__main__":
    tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
    print(f'Loaded tokenizer')
    model = LlamaForCausalLM.from_pretrained('/home/seungyoun/stanford_alpaca/ckpt/alpaca_7B/checkpoint-38000',
                                             device_map="auto")
    print(f'Loaded model')

    evaluate(input("Instruction: "))



