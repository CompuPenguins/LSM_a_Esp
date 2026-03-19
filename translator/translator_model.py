import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python translator_model.py "tu texto aquí"')
        sys.exit(1)

    splits = sys.argv[1].split(' ')
    first_token = splits[0]
    splits.pop(0)
    input_text = ' '.join(splits)
    dir = "./" + "esp_to_lsm" if  first_token == 'esp:' else "lsm_to_esp"
    print(dir)
    tokenizer = AutoTokenizer.from_pretrained(dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(dir)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
