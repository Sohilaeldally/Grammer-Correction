from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "alnnahwi/gemma-3-1b-arabic-gec-v1"

# تحميل التوكنيزر والموديل
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# بايبلاين للتوليد
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def correct_sentence(sentence):
    output = pipe(sentence, max_new_tokens=64, do_sample=False)
    return output[0]["generated_text"]

# تجربة
wrong = "انا داهب الى المدسة"
print("Input :", wrong)
print("Output:", correct_sentence(wrong))
