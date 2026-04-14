from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForCausalLM

def create_simple_llm():
    model_name = "distilgpt2"
    generator = pipeline('text-generation', model=model_name, pad_token_id=50256)

    return generator

def generator_text(generator, prompt, max_length = 1000):
    result = generator(prompt, max_length=max_length, do_sample=True, temperature=0.7)

    return result[0]["generated_text"]

def interactive_demo():
    generator = create_simple_llm()

    while True:
        prompt = input("\nDigite: ")
        if prompt.lower() == 'quit':
            break

        response = generator_text(generator, prompt)
        print('\nResposta: ')
        print(response)

interactive_demo()