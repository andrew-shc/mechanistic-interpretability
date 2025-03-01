# Use a pipeline as a high-level helper
from transformers import pipeline

def main():

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    pipe(messages)


if __name__ == "__main__":
    main()


