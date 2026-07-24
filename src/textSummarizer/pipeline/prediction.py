# from textSummarizer.config.configuration import ConfigurationManager
# from transformers import pipeline, AutoTokenizer


# class PredictionPipeline:
#     def __init__(self):
#         self.config = ConfigurationManager().get_model_evaluation_config()
        
    
#     def predict(self, text):
#         tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
#         gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
        
#         pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)
        
#         print("Dialogue:")
#         print(text)
        
#         output = pipe(text, **gen_kwargs)[0]["summary_text"]
        
#         print("\nModel Summary:")
#         print(output)
        
#         return output




from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load once at initialization (much better than loading on every request)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        # Generation parameters (kept close to your original ones)
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128,
            "min_length": 30,          # recommended for summarization
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=1024,          # adjust if your training used a different value
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        print("Dialogue:")
        print(text)

        # Generate
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode
        output = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        print("\nModel Summary:")
        print(output)

        return output
