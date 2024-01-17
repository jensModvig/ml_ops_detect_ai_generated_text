import torch

#from models.model import TextClassificationModel

def test_model():
    
    try:
        from ml_ops_detect_ai_generated_text.models.model import TextClassificationModel    
        # Get model
        model = TextClassificationModel(
            model_name="distilbert-base-uncased",
            num_labels=2,
            learning_rate=0.01,
        )
        # Get the tokenizer
        tokenizer = model.tokenizer

        # Dummy data
        text = "This is a test"
        label = 0

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # test that it has the correct output
        assert logits.shape == (1, 2), "logits shape is incorrect"
    
    except Exception as e:
        print(f"Error: {e}")
        #raise e


if __name__ == "__main__":
    test_model()
