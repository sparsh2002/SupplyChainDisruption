from transformers import BertTokenizer, BertForSequenceClassification
import torch
model_path = 'path to model directory'  # Replace with the path to the directory where the model is saved
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
# Set the model to evaluation mode
model.eval()
def preprocess_text(text, tokenizer, max_length):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs


def evaluate(input_text):

    # Preprocess the text
    inputs = preprocess_text(input_text, tokenizer, max_length=128)

    # Set the device (GPU if available, else CPU)
    device = torch.device('cpu')

    # Move the inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

   

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label and probabilities
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    label_probabilities = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

    print(f"Predicted Label: {predicted_label}")
    print(f"Label Probabilities: {label_probabilities}")

    return predicted_label , label_probabilities

