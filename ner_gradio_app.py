import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import gradio as gr

# Load the saved model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./ner_model")
tokenizer = AutoTokenizer.from_pretrained("./ner_model")

# Define the labels (update this based on your training labels)
label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# Function to perform NER prediction
def ner_prediction(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict using the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted labels for each token
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Decode tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [label_names[label] for label in predictions[0].cpu().numpy()]

    # Merge subwords and format output
    entities = []
    current_entity = ""
    current_label = None

    for token, label in zip(tokens, predicted_labels):
        if token.startswith("##"):
            current_entity += token[2:]
        else:
            if current_entity and current_label != "O":
                entities.append((current_entity, current_label))
            current_entity = token
            current_label = label
    if current_entity and current_label != "O":
        entities.append((current_entity, current_label))

    return {entity[0]: entity[1] for entity in entities}

# Define Gradio interface
iface = gr.Interface(
    fn=ner_prediction,
    inputs="text",
    outputs="json",
    title="Named Entity Recognition",
    description="Enter a sentence to extract named entities (Person, Organization, Location, etc.)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()