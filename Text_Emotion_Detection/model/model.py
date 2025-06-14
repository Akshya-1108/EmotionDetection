import torch 
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionClassifier:
    def __init__(self,model):
        self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.eval()

    def predict(self, text):
        input = self.tokenizer(text, truncation=True, return_tensors= "pt", padding= True).to(self.device)

        with torch.no_grad():
            logits = self.model(**input).logits
            
        prob = F.softmax(logits, dim =1)

        pred_idx = prob.argmax(dim = 1).item()
        
        labels = self.model.config.id2label
        final_label = labels[pred_idx]
        
        return final_label
    
        
        
    