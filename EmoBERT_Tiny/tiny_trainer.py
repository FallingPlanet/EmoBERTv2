# Standard library imports (if any)
import sys
sys.path.append(r'D:\Users\WillR\VsCodeProjects\Natural Language Processing\LowLatencyTCES\tinyEmoBERT')
import os
# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
# Local application/library s
from FallingPlanet.orbit.utils.Metrics import AdvancedMetrics
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard
import torchmetrics
from tqdm import tqdm
from FallingPlanet.orbit.utils.callbacks import EarlyStopping
from FallingPlanet.orbit.models import BertFineTuneTiny
from itertools import islice

class Classifier:
    def __init__(self,model, device, num_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.loss_criterion = CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)
        
        
        self.accuracy = torchmetrics.Accuracy(num_classes=num_labels, task='multiclass').to(device)
        self.precision = torchmetrics.Precision(num_classes=num_labels, task='multiclass').to(device)
        self.recall = torchmetrics.Recall(num_classes=num_labels, task='multiclass').to(device)
        self.f1= torchmetrics.F1Score(num_classes=num_labels, task = 'multiclass').to(device)
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_labels,task = 'multiclass').to(device)
        self.top2_acc = torchmetrics.Accuracy(top_k=2, num_classes=num_labels,task='multiclass').to(device)
        
    def compute_loss(self,logits, labels):
        loss = self.loss_criterion(logits,labels)
        return loss
    
    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc = f"Training Epoch {epoch}")
        
        for batch in pbar:
            input_ids, attention_masks, labels = [x.to(self.device) for x in batch]
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids,attention_masks=attention_masks)
            loss = self.compute_loss(outputs.logits, labels)
            loss.backward()
            
            optimizer.step()
            total_loss = loss.item()
           
           
            accuracy = self.accuracy(outputs.logits.argmax(dim=1), labels)
            precision = self.precision(outputs.logits.argmax(dim=1), labels)
            recall = self.recall(outputs.logits.argmax(dim=1), labels)
            f1 = self.f1(outputs.logits, labels)
            mcc = self.mcc(outputs.logits.argmax(dim=1), labels)
            
            # Update tqdm description with current loss and metrics
            pbar.set_postfix({
                'Loss': f'{total_loss / (pbar.n + 1):.4f}',
                'Acc': f'{accuracy:.4f}',
                'Prec': f'{precision:.4f}',
                'Rec': f'{recall:.4f}',
                'F1': f'{f1:.4f}',
                'MCC': f'{mcc:.4f}'
        })
            pbar.update(1)
             # Log metrics to TensorBoard
            self.writer.log_scalar('Training/Loss', loss, epoch)
            self.writer.log_scalar('Training/Accuracy', accuracy, epoch)
            self.writer.log_scalar('Training/Precision', precision, epoch)
            self.writer.log_scalar('Training/Recall', recall, epoch)
            self.writer.log_scalar('Training/F1', f1, epoch)
            self.writer.log_scalar('Training/MCC', mcc, epoch)

        pbar.close()
        avg_train_loss = total_loss / len(dataloader)
        self.writer.log_scalar('Training/Loss', avg_train_loss, epoch)  
    def val_step(self, dataloader, epoch):
        pbar = tqdm(dataloader,desc=f"Validation {epoch}")
        self.model(eval)
        total_loss = 0.0
        
        with torch.no_grad():
                    for batch in pbar:
                        input_ids, attention_masks, labels = [x.to(self.device) for x in batch]
                        
                     
                        outputs = self.model(input_ids,attention_masks=attention_masks)
                        loss = self.compute_loss(outputs.logits, labels)
                        total_loss = loss.item()
                    
                    
                        accuracy = self.accuracy(outputs.logits.argmax(dim=1), labels)
                        precision = self.precision(outputs.logits.argmax(dim=1), labels)
                        recall = self.recall(outputs.logits.argmax(dim=1), labels)
                        f1 = self.f1(outputs.logits, labels)
                        mcc = self.mcc(outputs.logits.argmax(dim=1), labels)
                        
                        # Update tqdm description with current loss and metrics
                        pbar.set_postfix({
                            'Loss': f'{total_loss / (pbar.n + 1):.4f}',
                            'Acc': f'{accuracy:.4f}',
                            'Prec': f'{precision:.4f}',
                            'Rec': f'{recall:.4f}',
                            'F1': f'{f1:.4f}',
                            'MCC': f'{mcc:.4f}'
                    })
                        pbar.update(1)
                        # Log metrics to TensorBoard
                        self.writer.log_scalar('Validation/Loss', loss, epoch)
                        self.writer.log_scalar('Validation/Accuracy', accuracy, epoch)
                        self.writer.log_scalar('Validation/Precision', precision, epoch)
                        self.writer.log_scalar('Validation/Recall', recall, epoch)
                        self.writer.log_scalar('Validation/F1', f1, epoch)
                        self.writer.log_scalar('Validation/MCC', mcc, epoch)

                    pbar.close()
                    avg_val_loss = total_loss / len(dataloader)
                    self.writer.log_scalar('Validation/Loss', avg_val_loss, epoch)
                    return avg_val_loss  
    def test_step(self, dataloader):
        self.model.eval()
        aggregated_metrics = {}
        
        aggregated_metrics = {
            'accuracy':  0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'top_2_acc': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Testing")
            
            for batch in pbar:
                input_ids, attention_masks, labels = [x.to(self.device) for x in batch]
                
                outputs = self.model(input_ids,attention_masks,labels)
                accuracy = self.accuracy(outputs.logits.argmax(dim=1), labels)
                precision = self.precision(outputs.logits.argmax(dim=1), labels)
                recall = self.recall(outputs.logits.argmax(dim=1), labels)
                f1 = self.f1(outputs.logits, labels)
                mcc = self.mcc(outputs.logits.argmax(dim=1), labels)
                top_2_acc = self.top2_acc(outputs.logits,labels).item()
                
                
                pbar.set_postfix({
                "Accuracy": accuracy,
                'top_2_acc': top_2_acc
            })

                # Update aggregated metrics
                aggregated_metrics['accuracy'] += accuracy
              
                aggregated_metrics['precision'] += precision
               
                aggregated_metrics['recall'] += recall
              
                aggregated_metrics['f1'] += f1
                
                aggregated_metrics['mcc'] += mcc
                
                aggregated_metrics['top_2_acc'] += top_2_acc

                pbar.update(1)

            pbar.close()
        num_batches = len(dataloader)
        for key in aggregated_metrics:
            aggregated_metrics[key] /= num_batches

        return aggregated_metrics

    
def main(mode = "full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
 
    emotion_data_train = torch.load(r"E:\text_datasets\saved\train_emotion_no_batch.pt")
    emotion_data_val = torch.load(r"E:\text_datasets\saved\val_emotion_no_batch.pt")
    emotion_data_test = torch.load(r"E:\text_datasets\saved\test_emotion_no_batch.pt")
    
    

   
    
    
    dataloader_train = DataLoader(emotion_data_train, batch_size=256, shuffle=True)
    dataloader_val = DataLoader(emotion_data_val, batch_size=256)
    dataloader_test = DataLoader(emotion_data_test, batch_size=256)
 
    NUM_EMOTION_LABELS = 9
    LOG_DIR = r"D:\Users\WillR\VsCodeProjects\Natural Language Processing\tinyEmoBERT\logging"
    

    model = BertFineTuneTiny(num_tasks=1, num_labels=9)
    optimizer = torch.optim.AdamW(model.parameters(),lr =1e-5, weight_decay=1e-6)
    classifier = Classifier(model, device,  NUM_EMOTION_LABELS, LOG_DIR)

    if mode in ["train", "full"]:
        # Your training logic here
        early_stopping = EarlyStopping(patience=25, min_delta=0.0001)  # Initialize Early Stopping
        num_epochs = 25
        for epoch in range(num_epochs):
            classifier.train_step(dataloader_train, dataloader_train, optimizer, epoch)
            val_loss = classifier.val_step(dataloader_val, epoch)

            if early_stopping.step(val_loss, classifier.model):
                print("Early stopping triggered. Restoring best model weights.")
                classifier.model.load_state_dict(early_stopping.best_state)
                break

        if early_stopping.best_state is not None:
            torch.save(early_stopping.best_state, 'EmoBERTv2-tiny.pth')

    if mode in ["test", "full"]:
        if os.path.exists('EmoBERTv2-tiny.pth'):
            classifier.model.load_state_dict(torch.load('EmoBERTv2-tiny.pth'))
    # Assuming you have test_step implemented in classifier
    test_results = classifier.test_step(dataloader_test)
    print("Test Results:", test_results)


if __name__ == "__main__":
    main(mode="test")  # or "train" or "test"  