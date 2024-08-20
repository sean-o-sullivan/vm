
import ast
import itertools
import os
import sys
from math import comb
from random import sample
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import Dataset, random_split
from tqdm import tqdm


torch.multiprocessing.set_sharing_strategy('file_system')


device = torch.device("cpu")

print("entering classes.py")




class EmbeddingNet(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=2, dim_feedforward=32, dropout=0.43547725274485294,
                 new_d_model=64, new_nhead=16, new_dim_feedforward=32, new_dropout=0.39472428838464607):
        super(EmbeddingNet, self).__init__()
        self.linear1 = nn.Linear(input_size, d_model)
        self.batch_norm1 = nn.BatchNorm1d(d_model)

        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1)
        self.linear2 = nn.Linear(d_model, d_model)
        self.batch_norm2 = nn.BatchNorm1d(d_model)

        
        new_encoder_layer = nn.TransformerEncoderLayer(d_model=new_d_model, nhead=new_nhead,
                                                       dim_feedforward=new_dim_feedforward, dropout=new_dropout, batch_first=True)
        self.new_transformer_encoder = nn.TransformerEncoder(
            new_encoder_layer, num_layers=1)
        self.linear3 = nn.Linear(d_model, d_model)
        self.batch_norm3 = nn.BatchNorm1d(d_model)

    def get_output_dim(self):
        
        return self.linear3.out_features

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.transformer_encoder(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)

        
        x = self.new_transformer_encoder(x)

        x = self.linear3(x)
        x = self.batch_norm3(x)

        return x


class SiameseNetwork(nn.Module):
    def __init__(self, input_size, new_nhead, new_dim_feedforward, new_dropout):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_net = EmbeddingNet(input_size)

    def forward(self, input1, input2):
        output1 = self.embedding_net(input1)
        output2 = self.embedding_net(input2)
        return torch.abs(output1 - output2)


class ClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)



class EmbeddingPairsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context_embedding = ast.literal_eval(row['context_embedding'])
        check_embedding = ast.literal_eval(row['check_embedding'])
        label = row['label']
        return (torch.tensor(context_embedding, dtype=torch.float32),
                torch.tensor(check_embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


class CustomDataset_fromvcsv(Dataset):
    def __init__(self, filler_csv_paths, percentages, focus_csv, batch_size):
        assert abs(sum(percentages) - 1) < 1e-6, "Percentages must sum up to 1"

        
        total_samples_needed = batch_size - 1

        
        initial_sample_counts = [
            int(percentage * total_samples_needed) for percentage in percentages]

        
        shortfall = total_samples_needed - sum(initial_sample_counts)

        
        for i in sorted(range(len(initial_sample_counts)), key=lambda i: -percentages[i]):
            if shortfall > 0:
                initial_sample_counts[i] += 1
                shortfall -= 1
            else:
                break

        filler_data_frames = []
        for csv_path, sample_count in zip(filler_csv_paths, initial_sample_counts):
            temp_data = pd.read_csv(csv_path)
            if sample_count > 0:
                sampled_data = temp_data.sample(
                    n=min(sample_count, len(temp_data)), replace=False)
                filler_data_frames.append(sampled_data)

        filler_data = pd.concat(filler_data_frames, ignore_index=True)

        assert len(filler_data) >= total_samples_needed, f"Filler data must contain at least {
            total_samples_needed} samples, got {len(filler_data)}."

        filler_data['is_focus'] = False

        
        focus_data = pd.read_csv(focus_csv)
        print("Focus data has been loaded!")
        focus_data['is_focus'] = True

        self.fixed_filler_samples = filler_data.iloc[:total_samples_needed]
        self.focus_data = focus_data
        self.batch_size = batch_size
        self.focus_len = len(focus_data)
        self.total_len = self.focus_len * batch_size
        print("we are at the end of the init function!")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        focus_index = idx // self.batch_size
        within_batch_index = idx % self.batch_size

        if within_batch_index == self.batch_size - 1:
            row = self.focus_data.iloc[focus_index]
        else:
            row = self.fixed_filler_samples.iloc[within_batch_index]

        context_embedding = np.array(
            ast.literal_eval(row['context_embedding']))
        check_embedding = np.array(ast.literal_eval(row['check_embedding']))
        label = row['label']
        is_focus = row['is_focus']

        return (torch.tensor(context_embedding, dtype=torch.float32),
                torch.tensor(check_embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(is_focus, dtype=torch.bool))
    



class oldCustomDataset(Dataset):
    def __init__(self, filler_csv_paths, percentages, focus_context_embedding, focus_check_embedding, batch_size):
        assert abs(sum(percentages) - 1) < 1e-6, "Percentages must sum up to 1"

        self.batch_size = batch_size
        total_samples_needed = batch_size - 1

        initial_sample_counts = [int(percentage * total_samples_needed) for percentage in percentages]

        shortfall = total_samples_needed - sum(initial_sample_counts)
        for i in sorted(range(len(initial_sample_counts)), key=lambda i: -percentages[i]):
            if shortfall > 0:
                initial_sample_counts[i] += 1
                shortfall -= 1

        filler_data_frames = []
        for csv_path, sample_count in zip(filler_csv_paths, initial_sample_counts):
            temp_data = pd.read_csv(csv_path)
            if sample_count > 0:
                sampled_data = temp_data.sample(n=min(sample_count, len(temp_data)), replace=False)
                filler_data_frames.append(sampled_data)

        filler_data = pd.concat(filler_data_frames, ignore_index=True)
        assert len(filler_data) >= total_samples_needed, f"Filler data must contain at least {total_samples_needed} samples."

        filler_data['is_focus'] = False

        
        self.focus_sample = {
            'context_embedding': focus_context_embedding,
            'check_embedding': focus_check_embedding,
            'label': None,  
            'is_focus': True
        }

        self.fixed_filler_samples = filler_data.iloc[:total_samples_needed]
        self.total_len = self.batch_size  

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        within_batch_index = idx % self.batch_size

        if within_batch_index == self.batch_size - 1:
            
            row = self.focus_sample
        else:
            
            row = self.fixed_filler_samples.iloc[within_batch_index].to_dict()

        context_embedding = np.array(ast.literal_eval(row['context_embedding']))
        check_embedding = np.array(ast.literal_eval(row['check_embedding']))
        
        
        label = row.get('label', None)
        
        is_focus = row['is_focus']

        
        label_tensor = torch.tensor(label, dtype=torch.float32) if label is not None else torch.tensor(-1, dtype=torch.float32)

        return (torch.tensor(context_embedding, dtype=torch.float32),
                torch.tensor(check_embedding, dtype=torch.float32),
                label_tensor,
                torch.tensor(is_focus, dtype=torch.bool))




class CustomDataset(Dataset):
    def __init__(self, filler_csv_paths, percentages, focus_context_embedding, focus_check_embedding, batch_size):
        assert abs(sum(percentages) - 1) < 1e-6, "Percentages must sum up to 1"

        self.batch_size = batch_size
        total_samples_needed = batch_size - 1

        initial_sample_counts = [int(percentage * total_samples_needed) for percentage in percentages]

        shortfall = total_samples_needed - sum(initial_sample_counts)
        for i in sorted(range(len(initial_sample_counts)), key=lambda i: -percentages[i]):
            if shortfall > 0:
                initial_sample_counts[i] += 1
                shortfall -= 1

        filler_data_frames = []
        for csv_path, sample_count in zip(filler_csv_paths, initial_sample_counts):
            temp_data = pd.read_csv(csv_path)
            if sample_count > 0:
                sampled_data = temp_data.sample(n=min(sample_count, len(temp_data)), replace=False)
                filler_data_frames.append(sampled_data)

        filler_data = pd.concat(filler_data_frames, ignore_index=True)
        assert len(filler_data) >= total_samples_needed, f"Filler data must contain at least {total_samples_needed} samples."

        filler_data['is_focus'] = False

        
        self.focus_sample = {
            'context_embedding': focus_context_embedding,
            'check_embedding': focus_check_embedding,
            'label': None,
            'is_focus': True
        }

        self.fixed_filler_samples = filler_data.iloc[:total_samples_needed]
        self.total_len = self.batch_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        within_batch_index = idx % self.batch_size

        if within_batch_index == self.batch_size - 1:
            row = self.focus_sample
        else:
            row = self.fixed_filler_samples.iloc[within_batch_index].to_dict()

        
        try:
            context_embedding = np.array(ast.literal_eval(row['context_embedding']))
            check_embedding = np.array(ast.literal_eval(row['check_embedding']))
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing embeddings: {e}")
            print(f'Check embedding: {row['check_embedding']}, Context embedding: {row['context_embedding']}')
            context_embedding = np.zeros((58,))  
            check_embedding = np.zeros((58,))

        label = row.get('label', None)
        is_focus = row['is_focus']

        label_tensor = torch.tensor(label, dtype=torch.float32) if label is not None else torch.tensor(-1, dtype=torch.float32)

        return (torch.tensor(context_embedding, dtype=torch.float32),
                torch.tensor(check_embedding, dtype=torch.float32),
                label_tensor,
                torch.tensor(is_focus, dtype=torch.bool))
    


def evaluate(model, classifier, dataloader, criterion, device):
    model.eval()  
    running_loss = 0.0
    total_focus_samples = 0  
    marked_labels = []  
    marked_predictions = []  

    correct_predictions = 0
    total_predictions = 0

    for i, (context, check, label, is_focus) in enumerate(dataloader):
        context, check, label, is_focus = context.to(device), check.to(
            device), label.to(device), is_focus.to(device)

        with torch.no_grad():  
            output = model(context, check)
            scores = classifier(output)
        sys.stdout.write(f'\rValidation {i}/{len(dataloader)}')
        sys.stdout.flush()  

        
        
        focus_indices = is_focus.bool().squeeze()
        if focus_indices.any():
            loss = criterion(scores[focus_indices],
                             label[focus_indices].unsqueeze(1))
            running_loss += loss.item()
            
            predictions = (scores > 0.3).float()
            correct_predictions += (predictions[focus_indices]
                                    == label[focus_indices].unsqueeze(1)).sum().item()
            total_predictions += focus_indices.sum().item()

            
            marked_labels.extend(label[focus_indices].cpu().numpy())
            marked_predictions.extend(predictions[focus_indices].cpu().numpy())

            total_focus_samples += focus_indices.sum().item()
        
        running_accuracy = 100. * correct_predictions / total_predictions

        
        sys.stdout.write(
            f'\rBatch: {i+1}/{len(dataloader)}, Running Accuracy: {running_accuracy:.2f}%')
        sys.stdout.flush()  

    
    marked_labels = np.array(marked_labels)
    marked_predictions = np.array(marked_predictions)

    
    overall_accuracy = accuracy_score(marked_labels, marked_predictions)
    precision = precision_score(
        marked_labels, marked_predictions, zero_division=0)
    recall = recall_score(marked_labels, marked_predictions, zero_division=0)
    f1 = f1_score(marked_labels, marked_predictions, zero_division=0)
    mcc = matthews_corrcoef(marked_labels, marked_predictions) if len(
        np.unique(marked_labels)) > 1 else 0  

    
    average_loss = running_loss / total_focus_samples if total_focus_samples > 0 else 0

    return average_loss, overall_accuracy, precision, recall, f1, mcc





def evaluate2(model, classifier, dataloader, criterion, device):
    print('Entered the evaluation function.')
    model.eval()

    running_loss_all = 0.0
    correct_predictions_all = 0
    total_predictions_all = 0
    labels_all = []
    predictions_all = []
    all_predictions = []  

    focus_predictions = []  

    for _, (context, check, label, is_focus) in enumerate(dataloader):
        context, check, label, is_focus = context.to(device), check.to(device), label.to(device), is_focus.to(device)
        with torch.no_grad():
            output = model(context, check)
            scores = classifier(output)
        
        loss = criterion(scores, label.unsqueeze(1))
        predictions = (scores > 0.666).float()

        running_loss_all += loss.item()
        correct_predictions_all += (predictions == label.unsqueeze(1)).sum().item()
        total_predictions_all += label.size(0)
        labels_all.extend(label.cpu().numpy())
        predictions_all.extend(predictions.cpu().numpy())

        
        all_predictions.extend(scores.squeeze().cpu().numpy())

        focus_indices = is_focus.bool().squeeze()
        if focus_indices.any():
            focus_predictions.extend(scores[focus_indices].cpu().numpy())

    accuracy_all = 100. * correct_predictions_all / total_predictions_all if total_predictions_all > 0 else 0
    print(f'\nOverall Accuracy (All Samples): {accuracy_all:.2f}%')

    if focus_predictions:
        for i, prediction in enumerate(focus_predictions):
            try:
                
                print(f'\nFocus Prediction (Focus Pair {i+1}): {float(prediction):.2f}')
            except ValueError as e:
                
                print(f"\nError formatting focus prediction {i+1}: {prediction}. Error: {e}")
                
                print(f"Type of problematic value: {type(prediction)}")

        
        try:
            focus_predictions_mean = np.mean(focus_predictions)
            print(f'\nMean Focus Prediction (Focus Pairs): {focus_predictions_mean:.2f}')
        except Exception as e:
            print(f"\nError calculating mean of focus predictions. Error: {e}")
    else:
        print("\nNo focus predictions were found.")


    
    
    
    
    
    
    


    
    return f'{float(prediction):.3f}'






print("exiting classes.py")
