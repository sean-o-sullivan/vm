import torch, os
import torch.nn as nn
from classes import *
from torch.utils.data import DataLoader


def predict_author(focus_context_embedding, focus_check_embedding):
    print("starting function: predict_author() in decision.py")
    
    input_size = 58
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_nhead = 8  
    new_dim_feedforward = 32  
    new_dropout = 0.39472428838464607

    siamese_net = SiameseNetwork(
        input_size, new_nhead, new_dim_feedforward, new_dropout).to(device)
    output_dim = siamese_net.embedding_net.get_output_dim()
    classifier_net = ClassifierNet(output_dim).to(
        device)  
    
    
    new_checkpoint_path = "model.pth"
    print('loading model')

    new_checkpoint = torch.load(new_checkpoint_path, map_location=device)
    print('model loaded')
    
    siamese_net.load_state_dict(new_checkpoint['siamese_model_state_dict'])
    
    classifier_net.load_state_dict(new_checkpoint['classifier_model_state_dict'])

    
    
    
    

    
    

    
    

   
   
   
   
   
   
   
   
   
   
   
   



 
 
 


    csv_files_dir = r'/Users/sean/Desktop/vscode/app/fillerCsvs'

    filler_csv_paths = [
        "5_3.csv",
        "FGPT4-2.csv",
        "4-2_2.csv",
        "4-2_1.csv",
        "3-1.5-3.csv",
        "GPT4-1_3.csv",
        "321KGPT3-2-3.csv",
        "VTL10_Gen2_test.csv",
        "VTL10_Gen3_test.csv",
        "VTL20_Gen2_t.csv",
        "VTL20_Gen3_t.csv",
        "VTL20_Gen4_t.csv",
        "5_3.csv",
        "FGPT4-2.csv",
        "4-2_2.csv",
        "4-2_1.csv",
        "3-1.5-3.csv",
        "GPT4-1_3.csv",
        "321KGPT3-2-3.csv",
        "VTL10_Gen2_test.csv",
        "VTL10_Gen3_test.csv",
        "VTL20_Gen2_t.csv",
        "VTL20_Gen3_t.csv",
        "VTL20_Gen4_t.csv",
        "VTL20_Gen5_t.csv"

    ]

    
    filler_csv_paths = [os.path.join(csv_files_dir, file_name) for file_name in filler_csv_paths]

    e = len(filler_csv_paths)
    print(e)
    percentages = [1/e for i in range(0,e)]
    print(percentages)

    
    

    batch_size = 1024  

    print("Creating the custom dataset now")
    custom_dataset = CustomDataset(filler_csv_paths, percentages, focus_context_embedding, focus_check_embedding, batch_size)
    print("Now creating the test dataloader")

    test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    print("Created the test loader.")

    print("now I am about to try iterate through the test loader we just made")
    
    np.set_printoptions(threshold=np.inf)

    
        
        
        
        
        
        
        
        
        

    criterion = nn.MSELoss()

    
    
    print("just about to predict authorship in decision.py")
    
    
    
    focus_prediction = evaluate2(siamese_net, classifier_net, test_loader, criterion, device)

    
    
    return f"{focus_prediction}"