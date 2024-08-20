import os
import re
import shutil
import pickle
from tqdm import tqdm


dir = "/Users/sean/Desktop/vm/datasets/archive"  
newDir = "/Users/sean/Desktop/vm/datasets/bigText"


if not os.path.exists(newDir):
    os.makedirs(newDir)


def extract_numeric_prefix(name):
    match = re.match(r'(\d+)', name)
    return int(match.group(1)) if match else float('inf')


dirc = 0
for childdir in sorted(os.listdir(dir), key=extract_numeric_prefix):
    for filename in os.scandir(os.path.join(dir, childdir)):
        if filename.is_file():
            dirc += 1

count = 0


pbar = tqdm(total=dirc, desc='Processing Files...', mininterval=1.0, miniters=1)


for childdir in sorted(os.listdir(dir), key=extract_numeric_prefix):
    childdir_path = os.path.join(dir, childdir)
    
    if os.path.isdir(childdir_path):
        
        files = sorted([f for f in os.listdir(childdir_path) if os.path.isfile(os.path.join(childdir_path, f))], key=extract_numeric_prefix)
        
        for filename in files:
            file_path = os.path.join(childdir_path, filename)
            
            
            total, used, free = shutil.disk_usage("/")
            free_gb = free / 1024**3  
            pbar.set_description(f"Available disk space is: {free_gb:.2f} GB")

            if free < 5 * 1024**3:  
                print("There's Less than 5GB of free disk space available. Stopping...")
                pbar.close()
                exit()

            try:
                filenam3 = os.path.basename(file_path)
                fileN, _ = os.path.splitext(filenam3)

                
                with open(file_path, 'rb') as f:
                    file = pickle.load(f)

                
                fileContent = ' '.join(str(file).split())
                pattern = r'[^\w\s\.,;:\""\'‘’`!?@#$%^&*\(\)\[\]{}+=<>-]'
                file2 = re.sub(pattern, ' ', fileContent).replace('  ', ' ').strip()

                
                file2 = re.sub(r'_(.*?)_', r'\1', file2)

                
                with open(os.path.join(newDir, f'{fileN}.txt'), 'w', encoding='utf-8') as f1:
                    f1.write(file2)

                count += 1
                pbar.update(1)
            except (pickle.UnpicklingError, IOError, OSError) as e:
                print(f"Error processing the file {file_path}: {e}")
                continue

pbar.close()
print(f'Final count is: {count}')


