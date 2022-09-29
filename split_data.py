import os
import random
import shutil

random.seed(0)

data_root = '.../inputs/data'
data_raw = os.path.join(data_root, 'raw')
data_train = os.path.join(data_root, 'train')
data_val  = os.path.join(data_root, 'val')
data_test  = os.path.join(data_root, 'test')

test_size = 0.15
val_size = 0.2 

def creat_copy(root, files):
    sub_folder = os.path.join(root, folder) 
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    for file in files:
        shutil.copyfile(file, os.path.join(sub_folder, os.path.basename(file)))

for folder in os.listdir(data_raw): 
    if folder[0]!='.':
        print('-'*10)
        print('Folder:', folder) 
        files_list = []
        full_folder = os.path.join(data_raw, folder) 
        
        for file in os.listdir(full_folder):
            file_path = os.path.join(full_folder, file)
            files_list.append(file_path)  
        print(f'Total file in folder {folder} =', len(files_list))
        
        num_test_files = int(test_size*len(files_list))
        num_val_files = int(val_size*len(files_list))

        random.shuffle(files_list)
        # Get list files of train, test, val
        test_files = files_list[:num_test_files]
        val_files = files_list[-num_val_files:]
        train_files = files_list[num_test_files:-num_val_files]

        print('Num files for test =', len(test_files))
        print('Num files for val =', len(val_files))
        print('Num files for train =', len(train_files))

        # #check unique
        # print(f'check for unquie in{folder}')
        # for file in train_files:
        #     if file in test_files: 
        #         print(file, 'in test files')
        #     if file in val_files:
        #         print(file, 'in val file')

        ##creat foler and copy
        # test
        print(f'Copy to test of {folder}...')
        creat_copy(data_test, test_files)
        # val
        print(f'Copy to val of {folder}...')
        creat_copy(data_val, val_files)
        # train
        print(f'Copy to train of {folder}...')
        creat_copy(data_train, train_files)

print('Done!')





