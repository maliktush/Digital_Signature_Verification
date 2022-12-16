import csv
import os
import random
import glob
import itertools
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def sample_format_generator(original_original,signer_number,label):

    new_original_original = []

    for pair in original_original:
        new_original_original.append([int(signer_number),pair[0],pair[1],label])
    
    return new_original_original
    

def partition_function(signers,original_original_pair,original_forged_pair):

    samples = []

    for signer_number in signers:
        original_forged_subset_pair = random.sample(original_forged_pair,len(original_original_pair)) 

        original_original = sample_format_generator(original_original_pair,int(signer_number),1)

        samples.extend(original_original)

        original_forged = sample_format_generator(original_forged_subset_pair,int(signer_number),0)

        samples.extend(original_forged)
    
    return samples


def write_csv(file_path, samples):
    with open(file_path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(samples)


def prepare_dataset_CEDAR(M,K,random_state = 0,data_directory='data/CEDAR'):

    def paths_from_samples(samples):

        new_samples = []

        for sample in samples:
            signer_number,signature_1,signature_2,label = sample
            if label == 1:
                signature_1 = os.path.join(data_directory, 'full_org', f'original_{signer_number}_{signature_1}.png')
                signature_2 = os.path.join(data_directory, 'full_org', f'original_{signer_number}_{signature_2}.png')
            else:
                signature_1 = os.path.join(data_directory, 'full_org', f'original_{signer_number}_{signature_1}.png')
                signature_2 = os.path.join(data_directory, 'full_forg', f'forgeries_{signer_number}_{signature_2}.png')
            
            new_samples.append((signature_1,signature_2,label))
            
        return new_samples

    random.seed(random_state)
    signers = list(range(1, K+1))
    original_signs = 24
    forged_signs = 24

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    original_original_pair = list(itertools.combinations(range(1, original_signs+1), 2))
    original_forged_pair = list(itertools.product(range(1, original_signs+1), range(1, forged_signs+1)))

    train_samples = partition_function(train_signers, original_original_pair, original_forged_pair)
    train_samples = paths_from_samples(train_samples)
    write_csv(os.path.join(data_directory, 'train.csv'), train_samples)
    test_samples = partition_function(test_signers, original_original_pair, original_forged_pair)
    test_samples = paths_from_samples(test_samples)
    write_csv(os.path.join(data_directory, 'test.csv'), test_samples)



def prepare_Bengali(M: int, K: int, random_state=0, data_directory='data/BHSig260/Bengali'):

    def paths_from_samples(samples):

        new_samples = []

        for sample in samples:
            signer_number,signature_1,signature_2,label = sample
            signer_number = int(signer_number)
            signature_1 = int(signature_1)
            signature_2 = int(signature_2)
            if label == 1:
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_1:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_1:02d}.tif')
                             
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_2:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_2:02d}.tif')
                        

            else:
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-G-{signature_1:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-G-{signature_1:02d}.tif')
                    
            
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-F-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number}-F-{signature_2:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-F-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'B-S-{signer_number:03d}-F-{signature_2:02d}.tif')

            new_samples.append((signature_1,signature_2,label))
            
        return new_samples


        

    random.seed(random_state)
    signers = list(range(1, K+1))
    original_signs = 24
    forged_signs = 30

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    original_original_pair = list(itertools.combinations(range(1, original_signs+1), 2))
    original_forged_pair = list(itertools.product(range(1, original_signs+1), range(1, forged_signs+1)))

    train_samples = partition_function(train_signers, original_original_pair, original_forged_pair)
    train_samples = paths_from_samples(train_samples)
    write_csv(os.path.join(data_directory, 'train.csv'), train_samples)
    test_samples = partition_function(test_signers, original_original_pair, original_forged_pair)
    test_samples = paths_from_samples(test_samples)
    write_csv(os.path.join(data_directory, 'test.csv'), test_samples)


    
    



def prepare_Hindi(M: int, K: int, random_state=0, data_directory='data/BHSig260/Bengali'):

    def paths_from_samples(samples):

        new_samples = []

        for sample in samples:
            signer_number,signature_1,signature_2,label = sample
            signer_number = int(signer_number)
            signature_1 = int(signature_1)
            signature_2 = int(signature_2)
            if label == 1:
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_1:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_1:02d}.tif')
                             
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_2:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_2:02d}.tif')
                        

            else:
                
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-G-{signature_1:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_1:02d}.tif')) 
                    file.close()
                    signature_1 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-G-{signature_1:02d}.tif')
                    
            
                try:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-F-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number}-F-{signature_2:02d}.tif')
                    
                except:
                    file = open(os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-F-{signature_2:02d}.tif')) 
                    file.close()
                    signature_2 = os.path.join(data_directory, f'{signer_number:03d}', f'H-S-{signer_number:03d}-F-{signature_2:02d}.tif')

            new_samples.append((signature_1,signature_2,label))
            
        return new_samples

    random.seed(random_state)
    signers = list(range(1, K+1))
    original_signs = 24
    forged_signs = 30

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    original_original_pair = list(itertools.combinations(range(1, original_signs+1), 2))
    original_forged_pair = list(itertools.product(range(1, original_signs+1), range(1, forged_signs+1)))

    train_samples = partition_function(train_signers, original_original_pair, original_forged_pair)
    train_samples = paths_from_samples(train_samples)
    write_csv(os.path.join(data_directory, 'train.csv'), train_samples)
    test_samples = partition_function(test_signers, original_original_pair, original_forged_pair)
    test_samples = paths_from_samples(test_samples)
    write_csv(os.path.join(data_directory, 'test.csv'), test_samples)

if __name__ == "__main__":

    data_directory = 'data/BHSig260/Hindi/123/'
    for signature_number in range(1,25):
        old_file = os.path.join(data_directory, f'H-S-{str(124)}-G-{signature_number:02d}.tif')
        new_file = os.path.join(data_directory, f'H-S-{str(123)}-G-{signature_number:02d}.tif')
        os.rename(old_file,new_file)   

    print('Preparing CEDAR dataset..')
    prepare_dataset_CEDAR(M=50, K=55)
    print('Preparing Bengali dataset..')
    prepare_Bengali(M=50, K=100, data_directory='data/BHSig260/Bengali')
    print('Preparing Hindi dataset..')
    prepare_Hindi(M=100, K=160, data_directory='data/BHSig260/Hindi')
    print('Done')
  