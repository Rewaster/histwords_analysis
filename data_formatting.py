import numpy as np
import glob
import os
import pandas as pd
import pickle

path_german = 'D:\\ger-all\\sgns\\*'
path_english = 'D:\\eng-all\\sgns\\*'
path_french = 'D:\\fre-all\\sgns\\*'
path_chinese = 'D:\\cmn-all\\sgns\\*'

# path_COHA = './Embeddings/coha-lemma_sgns/sgns/*'
# Output the word vectors from https://nlp.stanford.edu/projects/histwords/ in the format we need 
# (each decade in a separate file, where each line starts consists of the word followed by the embedding)
def decade_to_txt(path_all):
    for path in path_all: 
        files = glob.glob(path)
        for decade in range(1800, 2001, 10):
            decade_files = [filename for filename in files if str(decade) in filename]
            if decade_files:
                vocab_file = decade_files[0]
                vector_file = decade_files[1]
    
                dataset = str(path).split('\\')[2]
                language = str(path).split('\\')[1]
                drive = str(path).split('\\')[0]
    
                vector_array = np.load(vector_file)
    
                with open(vocab_file, 'rb') as f:
                    vocab_list = pickle.load(f)
    
                output_file = os.path.join(drive, '\\Data\\IndividualDecades', language + '-' + dataset, str(decade)+'s.txt')
                with open(output_file, 'w', encoding="utf-8") as o:
                    # first line contains shape of vectors
                    o.write(str(vector_array.shape[0]) + ' ' + str(vector_array.shape[1]) + '\n')
    
                    for i in range(vector_array.shape[0]):
                        # get vector representation as single-line string without brackets
                        vector_string = np.array2string(vector_array[i])[1:-1].replace('\n', ' ')
                        o.write(vocab_list[i] + ' ' + vector_string + '\n')
                print(dataset, decade, 'finished')
# Helper function that receives a year and path to pkl and npy files and returns a (100000, 300)-dimensional numpy array
# that contains the word vectors in alphabetical order and the alphabetical vocabulary list

def word_filtering(path, limiter = 5):
    files = glob.glob(path)
    
    if ('cmn' not in path):
        dia = range(1800, 2001, 10)
        first_year = 1800
    else:
        dia = range(1950, 2001, 10)
        first_year = 1950
    
    decade_files = [filename for filename in files if str(first_year) in filename]
    if decade_files:
        vocab_file = decade_files[0]
        vector_file = decade_files[1]

        vector_array = np.load(vector_file)

        with open(vocab_file, 'rb') as f:
            vocab_list = pickle.load(f)
            
        df = pd.DataFrame(columns = ['word','occurence_number'])
        df['word'] = vocab_list
        df['occurence_number'] = 0
    
    for decade in dia:
        decade_files = [filename for filename in files if str(decade) in filename]
        if decade_files:
            vocab_file = decade_files[0]
            vector_file = decade_files[1]
    
            vector_array = np.load(vector_file)
    
            with open(vocab_file, 'rb') as f:
                vocab_list = pickle.load(f)
            
            for i in range(len(vocab_list)):
                    if np.any(vector_array[i]):
                        df.loc[df.index[df['word'] == vocab_list[i]][0], 'occurence_number'] += 1
    
    df_filtered = df.loc[df.occurence_number >= limiter]
    filtered_vocab = list(df_filtered['word'])
    
    return filtered_vocab

def get_sorted_array(year, filtered_list, path):
    print(year, path)
    files = glob.glob(path)
    decade_files = [filename for filename in files if year in filename]
    
    vocab_file = decade_files[0]
    vector_file = decade_files[1]

    vector_array = np.load(vector_file)

    with open(vocab_file, 'rb') as f:
        vocab_list = np.array(pickle.load(f))
    
    vocab_index = np.in1d(vocab_list, filtered_list).nonzero()
    
    vocab_filtered = vocab_list[vocab_index]
    vector_filtered = vector_array[vocab_index]
    
    
    sort_idx = np.argsort(vocab_filtered)
    vocab_list_sorted = np.array(vocab_filtered)[sort_idx]
    vector_array_sorted = np.array(vector_filtered)[sort_idx]
    vector_array_sorted = np.expand_dims(vector_array_sorted, axis=2)
    
    # To ignore words that do not appear in a decade when calculating the average, we replace the value 0 in null vectors
    # with np.nan. 
    nullvector_idx = np.where(~vector_array_sorted.any(axis=1))[0] # get index of null rows
    vector_array_sorted[nullvector_idx] = np.nan # replace all values in null row with np.nan    
    
    return vector_array_sorted, vocab_list_sorted


# Calculate average for the decades 1800-1890 and 1900-1990
# Procedure:
# 1) Sort the word list alphabetically, sort word list in the same way
# 2) Add each decade list to a separate dimension of a numpy array
# 3) Get cell-wise average of the array

def century_to_txt(path_all):
    for path in path_all: 
        dataset = str(path).split('\\')[2]
        language = str(path).split('\\')[1]
        drive = str(path).split('\\')[0]
        
        # Find words that occur more than five times
        filtered_list = word_filtering(path, limiter = 5)
        
        # Get data from german, french, and english 
        if ('cmn' not in path):
            # Get data from 1800 as starter array for pre
            vector_array_pre, vocab_list_pre = get_sorted_array('1800', filtered_list, path)
            
            for decade in range(1810, 1891, 10):
                # Step 1)
                current_vector_array, _ = get_sorted_array(str(decade), filtered_list, path)
                # Step 2)
                vector_array_pre = np.append(vector_array_pre, current_vector_array, axis=2)
    
            vector_array_pre_mean = np.nanmean(vector_array_pre, axis=2)        
            vector_array_pre_mean = np.nan_to_num(vector_array_pre_mean) # replaces nan with 0.0
    
            output_file = os.path.join(drive, '\\Data\\AllDecadesPrePost', language + '-' + dataset, dataset + '-Pre.txt')
            with open(output_file, 'w', encoding="utf-8") as o:
                # first line contains shape of vectors
                o.write(str(vector_array_pre_mean.shape[0]) + ' ' + str(vector_array_pre_mean.shape[1]) + '\n')
    
                for i in range(vector_array_pre_mean.shape[0]):
                    # get vector representation as single-line string without brackets
                    vector_string = np.array2string(vector_array_pre_mean[i])[1:-1].replace('\n', ' ')
                    o.write(vocab_list_pre[i] + ' ' + vector_string + '\n')
            
            print('Pre', dataset, 'finished')
    
            # Get data from 1900 as starter array for post
            vector_array_post, vocab_list_post = get_sorted_array('1900', filtered_list, path)
    
            for decade in range(1910, 1991, 10):
                # Step 1)
                current_vector_array, _ = get_sorted_array(str(decade), filtered_list, path)
                # Step 2)
                vector_array_post = np.append(vector_array_post, current_vector_array, axis=2)
    
            vector_array_post_mean = np.nanmean(vector_array_post, axis=2)
            vector_array_post_mean = np.nan_to_num(vector_array_post_mean) # replaces nan with 0.0
    
            output_file = os.path.join(drive, '\\Data\\AllDecadesPrePost', language + '-' + dataset, dataset + '-Post.txt')
            with open(output_file, 'w', encoding="utf-8") as o:
                # first line contains shape of vectors
                o.write(str(vector_array_post_mean.shape[0]) + ' ' + str(vector_array_post_mean.shape[1]) + '\n')
    
                for i in range(vector_array_post_mean.shape[0]):
                    # get vector representation as single-line string without brackets
                    vector_string = np.array2string(vector_array_post_mean[i])[1:-1].replace('\n', ' ')
                    o.write(vocab_list_post[i] + ' ' + vector_string + '\n')
                
            print('Post', dataset, 'finished')
            
        else:   
            # Get data from 1950 as starter array for post
            vector_array_post, vocab_list_post = get_sorted_array('1950', filtered_list, path)
    
            for decade in range(1960, 1991, 10):
                # Step 1)
                current_vector_array, _ = get_sorted_array(str(decade), filtered_list, path)
                # Step 2)
                vector_array_post = np.append(vector_array_post, current_vector_array, axis=2)
    
            vector_array_post_mean = np.nanmean(vector_array_post, axis=2)
            vector_array_post_mean = np.nan_to_num(vector_array_post_mean) # replaces nan with 0.0
    
            output_file = os.path.join(drive, '\\Data\\AllDecadesPrePost', language + '-' + dataset, dataset + '-Post.txt')
            with open(output_file, 'w', encoding="utf-8") as o:
                # first line contains shape of vectors
                o.write(str(vector_array_post_mean.shape[0]) + ' ' + str(vector_array_post_mean.shape[1]) + '\n')
    
                for i in range(vector_array_post_mean.shape[0]):
                    # get vector representation as single-line string without brackets
                    vector_string = np.array2string(vector_array_post_mean[i])[1:-1].replace('\n', ' ')
                    o.write(vocab_list_post[i] + ' ' + vector_string + '\n')
                
            print('Post', dataset, 'finished')

def coha_to_txt(path):
    dataset = str(path).split('\\')[2]
    # Get data from 1810 as starter array for pre
    vector_array_pre, vocab_list_pre = get_sorted_array('1810', path)

    for decade in range(1820, 1891, 10):
        # Step 1)
        current_vector_array, _ = get_sorted_array(str(decade), path)
        # Step 2)
        vector_array_pre = np.append(vector_array_pre, current_vector_array, axis=2)

    vector_array_pre_mean = np.nanmean(vector_array_pre, axis=2)
    vector_array_pre_mean = np.nan_to_num(vector_array_pre_mean) # replaces nan with 0.0

    output_file = os.path.join('Data/AllDecadesPrePost', dataset + '-Pre.txt')
    with open(output_file, 'w', encoding="utf-8") as o:
        # first line contains shape of vectors
        o.write(str(vector_array_pre_mean.shape[0]) + ' ' + str(vector_array_pre_mean.shape[1]) + '\n')

        for i in range(vector_array_pre_mean.shape[0]):
            # get vector representation as single-line string without brackets
            vector_string = np.array2string(vector_array_pre_mean[i])[1:-1].replace('\n', ' ')
            o.write(vocab_list_pre[i] + ' ' + vector_string + '\n')
        
    print('Pre', dataset, 'finished')
    # Get data from 1950 as starter array for post
    vector_array_post, vocab_list_post = get_sorted_array('1900', path)

    for decade in range(1910, 1991, 10):
        # Step 1)
        current_vector_array, _ = get_sorted_array(str(decade), path)
        # Step 2)
        vector_array_post = np.append(vector_array_post, current_vector_array, axis=2)

    vector_array_post_mean = np.nanmean(vector_array_post, axis=2)
    vector_array_post_mean = np.nan_to_num(vector_array_post_mean) # replaces nan with 0.0

    output_file = os.path.join('Data/AllDecadesPrePost', dataset + '-Post.txt')
    with open(output_file, 'w', encoding="utf-8") as o:
        # first line contains shape of vectors
        o.write(str(vector_array_post_mean.shape[0]) + ' ' + str(vector_array_post_mean.shape[1]) + '\n')

        for i in range(vector_array_post_mean.shape[0]):
            # get vector representation as single-line string without brackets
            vector_string = np.array2string(vector_array_post_mean[i])[1:-1].replace('\n', ' ')
            o.write(vocab_list_post[i] + ' ' + vector_string + '\n')


if __name__ == '__main__':
    all_paths = [path_german, path_english, path_french, path_chinese]
    decade_to_txt(all_paths)
    century_to_txt(all_paths)