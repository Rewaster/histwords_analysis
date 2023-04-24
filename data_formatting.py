import numpy as np
import glob
import os
import pandas as pd
import pickle
path_german = './Embeddings/ger-all_sgns/sgns/*'
path_english = './Embeddings/eng-all_sgns/sgns/*'
path_french = './Embeddings/fre-all_sgns/sgns/*'
path_chinese = './Embeddings/chi-sim-all_sgns/sgns/*'
# path_COHA = './Embeddings/coha-lemma_sgns/sgns/*'
# Output the word vectors from https://nlp.stanford.edu/projects/histwords/ in the format we need 
# (each decade in a separate file, where each line starts consists of the word followed by the embedding)

for path in [path_german, path_french, path_chinese]: 
    files = glob.glob(path)
    for decade in range(1800, 2001, 10):
        decade_files = [filename for filename in files if str(decade) in filename]
        if decade_files:
            vocab_file = decade_files[0]
            vector_file = decade_files[1]

            dataset = str(path).split('/')[2]

            vector_array = np.load(vector_file)

            with open(vocab_file, 'rb') as f:
                vocab_list = pickle.load(f)

            output_file = os.path.join('Data/IndividualDecades', dataset, str(decade)+'s.txt')
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

def get_sorted_array(year, path):
    print(year, path)
    files = glob.glob(path)
    decade_files = [filename for filename in files if year in filename]
    
    vocab_file = decade_files[0]
    vector_file = decade_files[1]

    vector_array = np.load(vector_file)

    with open(vocab_file, 'rb') as f:
        vocab_list = np.array(pickle.load(f))
        
    sort_idx = np.argsort(vocab_list)
    vocab_list_sorted = np.array(vocab_list)[sort_idx]
    vector_array_sorted = np.array(vector_array)[sort_idx]
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

for path in [path_german, path_french, path_chinese]: 
    dataset = str(path).split('/')[2]
    
    # Get data from german and english 
    if ('chi-sim' not in path):
        # Get data from 1800 as starter array for pre
        vector_array_pre, vocab_list_pre = get_sorted_array('1800', path)
        
        for decade in range(1810, 1891, 10):
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

        # Get data from 1900 as starter array for post
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
            
        print('Post', dataset, 'finished')
            
        
    # COHA needs to be separate because of slightly different timeframe
    else:   
        # Get data from 1810 as starter array for pre
        #vector_array_pre, vocab_list_pre = get_sorted_array('1810', path)

        #for decade in range(1820, 1891, 10):
            # Step 1)
            #current_vector_array, _ = get_sorted_array(str(decade), path)
            # Step 2)
            #vector_array_pre = np.append(vector_array_pre, current_vector_array, axis=2)

        #vector_array_pre_mean = np.nanmean(vector_array_pre, axis=2)
        #vector_array_pre_mean = np.nan_to_num(vector_array_pre_mean) # replaces nan with 0.0

        #output_file = os.path.join('Data/AllDecadesPrePost', dataset + '-Pre.txt')
        #with open(output_file, 'w', encoding="utf-8") as o:
            # first line contains shape of vectors
            #o.write(str(vector_array_pre_mean.shape[0]) + ' ' + str(vector_array_pre_mean.shape[1]) + '\n')

            #for i in range(vector_array_pre_mean.shape[0]):
                # get vector representation as single-line string without brackets
                #vector_string = np.array2string(vector_array_pre_mean[i])[1:-1].replace('\n', ' ')
                #o.write(vocab_list_pre[i] + ' ' + vector_string + '\n')
            
        #print('Pre', dataset, 'finished')

        # Get data from 1950 as starter array for post
        vector_array_post, vocab_list_post = get_sorted_array('1950', path)

        for decade in range(1950, 1991, 10):
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
            
        print('Post', dataset, 'finished')
