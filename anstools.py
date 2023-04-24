import pickle
import scipy.spatial
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from gensim.models import KeyedVectors
from math import isclose
import os

class analysis:
    def sort_embeddings(self, mapped_embeddings_file_path, langs):
        """Sorts the embedding of the file containing all mapped word embeddings in four categories:
        'old' version of L1 words, 'new' version of L1 words, 'old' version of L2 words,
        'new' version of L2 words thanks to the tags added to the words. Returns one dictionary per category 
        with the words as keys and their embedding as values.

        Args:
            mapped_embeddings_file_path: the path of the file containing all mapped word embeddings.
            langs = list of 2 languages in the combined embedding space

        Returns:
            one dictionary per category.
        """
        #We read the file
        with open(mapped_embeddings_file_path, "r", encoding="utf-8") as file:
            #We drop the header as we only need the word embeddings
            file.readline()
            file_lines = file.read().splitlines()
        oldL1_embeddings = []
        newL1_embeddings = []
        oldL2_embeddings = []
        newL2_embeddings = []
        #We iterate on the word embeddings
        for line in file_lines:
            #We separate the tagged words from the embeddings
            tagged_word, embedding = line.split(" ", 1)
            #We separate the words from the tags
            word, tag = tagged_word[:-7], tagged_word[-6:]
            #We sort the words according to their tags
            if tag == "old" + langs[0].upper():
                oldL1_embeddings.append(word + ' ' + embedding)
            elif tag == "new" + langs[0].upper():
                newL1_embeddings.append(word + ' ' + embedding)
            elif tag == "old" + langs[1].upper():
                oldL2_embeddings.append(word + ' ' + embedding)
            elif tag == "new" + langs[1].upper():
                newL2_embeddings.append(word + ' ' + embedding)
        return oldL1_embeddings, newL1_embeddings, oldL2_embeddings, newL2_embeddings

    def corr_check(self, emb1_path, emb2_path, hrd_path, threshold = 0, scored = False):
        out_path = hrd_path[:-4] + '_results.csv'
        if not os.path.isfile(out_path):
            emb1 = KeyedVectors.load_word2vec_format(emb1_path, encoding ='utf-8')
            emb2 = KeyedVectors.load_word2vec_format(emb2_path, encoding ='utf-8')
            df = pd.read_csv(hrd_path)
            df['Similarity'] = np.nan
            if not scored:
                df['Type of change (machine)'] = np.nan
            for i in range(len(df)):
                try:
                    word_1 = df.loc[i][1].casefold()
                    word_2 = df.loc[i][0].casefold()
                    old_sim, new_sim, shift, W1_mono_distance, W2_mono_distance = self.change_calc(emb1[word_1], emb2[word_1], emb1[word_2], emb2[word_2])
                    if not scored:
                        df.loc[[i],'Similarity'] = shift
                    else:
                        df.loc[[i],'Similarity'] = new_sim
                    if not scored:
                        if isclose(W1_mono_distance, W2_mono_distance, abs_tol = 0.1):
                            result = 'Parallel Change'
                        elif shift > threshold:
                            result = 'Convergence'
                        elif shift < threshold:
                            result = 'Divergence'
                        df.loc[[i],'Type of change (machine)'] = result
                except:
                    pass
            df_nn = df.dropna(axis=0, how='any')
        else:
            df_nn = pd.read_csv(out_path)
        if not scored:
            score = accuracy_score(df_nn['Type of change'].values.tolist(), df_nn['Type of change (machine)'].values.tolist())
            print('Precision score between human scores and machine cosine similarity is ', score)
        else:
            score = df_nn['score'].corr(df_nn['Similarity'], method='spearman')
            print('Correlation between human rated similarity and machine determined one is ', score)
        
        df_nn.to_csv(out_path, index=False)
        return score
    
    def dump_pickle(self, dump_path, emb):
        with open(dump_path, 'wb') as f:
            pickle.dump(emb, f)
    def convert_embedding(self, source_embed, as_txt = False):
        """Converts embeddings from a list (or a .txt file) to a dictionary,
        with words as keys and vectors as items. Useful for closest words
        analysis.
        Args:
            source_embed: embedding to be converted, in a form of a list
                Note: if as_txt = True, should be a full or relative path
                to a .txt file.
            as_txt: False by default. Allows to convert from .txt if True.
        Returns:
            A dictionary with words as keys and vectors as items.
        """
        
        if as_txt:
            with open(source_embed, "r", encoding="utf-8") as file:
                #We drop the header as we only need the word embeddings
                file.readline()
                emb = file.read().splitlines()
        else:
            #We drop the header as we only need the word embeddings
            emb = source_embed[1:]
        converted_embedding = dict()
        #We iterate on the word embeddings
        for line in emb:
            #We separate the words from the embeddings
            word, embedding_str = line.split(" ", 1)
            embedding = list(map(float, embedding_str.split(" ")))
            #We combine word and their embeddings back into a dictionary for pickling and training
            converted_embedding[word] = embedding     
        return converted_embedding
    def find_closest_words(self, embedding1, embedding2, neighbours_number = 10, load_txt = False, converted = True):
        """Finds k closest neighbours in embedding2 for every word in embedding1.
        
        Args:
            embedding1: first language embedding to be converted, in the form of a list
                Note: if load_txt = True, should be a full or relative path
                to a .txt file.
            embedding2: second language embedding to be converted, in the form of a list
                Note: if load_txt = True, should be a full or relative path
                to a .txt file.
            load_txt: False by default. Allows to convert from .txt if True.
        Returns:
            A dictionary of closest L2 neighbours as items with L1 words as keys.
        """
        closest_words = dict()
        if load_txt:
            embedding1 = self.convert_embedding(embedding1, as_txt = True)
            embedding2 = self.convert_embedding(embedding2, as_txt = True)
        elif not converted:
            embedding1 = self.convert_embedding(embedding1)
            embedding2 = self.convert_embedding(embedding2)
        #We turn the items into a list so that we can access them with indexes
        embedding1_list = list(embedding1.items())
        embedding2_list = list(embedding2.items())
        #Should be optional, but my GPU can't load 12+ GB of data simultaneously, so I switched to CPU use (slightly slower, 2-3 mins for embedding for me)
        with tf.device("/CPU:0"):
            #We normalize the embeddings to avoid redundant calculations during the computation of cosine similarities
            embedding1_normalized = tf.math.l2_normalize(np.array([embedding for _, embedding in embedding1_list], dtype=np.float32), axis=1)
            embedding2_normalized = tf.math.l2_normalize(np.array([embedding for _, embedding in embedding2_list], dtype=np.float32), axis=1)
            #We compute the matrix multiplication between the matrix of the second language embeddings and the first language embeddings.
            #The element at the index (i, j) of the resulting matrix is the cosine similarity between the element at the index i in the list of second
            #language embeddings and the element at the index j in the list of first language embeddings.
            cosine_similarity = tf.matmul(embedding2_normalized, tf.transpose(embedding1_normalized, [1, 0]))
            #For each line of the cosine similarity matrix (corresponding to a L2 word), we get the indexes of the neighbours_number lowest values
            #which represent the indexes of the neighbours_number closest neighbours in the list of first language embeddings
            closest_neighbours_indexes = np.flip(np.argsort(cosine_similarity, axis = 1)[:,-neighbours_number:], axis = 1)
            #We add the closest neighbours to the dictionary
            for i in range(len(closest_neighbours_indexes)):
                embedding2_word = embedding2_list[i][0]
                closest_words[embedding2_word] = [embedding1_list[j][0] for j in closest_neighbours_indexes[i]]

        return closest_words
    def freq_filter(self, freq_dict_l1, freq_dict_l2, shift_list, l2):
        filtered = []
        for i in range(len(shift_list)):
            dict_test_1 = str(shift_list[i][1]).casefold() in (str(word).casefold() for word in freq_dict_l1)
            dict_test_2 = str(shift_list[i][0]).casefold() in (str(word).casefold() for word in freq_dict_l2)
            if dict_test_1 and dict_test_2:
                filtered.append(shift_list[i])
        return filtered
    
    def load_pickle(self, path):
        with open(path, 'rb') as f:
            loaded_embedding = pickle.load(f, encoding='latin1')
        return loaded_embedding
    
    def save_to_file(self, shift_list, file_path, langs, tag, max_lines=10000, par = False): 
        i=0
        l1 = langs[0]
        l2 = langs[1]
        with open(file_path, 'w', encoding='utf-8') as file:
            if not par:
                file.write(l2 + "_word " + l1 + "_word " + tag + ' Old_similarity' + ' New_similarity' + "\n")
            else:
                file.write(l2 + "_word " + l1 + "_word " + 'Average_distance' + ' L1_distance' + ' L2_distance' + "\n")
            for line in shift_list:
                if i == max_lines:
                    break
                file.write(line[0] + " " + line[1] + " " + str(line[2]) + " " + str(line[3]) + " " + str(line[4]) + "\n")
                i+=1
    
    def change_calc(self, old_emb1, new_emb1, old_emb2, new_emb2):
        W1_mono_distance = scipy.spatial.distance.cosine(old_emb1, new_emb1)
        W2_mono_distance = scipy.spatial.distance.cosine(old_emb2, new_emb2)
        old_sim = 1 - scipy.spatial.distance.cosine(old_emb1, old_emb2)
        new_sim = 1 - scipy.spatial.distance.cosine(new_emb1, new_emb2)
        shift = new_sim - old_sim
        return old_sim, new_sim, shift, W1_mono_distance, W2_mono_distance
    
    def find_change_type(self, oldL1_embedding, newL1_embedding, oldL2_embedding, newL2_embedding, old_dict, new_dict, change_type, langs, freq_dir, threshold = 0, abs_threshold = 0.1):
        oldL1_embedding_input = KeyedVectors.load_word2vec_format(oldL1_embedding, encoding ='utf-8')
        newL1_embedding_input = KeyedVectors.load_word2vec_format(newL1_embedding, encoding ='utf-8')
        oldL2_embedding_input = KeyedVectors.load_word2vec_format(oldL2_embedding, encoding ='utf-8')
        newL2_embedding_input = KeyedVectors.load_word2vec_format(newL2_embedding, encoding ='utf-8')
        convergence_list = []
        divergence_list = []
        parallel_change_list = []
        if change_type == 'Convergence' or change_type == 'Parallel Change':
            word_list_1 = new_dict
        elif change_type == 'Divergence':
            word_list_1 = old_dict
        freq_l1 = freq_dir + langs[0] + '_all_freqs.pkl'
        dict_l1 = self.load_pickle(freq_l1)
        
        freq_l2 = freq_dir + langs[1] + '_all_freqs.pkl'
        dict_l2 = self.load_pickle(freq_l2)
        
        freq_thresh = 10.0 ** (-1.0 * float(6))
        #We pick a L2 word and one of its english neighbours
        for L2_word in word_list_1:
            for L1_word in word_list_1[L2_word]:
                #We check if they have an embedding
                L1_word_freq_old = dict_l1[L1_word][1900]
                L2_word_freq_old = dict_l2[L2_word][1900]
                L1_word_freq_new = dict_l1[L1_word][1990]
                L2_word_freq_new = dict_l2[L2_word][1990]
                if L2_word in oldL2_embedding_input and L1_word in oldL1_embedding_input:
                    if L2_word in newL2_embedding_input and L1_word in newL1_embedding_input:
                        if L1_word_freq_old > freq_thresh and L2_word_freq_old > freq_thresh and L1_word_freq_new > freq_thresh and L2_word_freq_new > freq_thresh:
                            old_sim, new_sim, shift, L1_mono_distance, L2_mono_distance = self.change_calc(oldL1_embedding_input[L1_word], newL1_embedding_input[L1_word], oldL2_embedding_input[L2_word], newL2_embedding_input[L2_word])
                            if isclose(L1_mono_distance, L2_mono_distance, abs_tol = abs_threshold):
                                #We filter out the words with a 'parallelism value' that is too high
                                parallel_change_list.append([L2_word, L1_word, (L1_mono_distance + L2_mono_distance)/2, L1_mono_distance, L2_mono_distance])
                            if shift > threshold:
                                convergence_list.append([L2_word, L1_word, shift, old_sim, new_sim])
                            if shift < threshold:
                                divergence_list.append([L2_word, L1_word, shift, old_sim, new_sim])
                        #We sort the array to have the pairs with the strongest divergence first
        if change_type == 'Convergence':
                sorted_shifts = sorted(convergence_list, key = lambda ele: ele[2], reverse = True)
        elif change_type == 'Divergence':
                sorted_shifts = sorted(divergence_list, key = lambda ele: ele[2], reverse = False)
        elif change_type == 'Parallel Change':
                sorted_shifts = sorted(parallel_change_list, key = lambda ele: ele[2], reverse = True)
        return sorted_shifts