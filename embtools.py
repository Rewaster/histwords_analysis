import numpy as np
import pickle
import scipy.linalg
import os
import pandas as pd
"""
Function for working with embeddings after alignment (tagging, refinement, combining, etc.)
"""
class refine:
    def load_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as read_file:
            out_txt = read_file.read().splitlines()
        return out_txt
    
    def tag_word2vec_file(self, source_emb, tag, as_txt = True):
        """Adds a tag to the words in a given Word2Vec file. Saves the result in another file
            in the Word2Vec format.

        Args:
            file_path: the relative path to the Word2Vec file with untagged words.
            save_file_path: the relative path to the file where the result will be saved.
            tags: tags to be added to the words. tags should be added as a
            according to this structure: _old/_new + language name
            Note: with tags, different dictionaries for alignment should be
            used (see numbers_and_tagging function)
        Returns:
            A .txt document with words tagged with an appropriate tag.
        """
        if as_txt:
            with open(source_emb, "r", encoding="utf-8") as read_file:
                vsp = read_file.read().splitlines()
        else:
            vsp = source_emb
        target_emb = []
        #We keep the header as it is
        target_emb.append(vsp[0])
        #We add the tags to the other lines
        for i in range(1, len(vsp)):
            words = vsp[i].split(" ", 1)
            words[0] += tag
            target_emb.append(words[0] + " " + words[1])
        return target_emb
    
    def common_filter(self, emb_path):
        with open(emb_path, "r", encoding="utf-8") as emb_file:
            emb_file.readline()
            emb_lines = emb_file.read().splitlines()
        emb_words = set()
        for line in emb_lines:
            word, _ = line.split(" ", 1)
            emb_words.add(word)
        return emb_file, emb_words
    def intersect_check(self, list1, list2):
        common_emb_words = list1.intersection(list2)
        common_numbers = set()
        for word in common_emb_words:
            try:
                float(word)
                common_numbers.add(word)
            except ValueError:
                continue
        return common_numbers
    def decade_dicts(self, emb1_path, emb2_path, out_path, langs, decade, hrd_path = 0, words = True, common_numbers = True):
        dict_path = out_path + langs[0] + '-' + langs[1] + '_' + str(decade) + '_dict.txt'
        if words:
            df = pd.read_csv(hrd_path)
            hrd_l1 = df[langs[0]].values.tolist()
            hrd_l2 = df[langs[1]].values.tolist()
        if common_numbers:
            emb1_lines, emb1_words = self.common_filter(emb1_path)
            emb2_lines, emb2_words = self.common_filter(emb2_path)
            
            common_numbers = self.intersect_check(emb1_words, emb2_words)
        
        with open(dict_path, "w", encoding="utf-8") as file:
            if common_numbers:
                for number in common_numbers:
                    file.write(number + " " + number + "\n")
            if words:
                for i in range(len(df)):
                    word_l1 = hrd_l1[i]
                    word_l2 = hrd_l2[i]
                    file.write(word_l1 + " " + word_l2 + "\n")
    
    def century_dicts(self, oldL1_path, newL1_path, oldL2_path, newL2_path, common_save_path, langs, hrd_path = 0, words = True, common_numbers = True):
        """Finds the numbers that are present in both L1 (first language) Word2Vec embeddings files, in both L2 (second language)
        Word2Vec embeddings files, in both the 'old' files and in both 'new' files. Saves the result in 3 files: one for the L1 files,
        one for the L2 files, one for the 'old' and 'new' files. The goal is to find the common numbers to be able to add them to the seed dictionaries for the VecMap
        algorithm. After that tags four embedding .txt files provided with relevant tags for further analysis and plotting.

        Args:
            oldL1_path: the relative path to the 'old' L1 Word2Vec embeddings files.
            newL1_path: the relative path to the 'new' L1 Word2Vec embeddings files.
            oldL2_path: the relative path to the 'old' L2 Word2Vec embeddings files.
            newL2_path: the relative path to the 'new' L2 Word2Vec embeddings files.
            common_save_path: the relative path where the common numbers will be saved
                in the format of the VecMap seed dictionary (and tagged embeddings for the same languages).
            langs: list of languages (length = 2) to use as tags. Should be put in the list in the same
            order as paths to language embeddings (ex: ['L1','L2']). Language list for other files and embeddings
            for this project: ['eng', 'ger', 'fre', 'chn']
            
        Returns:
            Three computed dictionaries and four tagged .txt files.
        """
        if words:
            df = pd.read_csv(hrd_path)
            hrd_l1 = df[langs[0]].values.tolist()
            hrd_l2 = df[langs[1]].values.tolist()
        oldl1_tag = '_old' + langs[0].upper() + " "
        newl1_tag = '_new' + langs[0].upper() + " "
        oldl2_tag = '_old' + langs[1].upper() + " "
        newl2_tag = '_new' + langs[1].upper() + " "
        #We add the words to sets
        oldL1_lines, oldL1_words = self.common_filter(oldL1_path)
        newL1_lines, newL1_words = self.common_filter(newL1_path)
        oldL2_lines, oldL2_words = self.common_filter(oldL2_path)
        newL2_lines, newL2_words = self.common_filter(newL2_path)
        #We find the common words for each particular case
        common_L1_numbers = self.intersect_check(oldL1_words, newL1_words)
        common_L2_numbers = self.intersect_check(oldL2_words, newL2_words)
        common_old_numbers = self.intersect_check(oldL1_words, oldL2_words)
        common_new_numbers = self.intersect_check(newL1_words, newL2_words)
        #We save the results in the three files
        file_path_l1_dict = common_save_path + langs[0] + '_dict.txt'
        file_path_l2_dict = common_save_path + langs[1] + '_dict.txt'
        file_path_l2_l1_dict = common_save_path + langs[1] + '-' + langs[0] + '_tagged_dict.txt'
        with open(file_path_l1_dict, "w", encoding="utf-8") as file:
            if common_numbers:
                for number in common_L1_numbers:
                    file.write(number + " " + number + "\n")
            if words:
                for word in hrd_l1:
                    file.write(str(word).casefold() + " " + str(word).casefold() + "\n")
        with open(file_path_l2_dict, "w", encoding="utf-8") as file:
            if common_numbers:
                for number in common_L2_numbers:
                    file.write(number + " " + number + "\n")
            if words:
                for word in hrd_l2:
                    file.write(str(word).casefold() + " " + str(word).casefold() + "\n")
        with open(file_path_l2_l1_dict, "w", encoding="utf-8") as file:
            if common_numbers:
                for number in common_old_numbers:
                    file.write(number + oldl2_tag + number + oldl1_tag + "\n")
            if words:
                for i in range(len(df)):
                    word_l1 = str(hrd_l1[i]).casefold()
                    word_l2 = str(hrd_l2[i]).casefold()
                    file.write(word_l2 + oldl2_tag + word_l1 + oldl1_tag + "\n")          
            if common_numbers:
                for number in common_new_numbers:
                    file.write(number + newl2_tag + number + newl1_tag + "\n")
            if words:
                for i in range(len(df)):
                    word_l1 = str(hrd_l1[i]).casefold()
                    word_l2 = str(hrd_l2[i]).casefold()
                    file.write(word_l2 + newl2_tag + word_l1 + newl1_tag + "\n")
        return file_path_l1_dict, file_path_l2_dict, file_path_l2_l1_dict
                
    
    def read_embeddings(self, emb, threshold=0, vocabulary=None, dtype='float'):
        """Utilitary function for converting the embedding lists into a word list and a matrix
        for embeddings. Might be of some use separately, but currently only used as
        a function to convert the embeddings for processing.
        
        Args:
            emb: a list with the lines in the form of 'word vectors', first line is
            'token_number dimension'
            threshold: how many lines to read from the embedding, 0 by default,
            if equal to 0, takes the number from the first line of embeddings (total
            token number), if it's a number bigger  than 0, limit the line count to
            that number.
            vocabulary: which words to append to the resulting file, should be a list,
            None by default, if None, appends all of the words present in the embedding
            dtype: Type of data, float by default.
    
        Returns:
            A word list and a matrix for embedding vectors.
        """
        header = emb[0].split(' ')
        count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
        dim = int(header[1])
        words = []
        matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
        for i in range(1, count):
            word, vec = emb[i].split(' ', 1)
            if vocabulary is None:
                words.append(word)
                matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
            elif word in vocabulary:
                words.append(word)
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
        return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))
    
    def combine(self, emb_l1, emb_l2, out_vocab, out_npy):
        """Combines two aligned embedding lists into two HistWords-type embedding files
        (.npy and .pkl) in a single space and saves the result.
    
        Args:
            emb_l1: first language embedding list
            with the lines in the form of 'word vectors'
            emb_l2: first language embedding list
            with the lines in the form of 'word vectors'
            out_vocab: path to the file where the resulting .pkl will
                be saved.
            out_npy: path to the file where the resulting .npy will
                be saved.
            Note: path for out_vocab and out_npy must contain file
            extensions in their paths (.pkl and .npy respectively).
            full_prep function does this automatically.
        Returns:
            Nothing.
        """
        
        words_l1, vecs_l1 = self.read_embeddings(emb_l1)
        
        words_l2, vecs_l2 = self.read_embeddings(emb_l2)
        
        words_total = words_l1 + words_l2
        print(len(words_total))
        vecs_total = np.concatenate((vecs_l1, vecs_l2), axis=0)
        
        vocab = open(out_vocab, 'wb')
        npy = open(out_npy, 'wb')
        
        pickle.dump(words_total, vocab)
        vocab.close()
        
        np.save(npy, vecs_total)
        npy.close()
        
    def combine_txt(self, emb_l1, emb_l2, out_txt):
        """Combines two aligned embeddings into a single text file
        and saves the result.
    
        Args:
            file_l1: path to the first .txt file with multiple spaces.
            file_l2: path to the second .txt file with multiple spaces.
            out_txt: path to the file where the resulting .txt will
                be saved.
    
        Returns:
            Nothing.
        """
        emb1_len, emb1_shape = emb_l1[0].split(" ", 1)
        emb2_len, emb2_shape = emb_l2[0].split(" ", 1)
        
        emb1 = emb_l1[1:]
        emb2 = emb_l2[1:]
        emb_total = emb1 + emb2
        
        if emb1_shape == emb2_shape:
            emb_shape = emb1_shape
        else:
            raise ValueError('Embedding dimensions do not match or dimensions are not present in the file, please check the files')
            
        len_total = len(emb_total)
        print('New length of a combined embedding:', len_total)
        
        with open(out_txt, 'w', encoding="utf-8") as o:
            # first line contains shape of vectors
            o.write(str(len_total) + ' ' + str(emb_shape) + '\n')
    
            for i in range(len_total):
                o.write(emb_total[i] + '\n')
                
    def one_space(self, split_lines):
        """Changes the multiple spaces from a file to single spaces and saves the result in a new file.
    
        Args:
            file_path: the relative path to the file with multiple spaces.
            save_file_path: the relative path to the file where the result will
                be saved.
    
        Returns:
            A list with one-spaced lines.
        """
        one_spaced = []
        for line in split_lines:
            #We change the multiple spaces in single spaces
            one_space_line = " ".join(line.split())
            one_spaced.append(one_space_line)
        return one_spaced
    
    def words_percent(self, file_lines):
        """Finds the percentage of words with a null embedding
        in a Word2Vec file.
    
        Args:
            file_path: the relative path to the file for which
                the percentage will be computed.
    
        Returns:
            The computed percentage.
        """
        total = len(file_lines)
        invalid = 0
        for line in file_lines:
            _, embedding = line.split(" ", 1)
            embedding = list(map(float, embedding.split()))
            if scipy.linalg.norm(embedding) == 0:
                invalid +=1
        return invalid/total
    
    def zero_clean(self, file_lines):
        """Removes the words with a null embedding in a Word2Vec
        file and saves the result in a new file in the Word2Vec format.
    
        Args:
            file_path: the relative path to a Word2Vec file with some words having null embeddings.
        Returns:
            A list with zero-cleaned lines.
        """
        #We only keep the embeddings dimension from the header
        _, dimension = file_lines[0].split(" ")
        not_null_embeddings = []
        zero_cleaned = []
        for file_line in file_lines[1:]:
                word, embedding_str1 = file_line.split(" ", 1)
                embedding1 = list(map(float, embedding_str1.split()))
                #We only keep the words with an embedding that isn't null
                if scipy.linalg.norm(embedding1) != 0:
                    not_null_embeddings.append(word + " " + embedding_str1)
        zero_cleaned.append(str(len(not_null_embeddings)) + " " + dimension)
        for word_embedding in not_null_embeddings:
            zero_cleaned.append(word_embedding)
        return zero_cleaned
    
    def save_txt(self, emb, out_path):
        if len(emb[0].split()) == 2:
            add_dim = False
        else:
            add_dim = True
        with open(out_path, 'w', encoding="utf-8") as o:
            # get the dimension from the file itself
            if add_dim:
                test_word, test_emb = emb[0].split(" ", 1)
                test_dim = test_emb.split()
                # first line contains shape of vectors
                o.write(str(len(emb)) + ' ' + str(len(test_dim)) + '\n')
            for i in emb:
                o.write(str(i) + '\n')
    
    def two_step(self, file_path_1, file_path_2, zero_clean = False):
        with open(file_path_1, "r", encoding="utf-8") as read_file:
            multi_spaces_l1 = read_file.read().splitlines()
        with open(file_path_2, "r", encoding="utf-8") as read_file:
            multi_spaces_l2 = read_file.read().splitlines()
        one_space_l1 = self.one_space(multi_spaces_l1)
        one_space_l2 = self.one_space(multi_spaces_l2)
        if zero_clean:
            final_l1 = self.zero_clean(one_space_l1)
            final_l2 = self.zero_clean(one_space_l2)
        else:
            final_l1 = one_space_l1
            final_l2 = one_space_l2
        return final_l1, final_l2            
    
    def full_prep(self, temp_path, dict_path, emb_1_path, emb_2_path, aligned_emb_1_path, aligned_emb_2_path, vecmap_dir, out_path, lang, decades = True, tagging = True):
        """Does all three of the above commands to speed up the process if all 
        three are needed (zero cleaning, one spacing and percent calculation).
        Further explanation for each command can be found in their respective
        descriptions (zero_clean, one_space and alignment)
        Подумать - сначала зероклин или потом?
        Args:
            file_path_l1: path to the first .txt file with multiple spaces.
            file_path_l2: path to the second .txt file with multiple spaces.
            out_txt: path to the file where the resulting .txt will
            be saved.
            as_txt: choose to save resulting embeddings as a .txt file or
            as HistWords-type embeddings (.npy and .pkl). True by default.
            decades: whether decade-type embeddings or century-type embeddings
            are used as an input. True by default. Changes the output.
        Returns:
            A refined .txt file - one-spaced, without zero embeddings 
            OR:
            Two files(.npy and .pkl), prepared for HistWords scripts
        """
        
        os.chdir(vecmap_dir)
        temp_1 = temp_path + emb_1_path[emb_1_path.rfind('\\')+1:-4] 
        temp_2 = temp_path + emb_2_path[emb_2_path.rfind('\\')+1:-4]
        if not decades:
            if not os.path.isfile(aligned_emb_1_path) or not os.path.isfile(aligned_emb_2_path):
                final_l1, final_l2 = self.two_step(emb_1_path, emb_2_path, zero_clean = True)
                temp_1 = temp_1 + '-prepped.txt'
                temp_2 = temp_2 + '-prepped.txt'
                self.save_txt(final_l1, temp_1)
                self.save_txt(final_l2, temp_2)
                train_command = "python map_embeddings.py --semi_supervised " + dict_path + " " + temp_1 + " " + temp_2 + " " + aligned_emb_1_path + " " + aligned_emb_2_path + " --cuda"
                os.system(train_command) 
            if not os.path.isfile(out_path):        
                    if tagging:
                            aligned_emb_1 = self.tag_word2vec_file(aligned_emb_1_path, '_old'+lang.upper(), as_txt = True) 
                            aligned_emb_2 = self.tag_word2vec_file(aligned_emb_2_path, '_new'+lang.upper(), as_txt = True)
                    else:
                            aligned_emb_1 = self.load_txt(aligned_emb_1_path)
                            aligned_emb_2 = self.load_txt(aligned_emb_2_path)
                    self.combine_txt(aligned_emb_1, aligned_emb_2, out_path)
        else:
            year = emb_1_path[emb_1_path.rfind('\\')+1:-5]
            dir_path = out_path[:out_path.rfind('\\')+1]
            npy_path = dir_path + year + '-w.npy'
            pkl_path = dir_path + year + '-vocab.pkl'
            if not os.path.isfile(npy_path) or not os.path.isfile(pkl_path):
                final_l1, final_l2 = self.two_step(emb_1_path, emb_2_path, zero_clean = False)
                temp_1 = temp_1 + '-' + lang[0] + '-prepped.txt'
                temp_2 = temp_2 + '-' + lang[1] + '-prepped.txt'
                self.save_txt(final_l1, temp_1)
                self.save_txt(final_l2, temp_2)
                train_command = "python map_embeddings.py --semi_supervised " + dict_path + " " + temp_1 + " " + temp_2 + " " + aligned_emb_1_path + " " + aligned_emb_2_path + " --cuda"
                os.system(train_command)
                aligned_emb_1 = self.load_txt(aligned_emb_1_path)
                aligned_emb_2 = self.load_txt(aligned_emb_2_path)
                self.combine(aligned_emb_1, aligned_emb_2, pkl_path, npy_path)
