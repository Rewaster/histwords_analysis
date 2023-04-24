import os
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.manifold import TSNE
from sequentialembedding import SequentialEmbedding
import adjustText
import threading
from sklearn.neighbors import NearestNeighbors

class graphing:
    def get_cmap(self, n, name='YlGn', CMAP_MIN = 5):
        self.CMAP_MIN = CMAP_MIN
        return plt.cm.get_cmap(name, n+CMAP_MIN)
    # this is based on embedding.py get_time_sims
    def get_time_sims(self, word1):
        start = time.time()
        time_sims = collections.OrderedDict()
        lookups = {}
        nearests = {}
        sims = {}
        for year, embed in self.embeds.items():
            nearest = []
            nearests["%s|%s" % (word1, year)]= nearest
            time_sims[year] = []

            for sim, word in embed.closest(word1, n=15):
                ww = "%s|%s" % (word, year)
                nearest.append((sim, ww))
                if sim > 0.3:
                    time_sims[year].append((sim, ww))
                    lookups[ww] = embed.represent(word)
                    sims[ww] = sim

        print("GET TIME SIMS FOR %s TOOK %s" % (word1, time.time() - start))
        return time_sims, lookups, nearests, sims

    def load_embeddings(self, emb_path, year_1, year_2, dia):
        embed_lock = threading.Lock()
        EMBED_CACHE = {}
        with embed_lock:
            print("LOADING EMBEDDINGS %s" % emb_path)
            start = time.time()

            if emb_path in EMBED_CACHE:
                return EMBED_CACHE[emb_path]

            print("THIS MIGHT TAKE A WHILE...")

            embeddings = SequentialEmbedding.load(emb_path, range(year_1, year_2, dia))
            print("LOAD EMBEDDINGS TOOK %s" % (time.time() - start))

            EMBED_CACHE[emb_path] = embeddings
            return embeddings

    def get_embedding_list(self, dirname="embeddings"):
        dirs = []
        for f in os.listdir(dirname):
            fname = os.path.join(dirname, f)
            
            if os.path.isdir(fname):
                dirs.append(fname)
        return dirs

    def select_embedding(self):
        global EMBEDDING
        print("")
        print("Please select an embedding to load")
        embeddings = self.get_embedding_list()
        for i, embed in enumerate(embeddings):
            print("%s) %s" % (i+1, embed))

        while True:
            selection = input("Load which embedding? ")
            try:
                select_num = int(selection)
                embedding = embeddings[select_num-1]
                break
            except:
                print("Please select a number between %s and %s" % (1, len(embeddings)))

        print("")
        EMBEDDING = embedding

        return self.load_embeddings(embedding)


    def clear_figure(self):
        plt.figure(figsize=(20,20))
        plt.clf()

    def fit_tsne(self, values):
        start = time.time()
        mat = np.array(values)
        #model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
        model = TSNE(n_components=2, learning_rate='auto', init='random')
        fitted = model.fit_transform(mat)
        print("FIT TSNE TOOK %s" % (time.time() - start))

        return fitted


    def get_now(self):
        return int(time.time() * 1000)

    def plot_words(self, word1, words, fitted, cmap, sims):
        plt.scatter(fitted[:,0], fitted[:,1], alpha=0)
        plt.suptitle("%s" % word1, fontsize=30, y=0.1)
        plt.axis('off')

        annotations = []
        texts = []
        isArray = type(word1) == list
        for i in range(len(words)):
            pt = fitted[i]
            ww, decade = list(words)[i].split("|")
            color = cmap((int(decade) - 1840) / 10 + self.CMAP_MIN)
            word = ww
            sizing = sims[list(words)[i]] * 30

            # word1 is the word we are plotting against
            if ww == word1 or (isArray and ww in word1):
                annotations.append((ww, decade, pt))
                word = decade
                color = 'black'
                sizing = 15


            texts.append(plt.text(pt[0], pt[1], word, color=color, size=int(sizing)))
        adjustText.adjust_text(texts)
        return annotations

    def plot_annotations(self, annotations):
        # draw the movement between the word through the decades as a series of
        # annotations on the graph
        annotations.sort(key=lambda w: w[1], reverse=True)
        prev = annotations[0][-1]
        for ww, decade, ann in annotations[1:]:
            plt.annotate('', xy=prev, xytext=ann,
                arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.3,width=2, headwidth=15))
            print(prev, ann)
            prev = ann

    def savefig(self, name, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)

        fname = os.path.join(directory, name)

        plt.savefig(fname, dpi=1200, bbox_inches=0)
        
    def decade_plotting(self, WORDS, emb_path, out_path, year_1, year_2, dia):
        embeddings = self.load_embeddings(emb_path, year_1, year_2, dia)
        for word1 in WORDS:
            self.clear_figure()
            time_sims, lookups, nearests, sims = self.get_time_sims(embeddings, word1)

            words = lookups.keys()
            values = [ lookups[word] for word in words ]
            fitted = self.fit_tsne(values)
            if not len(fitted):
                print("Couldn't model word ", word1)
                continue

            # draw the words onto the graph
            cmap = self.get_cmap(len(time_sims))
            annotations = self.plot_words(word1, words, fitted, cmap, sims)

            if annotations:
                self.plot_annotations(annotations)

            self.savefig("%s_annotated_%s_%s_range_%s.jpg" % (word1, year_1, year_2, dia), out_path)
            for year, sim in time_sims.items():
                print(year, sim)
    
    def century_plotting(self, cent_folder, langs, word1, word2, out_path):
        def embed_filter(path):
            with open(path, 'r', encoding="utf-8") as f:
                x = np.loadtxt(f, skiprows=1, delimiter=" ", usecols=range(1, 301))
            with open(path, 'r', encoding="utf-8") as f:
                y_withTags = np.asarray([line.split(' ')[0] for line in f.readlines()][1:]) # skip first line containing embedding info
                y_withoutTags = np.asarray([word.split('_')[0] for word in y_withTags])     
            relevant_idx = np.where(x.any(axis=1))[0] # get index of all rows that are not null
            x = x[relevant_idx]
            y_withTags = y_withTags[relevant_idx]
            y_withoutTags = y_withoutTags[relevant_idx]
            return x, y_withTags, y_withoutTags
        def neighbors_find(x, y_withTags, y_withoutTags, word, neighbors_num = 10):
            reference_pos = np.where(y_withoutTags == word)
            neighbors_list = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(x)
            neighbors_idx = neighbors_list.kneighbors(x[reference_pos])[1][0]
            neighbors_list = y_withTags[neighbors_idx]
            return neighbors_list, neighbors_idx
        x_oldL1, y_oldL1_withTags, y_oldL1_withoutTags = embed_filter(cent_folder + "old" + langs[0].upper() + '.txt')
        x_oldL2, y_oldL2_withTags, y_oldL2_withoutTags = embed_filter(cent_folder + "old" + langs[1].upper() + '.txt')
        x_newL1, y_newL1_withTags, y_newL1_withoutTags = embed_filter(cent_folder + "new" + langs[0].upper() + '.txt')
        x_newL2, y_newL2_withTags, y_newL2_withoutTags = embed_filter(cent_folder + "new" + langs[1].upper() + '.txt')
        x_all, y_all_withTags, y_all_withoutTags = embed_filter(cent_folder + langs[0] + '-' + langs[1] + '.txt') 
        year_1 = '1800s'
        year_2 = '1900s'
        
        neighbors_oldL1, neighbors_idx_oldL1 = neighbors_find(x_oldL1, y_oldL1_withTags, y_oldL1_withoutTags, word1)
        neighbors_oldL2, neighbors_idx_oldL2 = neighbors_find(x_oldL2, y_oldL2_withTags, y_oldL2_withoutTags, word2)
        neighbors_newL1, neighbors_idx_newL1 = neighbors_find(x_newL1, y_newL1_withTags, y_newL1_withoutTags, word1)
        neighbors_newL2, neighbors_idx_newL2 = neighbors_find(x_newL2, y_newL2_withTags, y_newL2_withoutTags, word2)

        # Create one list of all nearest neighbours
        neighbors = list(neighbors_newL1) + list(neighbors_oldL1) + list(neighbors_newL2) + list(neighbors_oldL2)

        # Find those nearest neighbors' positions in the mapped file
        neighbors_pos_in_mapped = np.isin(y_all_withoutTags, neighbors)

        # Get nearest neighbor's embeddings, names and names with tags
        relevant_embeds_long = np.squeeze(x_all[neighbors_pos_in_mapped]) # embeddings of only neighbours to reference word
        relevant_words_long = np.squeeze(y_all_withoutTags[neighbors_pos_in_mapped]) # "names" of neighbors
        relevant_words_tags_long = np.squeeze(y_all_withTags[neighbors_pos_in_mapped]) # "names" of neighbors (with tags)

        # Calculate tSNE

        tsne_result = self.fit_tsne(relevant_embeds_long)
        #n_components = 2 # number of dimensions we want after tSNE
        #tsne = TSNE(n_components, learning_rate='auto', init='random')
        #tsne_result = tsne.fit_transform(relevant_embeds_long)

        # We only want to visualise according to the new meaning of the words, but we also need the nearest neighbors of the 
        # old meaning. Here we collect the new and old close nearest neighbors and make a list with them tagged as new
        close_neighborsL1_noTags = list(set(list(y_oldL1_withoutTags[neighbors_idx_oldL1]) + 
                                   (list(y_newL1_withoutTags[neighbors_idx_newL1]))))
        close_neighborsL2_noTags = list(set((list(y_newL2_withoutTags[neighbors_idx_newL2])) + 
                                  (list(y_oldL2_withoutTags[neighbors_idx_oldL2]))))

        close_neighborsL1 = [word + '_new' + langs[0].upper() for word in close_neighborsL1_noTags]
        close_neighborsL2 = [word + '_new' + langs[1].upper() for word in close_neighborsL2_noTags]

        # In case an L2 word does not appear in the new data, we use the old embedding in the visualisation for clarity.
        for word in close_neighborsL2_noTags:
            if not (word + '_new' + langs[1].upper() in relevant_words_tags_long):
                close_neighborsL2.append(word + '_old' + langs[0].upper())
                
        # In case an L1 word does not appear in the new data, we do the same.
        for word in close_neighborsL1_noTags:
            if not (word + '_new' + langs[0].upper() in relevant_words_tags_long):
                close_neighborsL1.append(word + '_old' + langs[1].upper())

        # Find close nearest neighbors' positions in the tsne
        close_neighbors_pos_in_tsneL1 = np.where(np.isin(relevant_words_tags_long, close_neighborsL1))
        close_neighbors_pos_in_tsneL2 = np.where(np.isin(relevant_words_tags_long, close_neighborsL2))
        
        # Plot

        plt.figure(figsize=(20,10))
        plt.suptitle(f'{word1}, {word2}', y=0.1)
        plt.axis('off')

        for position in close_neighbors_pos_in_tsneL1[0]:
            plt.scatter(tsne_result[position][0], tsne_result[position][1], color='darkblue', s=9)
            plt.text(tsne_result[position][0], tsne_result[position][1], relevant_words_long[position], color='darkblue', fontsize=14)
            
        for position in close_neighbors_pos_in_tsneL2[0]:
            plt.scatter(tsne_result[position][0], tsne_result[position][1], color='darkgreen', s=9)
            plt.text(tsne_result[position][0], tsne_result[position][1], relevant_words_long[position], color='darkgreen', fontsize=14)

         
        start_posL1 = np.where(relevant_words_tags_long == word1 + '_old' + langs[0].upper())
        end_posL1 = np.where(relevant_words_tags_long == word1 + '_new' + langs[0].upper())
                
        startL1 = tsne_result[start_posL1][0]
        endL1 = tsne_result[end_posL1][0]

        plt.scatter(startL1[0], startL1[1], color='purple')
        plt.scatter(endL1[0], endL1[1], color='purple')
        plt.text(startL1[0], startL1[1], relevant_words_long[start_posL1][0]+ ' ' + year_1, color='purple', fontsize=14)
        plt.text(endL1[0], endL1[1], relevant_words_long[start_posL1][0]+ ' ' + year_2, color='purple', fontsize=14)

        #adjustText.adjust_text(texts)
        
        start_posL2 = np.where(relevant_words_tags_long == word2 + '_old' + langs[1].upper())
        end_posL2 = np.where(relevant_words_tags_long == word2 + '_new' + langs[1].upper())
                
        startL2 = tsne_result[start_posL2][0]
        endL2 = tsne_result[end_posL2][0]

        plt.scatter(startL2[0], startL2[1], color='yellowgreen')
        plt.scatter(endL2[0], endL2[1], color='yellowgreen')
        plt.text(startL2[0], startL2[1], relevant_words_long[start_posL2][0] + ' ' + year_1, color='yellowgreen', fontsize=14)
        plt.text(endL2[0], endL2[1], relevant_words_long[start_posL2][0] + ' ' + year_2, color='yellowgreen', fontsize=14)
            
        plt.annotate('', xy=(endL1[0], endL1[1]), xytext=(startL1[0], startL1[1]), arrowprops=dict(arrowstyle="->, head_width=0.6, head_length=0.6", color='purple'))
        plt.annotate('', xy=(endL2[0], endL2[1]), xytext=(startL2[0], startL2[1]), arrowprops=dict(arrowstyle="->, head_width=0.6, head_length=0.6", color='yellowgreen'))
        self.savefig("%s_%s_%s_%s.jpg" % (word1, word2, langs[0], langs[1]), out_path)