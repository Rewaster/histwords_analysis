from sequentialembedding import SequentialEmbedding
import pandas as pd
import numpy as np
import os
import random
from matplotlib import pyplot as plt

#Here you can change the parameters for the Cosine Similarity Tests

#languages
lang_list = ['eng', 'ger']

hrd_dir = 'D:\\DL4NLP\\all_dicts\\'
path_dictionary = hrd_dir + lang_list[0] + '-' + lang_list[1] + '-muse.txt'
#Hint: Make sure that the dictionary contains language 1 before language 2


path_embeddings = "D:\\DL4NLP\\aligned_embeddings\\" + lang_list[0] + '-' + lang_list[1] + '\\'

path_frequency = 'D:\\DL4NLP\\freq-dicts\\'
path_frequency_sheet_1 = path_frequency + 'internet-' + lang_list[0] + '-wf.txt'
path_frequency_sheet_2 = path_frequency + 'internet-' + lang_list[1] + '-wf.txt'


name_excel_sheet = lang_list[0] + '-' + lang_list[1]

start = 1800
end = 2000
step = 40

path_output_excel = "./cosine_sims_" + lang_list[0] + '_' + lang_list[1] + '_' + str(start) + '_' + str(end) + '_' + str(step) + ".xlsx"
path_output_csv = "./cosine_sims_" + lang_list[0] + '_' + lang_list[1] + '_' + str(start) + '_' + str(end) + '_' + str(step) + ".csv"

similarity_marker = 0.5

threshhold_change = 0.15
     

"""
Here the actual code for getting all the embeddings done starts
"""
def test_valid(word, embed):

    #French - English

    #word_list = [("œuvré","œsophagien"), ("œstradiol", "œu" )]

    #German - English

    word_list = [("égard", "époque"), ("co", "étrangères"), ("co", "œuvré"), ("œuvré","œsophagien"), ("œstradiol", "œu" ), ("übungsbuch", "üppig" )]

    for a, b in word_list:
        if(embed.__contains__(a) and embed.__contains__(b)):
            if (abs(embed.similarity(word, a) - embed.similarity(a, b))<0.05):
                return False


    
    return True
    
def read_freq_tables():

    file = open(path_frequency_sheet_1, "r", encoding = 'utf-8')
    dict_l1 = {}
    
    for line in file:
        if line != "\n":
            rank, rel_freq, word = line.split(' ')
            word = word[:-1]
            dict_l1[word] = (rank, rel_freq)
        
    file.close()
    
    file = open(path_frequency_sheet_2, "r", encoding = 'utf-8')
    dict_l2 = {}
    
    for line in file:
        if line != "\n":
            rank, rel_freq, word = line.split(' ')
            word = word[:-1]
            word = word.lower()
            dict_l2[word] = (rank, rel_freq)
        
    file.close()
    
    return dict_l1, dict_l2
    
    
def get_word_freq(word, freq_table):
    try:
        rank, freq = freq_table[word]
    except:
        freq = float("nan")
        
    return freq
    
def get_word_rank(word, freq_table):

    try:
        rank, freq = freq_table[word]
    except:
        rank = float("nan")
    return rank
    

def read_dict():
    dict_1_2 = {}
    dict_2_1 = {}
    with open(path_dictionary, "r", encoding = 'utf-8') as file:
        for line in file:
            if line != "\n":
                word_1, word_2 = line.split(' ', 1)
                word_2 = word_2[:-1]
                if(word_1 in dict_1_2.keys()):
                    dict_1_2[word_1].append(word_2)
                else:
                    dict_1_2[word_1] = [word_2]
                if(word_2 in dict_2_1.keys()):
                    dict_2_1[word_2].append(word_1)
                else:
                    dict_2_1[word_2] = [word_1]

    file.close()
    
    return dict_1_2, dict_2_1
    
def assign_category(diff, temp_change):
    temp_change = np.abs(temp_change)
    
    if(diff > threshhold_change):
        return "Convergence"
    elif(diff < -threshhold_change):
        return "Divergence" 
    elif(temp_change > threshhold_change):
        return "Parallel Change"
    else:
        return "No Semantic Change"
    

def analyze_data():

    time_frame = range(start, end, step)
    fiction_embeddings = SequentialEmbedding.load(path_embeddings, time_frame) #change languages
    dict_1, dict_2 = read_dict()
    freq_1, freq_2 = read_freq_tables()
    print(freq_1)
    
    categories = []
    
    for year in time_frame:
        categories.append(str(year))
    
    categories.append("Difference Start-End")
    categories.append("Biggest Difference")
    categories.append("Year Biggest")
    categories.append("Year Smallest")
    categories.append("Biggest Similarity")
    categories.append("Smallest Similarity")
    categories.append("Temporal Change Word 1")
    categories.append("Temporal Change Word 2")
    categories.append("Rank Word 1")
    categories.append("Frequency Word 1")
    categories.append("Rank Word 2")
    categories.append("Frequency Word 2")
    categories.append("Polysemy Word 1")
    categories.append("Polysemy Word 2")
    categories.append("Category")
    
    
    total_info = {'Word Pair': categories}
    
    for word_1 in dict_1.keys():
    
        print(word_1, get_word_freq(word_1, freq_1))
    
        for word_2 in dict_1[word_1]:
        
            all_contain_1 = True
            all_contain_2 = True
            equal = (word_1 == word_2)
            
            for year, embed in fiction_embeddings.embeds.items():
               all_contain_1 = (all_contain_1 and embed.__contains__(word_1) and (test_valid(word_1, embed)))
               all_contain_2 = (all_contain_2 and embed.__contains__(word_2) and (test_valid(word_2, embed)))
            
            time_sims = fiction_embeddings.get_time_sims(word_1, word_2)
            
            embed_first = None
            embed_last = None

            
            
            for year, embed in fiction_embeddings.embeds.items():
                if(year == time_frame[0]):
                    embed_first = embed
                elif(year == time_frame[-1]):
                    embed_last = embed
                    
            word_1_change = embed_first.represent(word_1).dot(embed_last.represent(word_1))
            word_2_change = embed_first.represent(word_2).dot(embed_last.represent(word_2))
                    
            
            word_pair = (word_1 + ", " + word_2)
            
            info = []
            
            polysemy_word_1 = len(dict_1[word_1])
            polysemy_word_2 = len(dict_2[word_2])
            
            sims = {}
            biggest = -1.0
            smallest = 1.0
            change1, change2 = False, False
            
            #print("Similarity between ", word_1, " and ", dict_1[word_1], " from 1900s to the 1990s:")
            for year, sim in time_sims.items():
                info.append(str(sim))
                if(sim >= biggest):
                    biggest = sim
                    change1 = True
                    
                if(sim <= smallest):
                    smallest = sim
                    change2 = True
                sims[sim] = year
                #print("{year:d}, cosine similarity={sim:0.2f}".format(year=year,sim=sim))
                

            if(change1 and change2):
                if(sims[smallest] > sims[biggest]):
                    biggest_diff = smallest - biggest
                else:
                    biggest_diff = biggest - smallest
                year_smallest = sims[smallest]
                year_biggest = sims[biggest]
            else:
                biggest_diff = 0
                year_smallest = float("nan")
                year_biggest = float("nan")                
            
            #Optional Filter to make sure only actually "similar" words get into file
            if(biggest < similarity_marker or biggest >= 0.999 or biggest == 0.8939662503622092 or biggest == 0.9788323264090899 or biggest == 0.943592763032786 or biggest == 0.7871050578466212 or biggest == 0.9360291047366865 or biggest == 0.9559455846214017 or biggest == 0.9195477336067494): #this is weirdly necessary specifically for french embeddings
                break
            
            if(len(info) >= 2 and all_contain_1 and all_contain_2 and (not equal)): #change len here for more sim reading
                diff = float(info[-1]) - float(info[0])
                info.append(diff)
                info.append(biggest_diff)
                info.append(year_biggest)
                info.append(year_smallest)
                info.append(biggest)
                info.append(smallest)
                info.append(word_1_change)
                info.append(word_2_change)
                info.append(get_word_rank(word_1, freq_1))
                info.append(get_word_freq(word_1, freq_1))
                info.append(get_word_rank(word_2, freq_2))
                info.append(get_word_freq(word_2, freq_2))
                info.append(polysemy_word_1)
                info.append(polysemy_word_2)
                info.append(assign_category(biggest_diff, ((word_1_change+word_2_change)/2)))
                total_info[word_pair] = info
        
    df1 = pd.DataFrame(total_info)
    
    df1 = df1.swapaxes("columns", "index")

    writer = pd.ExcelWriter(path_output_excel, engine='xlsxwriter')
    
    df1.to_excel(writer, sheet_name=name_excel_sheet, index=True)
    writer.save()
    
    df1.to_csv(path_output_csv, index=True)
    
    return
    

def get_avg_sim_nearest():

    time_frame = range(start, end, step)
    fiction_embeddings = SequentialEmbedding.load(path_embeddings, time_frame) #change languages
    dict_1, dict_2 = read_dict()
    
    
    embed = fiction_embeddings.embeds[1950]
    sim_list = []
    counter = 0
    words = list(dict_1.keys())
    random.shuffle(words)
    for word_1 in words:
        if(counter > 1000):
            break
        if(test_valid(word_1, embed)):
            avg_sim = 0
            n = 10
            for sim, word_2 in embed.closest(word_1, n=n):
                avg_sim += (sim/n)
            if(not (avg_sim >= 0.99)):
                sim_list.append(avg_sim)
                print(word_1, avg_sim)
                counter += 1
                
           
            
    avg_sim_total = sum(sim_list)/len(sim_list)
    print("The average similarity of a words 10 nearest neighbors in ", 1990, " is ", avg_sim_total, ".\n")
  
    return
    
    
if __name__ == "__main__":
        
    analyze_data()
    
    
    

    
        


