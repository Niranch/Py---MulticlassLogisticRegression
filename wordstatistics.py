# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:41:35 2020

@author: Niranch
"""

import pandas as pd 
import numpy as np 
from collections import Counter

#Code to process lengthy text to get word statistics. for eg. text from Hamlet 
#can be read by this code and provide details on number of words, number of unique words,
#the words that appear frequently and how they differ from other language counterparts etc.,


#From the given text, identify and replace special characters.
#Generate word count using Counter after removing spaces
def count_words_fast(text): 
    text = text.lower() 
    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"] 
    for ch in skips: 
        text = text.replace(ch, "") 
    word_counts = Counter(text.split(" ")) 
    return word_counts

#Given language and text, get the statistics and generate a dataframe with added details
def summarize_text(language, text):
    counted_text = count_words_fast(text)
 
#Create a data frame called data and assign the dicitonary elements to it's col
    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })
    
#Create additional col called "frequency" and assign value based on the value
#from col "count"
    data.loc[data["count"] > 10,  "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1,  "frequency"] = "unique"
    print(data.head(5))
#Add another col called "length" and assign length of the word   
    data["length"] = data["word"].apply(len)

#Generate a seperate data frame by aggregating data grouping by frequency
    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ["frequent","infrequent","unique"],
        "mean_word_length": data.groupby(by = "frequency")["length"].mean(),
        "num_words": data.groupby(by = "frequency").size()
    })
    print(sub_data.head(5))
    return(sub_data)
    

hamlets = pd.read_csv("\\LanguageProcessing\\hamlets.csv",index_col=0)
#Generating info on Hamlet
language, text = hamlets.iloc[0]
sub_data=summarize_text(language,text)
print(sub_data)

grouped_data = pd.DataFrame(columns = ["language", "frequency", "mean_word_length", "num_words"])
for i in range(hamlets.shape[0]):
    language, text = hamlets.iloc[i]
    sub_data = summarize_text(language, text)
    grouped_data = grouped_data.append(sub_data)
    

#Plotting Data

colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o","infrequent": "s", "unique": "^"}
import matplotlib.pyplot as plt
for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words,
        marker=markers[row.frequency],
        color = colors[row.language],
        markersize = 10
    )

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
        color=colors[color],
        marker="o",
        label = color, markersize = 10, linestyle="None")
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
        color="k",
        marker=markers[marker],
        label = marker, markersize = 10, linestyle="None")
    )
plt.legend(numpoints=1, loc = "upper left")

plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
plt.savefig("langplot.pdf")


# the graph indicates that German language has the highest number of unique words
#portugese follows it closely. All three languages, English, Portugese and German
#have low number of repeated words










