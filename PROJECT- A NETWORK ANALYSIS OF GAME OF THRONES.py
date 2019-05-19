#Load dataset

import pandas as pd

book1 = pd.read_csv('datasets/book1.csv')
book1.head()

# Creatin the Network

import networkx as nx

G_book1 = nx.Graph()
for _, edge in book1.iterrows():
    G_book1.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])

books = [G_book1]
book_fnames = ['datasets/book2.csv', 'datasets/book3.csv', 'datasets/book4.csv', 'datasets/book5.csv']
for i in book_fnames:
    G_book = nx.Graph()
    book = pd.read_csv(i)
    for _, edge in book.iterrows():
        G_book.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
    books.append(G_book)

# Calculating de degree of centrality of book 1 and 5

deg_cen_book1 = nx.degree_centrality(books[0])
deg_cen_book5 = nx.degree_centrality(books[4])

sorted_deg_cen_book1 = sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:10]
sorted_deg_cen_book5 = sorted(deg_cen_book5.items(), key=lambda x:x[1], reverse=True)[0:10]

print(sorted_deg_cen_book1, sorted_deg_cen_book5)

# Evolution of degree centrality of Eddard, Jon and Tyrion

evol = [nx.degree_centrality(i) for i in books]
degree_evol_df = pd.DataFrame.from_records(evol)
degree_evol_df[['Eddard-Stark', 'Tyrion-Lannister','Jon-Snow']].plot()

# Ploting the evolution of betweenness centrality of this network over the five books

evol = [nx.betweenness_centrality(i, weight='weight') for i in books]
betweenness_evol_df = pd.DataFrame.from_records(evol)

set_of_char = set()
for i in range(5):
    set_of_char |= set(list(betweenness_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

betweenness_evol_df[list_of_char].plot(figsize=(13, 7))

# Ploting the evolution of the charachters base on PageRank

evol = [nx.pagerank(i, weight = 'weight') for i in books]
pagerank_evol_df = pd.DataFrame.from_records(evol)

set_of_char = set()
for i in range(5):
    set_of_char |= set(list(pagerank_evol_df.T[i].sort_values(ascending=False)[0:4].index))
list_of_char = list(set_of_char)

pagerank_evol_df[list_of_char].plot(figsize=(13,7))

# Comparing the Degree of centrality, the betweenness of centrality and the Pagerank evolution.

measures = [nx.pagerank(books[4]),
            nx.betweenness_centrality(books[4], weight='weight'),
            nx.degree_centrality(books[4])]

cor = pd.DataFrame.from_records(measures)
cor.T.corr()

# What is the most important character base on the measures?
p_rank, b_cent, d_cent = cor.idxmax(axis=1)
print(p_rank, b_cent, d_cent)
