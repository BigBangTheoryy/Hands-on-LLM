import pandas as pd
from urllib import request
from gensim.models import Word2Vec
import numpy as np

#Getting Playlist Dataset file:
link = "https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt"
data =  request.urlopen(link)

#Parse playlist and skip first 2 lines as it contains metadata:
lines = data.read().decode("utf-8").split("\n")[2:]

#Remove playlist with only 1 song:
playlists = [line.rstrip().split() for line in lines if len(line.split()) > 1 ]

#Load Song Metadata:
song_link = "https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt"
songs_file = request.urlopen(song_link)

#Loading Songs Metadata
songs_file = songs_file.read().decode("utf-8").split("\n")

songs = [song.rstrip().split("\t") for song in songs_file]

songs_df = pd.DataFrame(data = songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index("id")

#print(playlists[0])


#Training the Word2Vec Model
model = Word2Vec(playlists, vector_size = 32, window = 20, negative = 50, min_count = 1, workers = 4)

#Ask the model for song similar to song_id = 2172
song_id = 2172

similar_songs = model.wv.most_similar(positive=str(song_id))

#print(similar_songs) #[('3167', 0.9980922937393188), ('2849', 0.9977917671203613), ('2976', 0.9976602792739868), ('5586', 0.996351957321167), ('5634', 0.9963517189025879), ('3105', 0.9959948658943176), ('3094', 0.995680570602417), ('2704', 0.9954226613044739), ('3116', 0.9952256679534912), ('9994', 0.9949663877487183)]

#Printing the details of the song
print(songs_df.iloc[song_id])

def print_song_recommendations(song_id):
    similar_song = np.array(model.wv.most_similar(positive=str(song_id)))[:, 0]

    return songs_df.iloc[similar_song]


print(print_song_recommendations(song_id))