---
layout: post
title: Do All Playboi Carti Songs Sound The Same - A Study
subtitle: Do they?
tags: [math, tech]
---
![Alt Text](https://media0.giphy.com/media/McmAF0TXnXNZPBqBud/giphy.gif)

<p class="music-read"><a href="spotify:track:0oJHQgG9iC2imbuRjzNKZE?si=d345221d92004cde">Click Me</a></p>

### As I learn more and more in my postgraduate degree in mathematics, I can't help but return back to the age-old question - the question everyone wants an answer to - "do all Playboi Carti songs sound the same?". Luckily, now we can actually mathematically find out. Let's do some data visualisation and see whats going on with Jordan Carter's music:

```python
#import python modules for data manipulation and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
```


```python
#import the spotipy modules
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

cid ="xx"
secret = "xx"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
```

Here we're creating some empty lists to store the necessary values and create a dataframe. I'm using the package spotipy to get all the information I need. NEED:


```python
# create empty lists where the results are going to be stored
artist_name = []
album_name = []
track_name = []
popularity = []
track_id = []

#get all relevant info for Ariana Grande tracks via a 'search' query to the API
#this took about 2 seconds to run
for i in range(0,400,50):
    track_results = sp.search(q='artist:"playboi carti"', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        album_name.append(t['album']['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
```


```python
#turn lists into a df
df_carti_tracks = pd.DataFrame({'artist_name':artist_name,
                                'album_name':album_name,
                                'track_name':track_name,
                                'track_id':track_id,
                                'popularity':popularity})
```


```python
#let's just look at Carti's last 3 albums
df_carti_tracks = df_carti_tracks[df_carti_tracks['album_name'].isin(['Playboi Carti', 'Die Lit', 'Whole Lotta Red'])]
df_carti_tracks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Playboi Carti</td>
      <td>Whole Lotta Red</td>
      <td>Sky</td>
      <td>29TPjc8wxfz4XMn21O7VsZ</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Playboi Carti</td>
      <td>Die Lit</td>
      <td>Shoota (feat. Lil Uzi Vert)</td>
      <td>2BJSMvOGABRxokHKB0OI8i</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Playboi Carti</td>
      <td>Playboi Carti</td>
      <td>Magnolia</td>
      <td>1e1JKLEDKP7hEQzJfNAgPl</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Playboi Carti</td>
      <td>Die Lit</td>
      <td>Fell In Luv (feat. Bryson Tiller)</td>
      <td>1s9DTymg5UQrdorZf43JQm</td>
      <td>78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Playboi Carti</td>
      <td>Playboi Carti</td>
      <td>Location</td>
      <td>3yk7PJnryiJ8mAPqsrujzf</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>



So we have our dataframe of Carti songs. As we can see, 'Sky' is the most popular - and this is why I hate TikTok. (The real answer is... actually I'm gonna wait until the end of this to make sure you read it)

Spotify describes its audio features as follows:

- **acousticness**: a confidence measure of whether the track is acoustic (0.0-1.0).
- **danceability**: how suitable a track is for dancing based on tempo, rhythm stability, beat strength, overall regularity and other elements (0.0-1.0).
- **energy**: a perceptual measure of intensity and activity, or how ‘fast, loud and noisy’ a track is (0.0-1.0).
- **loudness**: loudness in dB (averaged across the track, typically between -60.0-0.0).
- **instrumentalness**: a confidence measure of whether a track contains no vocals/lyrics (0.0-1.0).
- **liveness**: probability that the track was performed live, based on whether an audience is present.
- **speechiness**: detects presence of spoken words in a track (0.0-1.0).
- **valence**: the musical ‘positiveness’ conveyed by a track, i.e. sounding happy, cheerful or euphoric rather than sad, depressed or angry (0.0-1.0).

Let's pull all the audio features using the .audio_features method:


```python
# empty list, batchsize and the counter for None results
rows = []
batchsize = 100
None_counter = 0

for i in range(0,len(df_carti_tracks['track_id']),batchsize):
    batch = df_carti_tracks['track_id'][i:i+batchsize]
    feature_results = sp.audio_features(batch)
    for i, t in enumerate(feature_results):
        if t == None:
            None_counter = None_counter + 1
        else:
            rows.append(t)

print('Number of tracks where no audio features were available:',None_counter)
```

    Number of tracks where no audio features were available: 0



```python
df_carti_tracks['danceability'] = [i['danceability'] for i in feature_results]
df_carti_tracks['energy'] = [i['energy'] for i in feature_results]
df_carti_tracks['loudness'] = [i['loudness'] for i in feature_results]
df_carti_tracks['speechiness'] = [i['speechiness'] for i in feature_results]
df_carti_tracks['acousticness'] = [i['acousticness'] for i in feature_results]
df_carti_tracks['instrumentalness'] = [i['instrumentalness'] for i in feature_results]
df_carti_tracks['liveness'] = [i['liveness'] for i in feature_results]
df_carti_tracks['valence'] = [i['valence'] for i in feature_results]
```


```python
df_carti_tracks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th></th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Playboi Carti</td>
      <td>Whole Lotta Red</td>
      <td>Sky</td>
      <td>29TPjc8wxfz4XMn21O7VsZ</td>
      <td>83</td>
      <td>0.785</td>
      <td>0.903</td>
      <td>-4.184</td>
      <td>0.210</td>
      <td>0.2580</td>
      <td>0.000000</td>
      <td>0.169</td>
      <td>0.565</td>
      <td>0.565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Playboi Carti</td>
      <td>Die Lit</td>
      <td>Shoota (feat. Lil Uzi Vert)</td>
      <td>2BJSMvOGABRxokHKB0OI8i</td>
      <td>80</td>
      <td>0.673</td>
      <td>0.649</td>
      <td>-8.433</td>
      <td>0.196</td>
      <td>0.1880</td>
      <td>0.000000</td>
      <td>0.122</td>
      <td>0.470</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Playboi Carti</td>
      <td>Playboi Carti</td>
      <td>Magnolia</td>
      <td>1e1JKLEDKP7hEQzJfNAgPl</td>
      <td>79</td>
      <td>0.791</td>
      <td>0.582</td>
      <td>-7.323</td>
      <td>0.286</td>
      <td>0.0114</td>
      <td>0.000000</td>
      <td>0.350</td>
      <td>0.443</td>
      <td>0.443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Playboi Carti</td>
      <td>Die Lit</td>
      <td>Fell In Luv (feat. Bryson Tiller)</td>
      <td>1s9DTymg5UQrdorZf43JQm</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.668</td>
      <td>-6.208</td>
      <td>0.136</td>
      <td>0.0273</td>
      <td>0.000000</td>
      <td>0.320</td>
      <td>0.227</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Playboi Carti</td>
      <td>Playboi Carti</td>
      <td>Location</td>
      <td>3yk7PJnryiJ8mAPqsrujzf</td>
      <td>76</td>
      <td>0.717</td>
      <td>0.790</td>
      <td>-4.213</td>
      <td>0.200</td>
      <td>0.3300</td>
      <td>0.000125</td>
      <td>0.518</td>
      <td>0.371</td>
      <td>0.371</td>
    </tr>
  </tbody>
</table>
</div>



We have our dataframe! This is actually really cool - below I've used the .describe method to show you how the features vary. Take a look at the mean, min and max values. So now we've gathered one thing that we know for sure (in case you didn't previously know this): Carti's songs are danceable! This also depends on who you are as a person but I won't go into that :)


```python
df_characteristics = df_carti_tracks[['danceability','energy','loudness','speechiness',
                                      'acousticness','instrumentalness','liveness','valence']]

df_characteristics.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
      <td>58.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.762328</td>
      <td>0.642810</td>
      <td>-5.974672</td>
      <td>0.204453</td>
      <td>0.080911</td>
      <td>0.000035</td>
      <td>0.218678</td>
      <td>0.396116</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.105690</td>
      <td>0.119446</td>
      <td>1.315008</td>
      <td>0.108011</td>
      <td>0.105680</td>
      <td>0.000205</td>
      <td>0.145833</td>
      <td>0.198384</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.437000</td>
      <td>0.408000</td>
      <td>-10.227000</td>
      <td>0.033300</td>
      <td>0.000674</td>
      <td>0.000000</td>
      <td>0.072600</td>
      <td>0.040000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.700500</td>
      <td>0.570000</td>
      <td>-6.624750</td>
      <td>0.109250</td>
      <td>0.011475</td>
      <td>0.000000</td>
      <td>0.114000</td>
      <td>0.240750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.757000</td>
      <td>0.633000</td>
      <td>-5.746500</td>
      <td>0.214500</td>
      <td>0.030600</td>
      <td>0.000000</td>
      <td>0.157000</td>
      <td>0.415000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.841750</td>
      <td>0.720000</td>
      <td>-5.083000</td>
      <td>0.287500</td>
      <td>0.100725</td>
      <td>0.000000</td>
      <td>0.321500</td>
      <td>0.511000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.956000</td>
      <td>0.946000</td>
      <td>-3.593000</td>
      <td>0.385000</td>
      <td>0.478000</td>
      <td>0.001550</td>
      <td>0.830000</td>
      <td>0.948000</td>
    </tr>
  </tbody>
</table>
</div>



Let's also take a look at how the different features behave against one another, and we'll also make a correlation matrix of the data:


```python
sns.pairplot(df_characteristics)
```




    <seaborn.axisgrid.PairGrid at 0x7f9932bb0d30>




![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_15_1.png)



```python
df_characteristics.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>danceability</th>
      <td>1.000000</td>
      <td>-0.202824</td>
      <td>-0.164464</td>
      <td>0.277217</td>
      <td>-0.107395</td>
      <td>-0.067340</td>
      <td>-0.370464</td>
      <td>0.223903</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>-0.202824</td>
      <td>1.000000</td>
      <td>0.328701</td>
      <td>-0.344519</td>
      <td>0.085728</td>
      <td>-0.253888</td>
      <td>0.172265</td>
      <td>0.122383</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>-0.164464</td>
      <td>0.328701</td>
      <td>1.000000</td>
      <td>-0.521344</td>
      <td>0.171525</td>
      <td>-0.190320</td>
      <td>0.182833</td>
      <td>-0.330256</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>0.277217</td>
      <td>-0.344519</td>
      <td>-0.521344</td>
      <td>1.000000</td>
      <td>0.032378</td>
      <td>0.191928</td>
      <td>-0.126825</td>
      <td>0.380421</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>-0.107395</td>
      <td>0.085728</td>
      <td>0.171525</td>
      <td>0.032378</td>
      <td>1.000000</td>
      <td>-0.073379</td>
      <td>0.320131</td>
      <td>-0.003815</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>-0.067340</td>
      <td>-0.253888</td>
      <td>-0.190320</td>
      <td>0.191928</td>
      <td>-0.073379</td>
      <td>1.000000</td>
      <td>-0.082940</td>
      <td>-0.175039</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>-0.370464</td>
      <td>0.172265</td>
      <td>0.182833</td>
      <td>-0.126825</td>
      <td>0.320131</td>
      <td>-0.082940</td>
      <td>1.000000</td>
      <td>-0.090327</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>0.223903</td>
      <td>0.122383</td>
      <td>-0.330256</td>
      <td>0.380421</td>
      <td>-0.003815</td>
      <td>-0.175039</td>
      <td>-0.090327</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Speechiness and valence seem to have a high correlation. Let's take a look:


```python
import matplotlib.patches

plt.figure(figsize = (15,10))

levels, categories = pd.factorize(df_carti_tracks['album_name'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df_carti_tracks['speechiness'], df_carti_tracks['valence'], c=colors)
plt.gca().set(xlabel='Speechiness: detects presence of spoken words in a track (0.0-1.0)',
              ylabel='Valence: the musical ‘positiveness’ conveyed by a track (0.0-1.0)',
              title='Speechiness vs Valence')

plt.legend(handles=handles, title='Album')

plt.text(df_carti_tracks.speechiness[df_carti_tracks.track_name=='New N3on']+0.005,
         df_carti_tracks.valence[df_carti_tracks.track_name=='New N3on'],"New N3on")
plt.text(df_carti_tracks.speechiness[df_carti_tracks.track_name=='On That Time']+0.005,
         df_carti_tracks.valence[df_carti_tracks.track_name=='On That Time'],"On That Time")
plt.text(df_carti_tracks.speechiness[df_carti_tracks.track_name=='Punk Monk']+0.005,
         df_carti_tracks.valence[df_carti_tracks.track_name=='Punk Monk'],"Punk Monk")
plt.text(df_carti_tracks.speechiness[df_carti_tracks.track_name=='Location']+0.005,
         df_carti_tracks.valence[df_carti_tracks.track_name=='Location'],"Location")
plt.text(df_carti_tracks.speechiness[df_carti_tracks.track_name=='Yah Mean']+0.005,
         df_carti_tracks.valence[df_carti_tracks.track_name=='Yah Mean'],"Yah Mean")
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_18_0.png)


So we can see that as speechiness increases (more words... lol), valence also tends to increase. This means the more words Carti says, the more positive the song is. I can't stop laughing at this result to be honest with you.

![Alt Text](https://media0.giphy.com/media/l1J3KFPMztOKgcp32/giphy.gif?cid=ecf05e47uz39p8nopov6f4smzh98vtq7om1fd7sixkmbbcun&rid=giphy.gif&ct=g)

Now we're going to use a tSNE plot (a method used for dimensionality reduction - since we have 8 dimensions, one for each characteristic, we need to use this method to essentially map the data onto a 2D plane). More information on tSNE can be found online, and there are some great youtube videos on it. Let's run the code to do this:


```python
#import python modules
from sklearn.preprocessing import  MinMaxScaler
from sklearn.manifold import TSNE

#normalise data
scaler = MinMaxScaler()
df_carti_audio_features_final = pd.DataFrame(scaler.fit_transform(df_characteristics), columns = df_characteristics.columns)

#run tSNE with perplexity 20
df_carti_tsne_values = df_carti_audio_features_final.values
df_carti_tsne_values

carti_tsne = TSNE(
    n_components=2,
    perplexity=6,
    verbose=2).fit_transform(df_carti_tsne_values)

#attach tsne variables to original dataset
df_final = df_carti_tracks
df_final['tsne1'] = carti_tsne[:,0]
df_final['tsne2'] = carti_tsne[:,1]
```

    [t-SNE] Computing 19 nearest neighbors...
    [t-SNE] Indexed 58 samples in 0.000s...
    [t-SNE] Computed neighbors for 58 samples in 0.001s...
    [t-SNE] Computed conditional probabilities for sample 58 / 58
    [t-SNE] Mean sigma: 0.212884
    [t-SNE] Computed conditional probabilities in 0.001s
    [t-SNE] Iteration 50: error = 73.5875397, gradient norm = 0.3818989 (50 iterations in 0.030s)
    [t-SNE] Iteration 100: error = 65.7292938, gradient norm = 0.4719713 (50 iterations in 0.013s)
    [t-SNE] Iteration 150: error = 64.5578613, gradient norm = 0.4816838 (50 iterations in 0.011s)
    [t-SNE] Iteration 200: error = 75.7814484, gradient norm = 0.3106402 (50 iterations in 0.009s)
    [t-SNE] Iteration 250: error = 66.0760651, gradient norm = 0.5483950 (50 iterations in 0.010s)
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 66.076065
    [t-SNE] Iteration 300: error = 1.7484283, gradient norm = 0.0068930 (50 iterations in 0.011s)
    [t-SNE] Iteration 350: error = 1.3430145, gradient norm = 0.0016589 (50 iterations in 0.015s)
    [t-SNE] Iteration 400: error = 1.1108661, gradient norm = 0.0013740 (50 iterations in 0.012s)
    [t-SNE] Iteration 450: error = 1.8232965, gradient norm = 0.0026278 (50 iterations in 0.009s)
    [t-SNE] Iteration 500: error = 1.5070864, gradient norm = 0.0036373 (50 iterations in 0.010s)
    [t-SNE] Iteration 550: error = 1.4013243, gradient norm = 0.0006672 (50 iterations in 0.010s)
    [t-SNE] Iteration 600: error = 1.3269188, gradient norm = 0.0002550 (50 iterations in 0.009s)
    [t-SNE] Iteration 650: error = 1.2616950, gradient norm = 0.0002884 (50 iterations in 0.010s)
    [t-SNE] Iteration 700: error = 1.1635824, gradient norm = 0.0006236 (50 iterations in 0.009s)
    [t-SNE] Iteration 750: error = 1.0290606, gradient norm = 0.0003722 (50 iterations in 0.010s)
    [t-SNE] Iteration 800: error = 0.9731370, gradient norm = 0.0002022 (50 iterations in 0.009s)
    [t-SNE] Iteration 850: error = 0.9165962, gradient norm = 0.0002257 (50 iterations in 0.010s)
    [t-SNE] Iteration 900: error = 0.8514358, gradient norm = 0.0001298 (50 iterations in 0.020s)
    [t-SNE] Iteration 950: error = 0.8309491, gradient norm = 0.0001405 (50 iterations in 0.020s)
    [t-SNE] Iteration 1000: error = 0.7976543, gradient norm = 0.0001546 (50 iterations in 0.014s)
    [t-SNE] KL divergence after 1000 iterations: 0.797654



```python
plt.figure(figsize = (15,10))

levels, categories = pd.factorize(df_carti_tracks['album_name'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df_carti_tracks['tsne1'], df_carti_tracks['tsne2'], c=colors)
plt.gca().set(xlabel='tsne1',
              ylabel='tsne2',
              title='t-SNE')

plt.legend(handles=handles, title='Album')
plt.show()
#for i in range(58):
#    plt.annotate(df_carti_tracks.track_name.iloc[i], (df_carti_tracks.tsne1.iloc[i], df_carti_tracks.tsne2.iloc[i] + 0.2))
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_21_0.png)


The points span a large amount of the space, and we can see some clusters forming. I tested various perplexity values and 6 pulled clusters together without contraining them too much. We can clearly see variability as well as similarities between some songs! Let's take a look at the clusters:


```python
plt.figure(figsize = (15,10))

levels, categories = pd.factorize(df_carti_tracks['album_name'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df_carti_tracks['tsne1'], df_carti_tracks['tsne2'], c=colors)
plt.gca().set(xlabel='tsne1',
              ylabel='tsne2',
              title='t-SNE')

plt.legend(handles=handles, title='Album')

plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Long Time - Intro']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Long Time - Intro']+5,"Long Time - Intro")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Location']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Location']+5,"Location")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Flex']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Flex']+5,"Flex")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Home (KOD)']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Home (KOD)']+5,"Home (KOD)")

plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_23_0.png)


Long Time, Location, Home, and Flex are being pulled together - that kind of makes sense to me because these are all pretty spacey, airy beats with a lot of distance between the listener and the audio.


```python
plt.figure(figsize = (15,10))

levels, categories = pd.factorize(df_carti_tracks['album_name'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df_carti_tracks['tsne1'], df_carti_tracks['tsne2'], c=colors)
plt.gca().set(xlabel='tsne1',
              ylabel='tsne2',
              title='t-SNE')

plt.legend(handles=handles, title='Album')

plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Half & Half']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Half & Half']+5,"Half & Half")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Other Shit']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Other Shit']+8,"Other Shit")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Go2DaMoon (feat. Kanye West)']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Go2DaMoon (feat. Kanye West)']+3,"Go2DaMoon (feat. Kanye West)")

plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_25_0.png)


Half & Half, Other Sh-t, Go2DaMoon are clustering. These all have pretty bouncy 808s with space between the sample one-shots (I understand it may seem like I'm just making stuff up at this point but I'm not).


```python
plt.figure(figsize = (15,10))

levels, categories = pd.factorize(df_carti_tracks['album_name'])
colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

plt.scatter(df_carti_tracks['tsne1'], df_carti_tracks['tsne2'], c=colors)
plt.gca().set(xlabel='tsne1',
              ylabel='tsne2',
              title='t-SNE')

plt.legend(handles=handles, title='Album')

plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Fell In Luv (feat. Bryson Tiller)']-160,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Fell In Luv (feat. Bryson Tiller)'],"Fell In Luv (feat. Bryson Tiller)")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Lean 4 Real (feat. Skepta)']-140,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Lean 4 Real (feat. Skepta)']-20,"Lean 4 Real (feat. Skepta)")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='Control']+5,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='Control']+5,"Control")
plt.text(df_carti_tracks.tsne1[df_carti_tracks.track_name=='ILoveUIHateU']-50,
         df_carti_tracks.tsne2[df_carti_tracks.track_name=='ILoveUIHateU']+10,"ILoveUIHateU")

plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/carti/output_27_0.png)


I'm really not sure how to explain this one, especially Lean 4 Real - maybe one of you can help me out. Message me what you think.

### So we have an answer - NOT ALL CARTI SONGS SOUND THE SAME. But you already knew that, didn't you?
### See you next time (Lil Uzi?)

![Alt Text](https://media0.giphy.com/media/l4FssXeliObIRSCmQ/giphy.gif?cid=ecf05e47uq3eqixaz6t4l4gdv029czyyeszaf4k0w5s73a5d&rid=giphy.gif&ct=g)


```python

```
