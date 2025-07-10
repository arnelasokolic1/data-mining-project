#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

# Load the dataset from the given path
file_path = r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first 5 rows
print(df.head())


# In[2]:


df.head (10)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


num_records = len(df)

print('Total numbers of records: {}'.format(num_records))

print(df.shape[0])
print(df.shape[1])


# In[6]:


# Check for missing values in the dataset
print(df.isnull().sum())

# Check for duplicate rows in the dataset
print(df.duplicated().any())


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset 
df = pd.read_csv("C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv")

# Convert 'Runtime' column to numeric by removing ' min' and converting to float
df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(float)

# Convert 'Gross' column to numeric by removing commas and converting to float
df["Gross"] = df["Gross"].str.replace(",", "").astype(float)

# Choose a numeric column for analysis
numeric_column = "IMDB_Rating"  # You can change this to 'Meta_score', 'Runtime', or 'Gross'

# Calculate statistics
mean_value = df[numeric_column].mean()
median_value = df[numeric_column].median()
mode_value = df[numeric_column].mode()[0]
midrange = (df[numeric_column].min() + df[numeric_column].max()) / 2

# Compute quartiles
q1 = df[numeric_column].quantile(0.25)
q3 = df[numeric_column].quantile(0.75)
min_value = df[numeric_column].min()
max_value = df[numeric_column].max()

# Five-number summary
five_number_summary = {
    "Minimum": min_value,
    "Q1": q1,
    "Median": median_value,
    "Q3": q3,
    "Maximum": max_value
}

# Print statistics
print("\nStatistics:")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"Midrange: {midrange}")
print(f"Five-Number Summary: {five_number_summary}")

# Visualizing distribution
plt.figure(figsize=(10, 5))
sns.histplot(df[numeric_column], bins=20, kde=True, color='blue')
plt.xlabel(numeric_column)
plt.ylabel("Frequency")
plt.title(f"Distribution of {numeric_column}")
plt.show()


# In[11]:


import pandas as pd

df = pd.read_csv(r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv")

# Filling missing numbers with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Filling missing text with mode
df.fillna(df.mode().iloc[0], inplace=True)

print(df.isnull().sum())  


# In[12]:


print("Duplicates before:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after:", df.duplicated().sum())


# In[13]:


# Convert 'Released_Year' to numeric, coercing errors to NaN
df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors='coerce')

# Convert to integer while keeping NaN values
df["Released_Year"] = df["Released_Year"].astype("Int64")

# Check new data types
print(df.dtypes)


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=df["IMDB_Rating"])
plt.show()


# In[15]:


sns.histplot(df["IMDB_Rating"], bins=10, kde=True)
plt.show()


# In[16]:


df["Decade"] = (df["Released_Year"] // 10) * 10
df["Decade"].value_counts().sort_index().plot(kind="bar")
plt.show()


# In[17]:


print(df.isnull().sum())  # Check missing values in each column


# In[18]:


most_common_year = df["Released_Year"].mode()[0]
df["Released_Year"].fillna(most_common_year, inplace=True)


# In[19]:


df["Decade"] = (df["Released_Year"] // 10) * 10


# In[20]:


print(df.isnull().sum())  


# In[21]:


df["Gross"] = df["Gross"].replace(",", "", regex=True).astype(float)


# In[22]:


df.info()


# In[23]:


df = df[(df["Released_Year"] >= 1900) & (df["Released_Year"] <= 2025)]  # Keep valid years only


# In[24]:


print(df["Released_Year"].min(), df["Released_Year"].max())


# In[25]:


print(df[~df["Released_Year"].between(1900, 2025)])  


# In[26]:


print("Rows before filtering:", len(df))
df = df[(df["Released_Year"] >= 1900) & (df["Released_Year"] <= 2025)]
print("Rows after filtering:", len(df))


# In[1]:


#MILESTONE 2 STARTING FROM HERE 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Set style
sns.set_style("whitegrid")


# In[3]:


plt.figure(figsize=(10, 6))
sns.histplot(df['IMDB_Rating'], bins=20, kde=True, color='blue')
plt.title('Distribution of IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Frequency')
plt.show()


# In[4]:


top_rated = df.nlargest(10, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating']]
print(top_rated)


# In[5]:


top_rated = df.nlargest(10, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating']]

# Plot top-rated movies
plt.figure(figsize=(12, 6))
sns.barplot(y=top_rated['Series_Title'], x=top_rated['IMDB_Rating'], palette='Blues_r')
plt.title('Top 10 Highest Rated Movies')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.xlim(8.5, 10)  
plt.show()


# In[6]:


plt.figure(figsize=(12, 6))
sns.countplot(y=df['Director'], order=df['Director'].value_counts().head(10).index, palette='viridis')
plt.title('Top 10 Directors with Most Movies')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()


# In[7]:


plt.figure(figsize=(10, 6))
sns.histplot(df['IMDB_Rating'], bins=20, kde=False, color='blue')
plt.title('Distribution of IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Frequency')
plt.show()


# In[8]:


top_voted_movies = df.nlargest(10, 'No_of_Votes')[['Series_Title', 'No_of_Votes']]

plt.figure(figsize=(12, 6))
sns.barplot(y=top_voted_movies['Series_Title'], x=top_voted_movies['No_of_Votes'], palette='Blues_r')
plt.title('Top 10 Movies with the Most Votes')
plt.xlabel('Number of Votes')
plt.ylabel('Movie Title')
plt.show()


# In[9]:


plt.figure(figsize=(14, 7))
sns.barplot(x=df['Released_Year'].value_counts().index, 
            y=df['Released_Year'].value_counts().values, 
            palette='magma')
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.xticks(rotation=90)
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
sns.barplot(y=df['Certificate'].value_counts().index, 
            x=df['Certificate'].value_counts().values, 
            palette='pastel')
plt.title('Distribution of Movie Certificates')
plt.xlabel('Number of Movies')
plt.ylabel('Certificate')
plt.show()


# In[11]:


df = pd.read_csv(r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv")

print(df.head())  # Check if Genre column has values again


# In[12]:


df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')
genre_counts = df_exploded['Genre'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='coolwarm')
plt.title('Top 10 Most Common Movie Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()


# In[13]:


# Combine all stars into a single column
all_actors = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']])

# Count the occurrences of each actor
actor_counts = all_actors.value_counts().head(10)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=actor_counts.index, x=actor_counts.values, palette='rocket')
plt.title('Top 10 Most Frequent Lead Actors')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.show()


# In[14]:


# Combine all actors into a single column
all_actors = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']])

# Count occurrences
actor_counts = all_actors.value_counts().head(15)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=actor_counts.index, x=actor_counts.values, palette='magma')
plt.title('Top 15 Most Frequent Actors in IMDb Top 1000')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.show()


# In[15]:


# Filter movies where Al Pacino is in any of the star columns
al_pacino_movies = df[(df['Star1'] == 'Al Pacino') | 
                      (df['Star2'] == 'Al Pacino') | 
                      (df['Star3'] == 'Al Pacino') | 
                      (df['Star4'] == 'Al Pacino')]

# Count movies
num_al_pacino_movies = len(al_pacino_movies)

print(f'Al Pacino has acted in {num_al_pacino_movies} movies in the IMDb Top 1000.')

# Display movies
print(al_pacino_movies[['Series_Title', 'Released_Year', 'IMDB_Rating']])


# In[16]:


# Get the lowest-rated movies
lowest_rated_movies = df.nsmallest(10, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year']]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=lowest_rated_movies['Series_Title'], x=lowest_rated_movies['IMDB_Rating'], palette='Reds_r')
plt.title('Bottom 10 Lowest Rated Movies')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.xlim(0, 7)  # Adjust x-axis
plt.show()

# Display movies
print(lowest_rated_movies)


# In[17]:


# Get top-rated movies
top_rated_movies = df.nlargest(10, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year']]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=top_rated_movies['Series_Title'], x=top_rated_movies['IMDB_Rating'], palette='Blues_r')
plt.title('Top 10 Highest Rated Movies')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.xlim(8.5, 10)  # Adjust x-axis for better visualization
plt.show()

# Display movies
print(top_rated_movies)


# In[18]:


from itertools import combinations
from collections import Counter

# Create actor pairs from each movie
actor_pairs = []
for i, row in df.iterrows():
    actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
    pairs = combinations(actors, 2)  # Create all 2-actor combinations
    actor_pairs.extend(pairs)

# Count occurrences
pair_counts = Counter(actor_pairs)
top_pairs = pair_counts.most_common(10)

# Convert to DataFrame
top_pairs_df = pd.DataFrame(top_pairs, columns=['Pair', 'Count'])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=top_pairs_df['Pair'].astype(str), x=top_pairs_df['Count'], palette='coolwarm')
plt.title('Top 10 Most Frequent Co-Stars')
plt.xlabel('Number of Movies Together')
plt.ylabel('Actor Pair')
plt.show()


# In[19]:


# Combine all actor columns into a single column
actors = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']])

# Count occurrences of each actor
actor_counts = actors.value_counts().head(20)  # Top 20 most frequent actors

# Plot
plt.figure(figsize=(14, 6))
sns.barplot(y=actor_counts.index, x=actor_counts.values, palette='viridis')
plt.title('Top 20 Actors with Most Movies in IMDb 1000')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.show()


# In[20]:


director_counts = df['Director'].value_counts().head(20)  # Top 20 directors

plt.figure(figsize=(14, 6))
sns.barplot(y=director_counts.index, x=director_counts.values, palette='coolwarm')
plt.title('Top 20 Directors with Most Movies')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()


# In[21]:


# Filter movies with Robert De Niro and Leonardo DiCaprio
de_niro_movies = df[df[['Star1', 'Star2', 'Star3', 'Star4']].eq("Robert De Niro").any(axis=1)]
dicaprio_movies = df[df[['Star1', 'Star2', 'Star3', 'Star4']].eq("Leonardo DiCaprio").any(axis=1)]

# Compare IMDb ratings
plt.figure(figsize=(12, 6))
sns.kdeplot(de_niro_movies['IMDB_Rating'], label='Robert De Niro', fill=True, color='blue', alpha=0.5)
sns.kdeplot(dicaprio_movies['IMDB_Rating'], label='Leonardo DiCaprio', fill=True, color='red', alpha=0.5)
plt.title('IMDb Rating Distribution: Robert De Niro vs. Leonardo DiCaprio')
plt.xlabel('IMDb Rating')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[22]:


#If movies are getting better or worse by the time
# Convert Released_Year to numeric
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# Average IMDb Rating per year
yearly_ratings = df.groupby('Released_Year')['IMDB_Rating'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values, marker='o', color='purple')
plt.title('IMDb Rating Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.grid(True)
plt.show()


# In[23]:


#WHO IS CONSTANTLY MAKING GREAT MOVIES
# Get directors with at least 5 movies
top_directors = df['Director'].value_counts()[df['Director'].value_counts() >= 5].index
director_ratings = df[df['Director'].isin(top_directors)].groupby('Director')['IMDB_Rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(y=director_ratings.index, x=director_ratings.values, palette='magma')
plt.title('Top Directors by Average IMDb Rating (Min. 5 Movies)')
plt.xlabel('Average IMDb Rating')
plt.ylabel('Director')
plt.show()


# In[24]:


# Filter movies where Christopher Nolan is the director and Leonardo DiCaprio is one of the stars
leo_nolan_movies = df[(df['Director'] == 'Christopher Nolan') & 
                      ((df['Star1'] == 'Leonardo DiCaprio') | 
                       (df['Star2'] == 'Leonardo DiCaprio') | 
                       (df['Star3'] == 'Leonardo DiCaprio') | 
                       (df['Star4'] == 'Leonardo DiCaprio'))]

# Display results
print(leo_nolan_movies[['Series_Title', 'Released_Year', 'IMDB_Rating']])


# In[25]:


# Filter movies where Christopher Nolan is the director and Al Pacino is one of the stars
pacino_nolan_movies = df[(df['Director'] == 'Christopher Nolan') & 
                         ((df['Star1'] == 'Al Pacino') | 
                          (df['Star2'] == 'Al Pacino') | 
                          (df['Star3'] == 'Al Pacino') | 
                          (df['Star4'] == 'Al Pacino'))]

# Display results
print(pacino_nolan_movies[['Series_Title', 'Released_Year', 'IMDB_Rating']])


# In[26]:


# Convert 'Runtime' to numeric (removing ' min')
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)

# Find the movie with the longest runtime
longest_movie = df.nlargest(1, 'Runtime')[['Series_Title', 'Runtime', 'Released_Year', 'IMDB_Rating']]

# Display result
print(longest_movie)


# In[27]:


# Find the movie with the shortest runtime
shortest_movie = df.nsmallest(1, 'Runtime')[['Series_Title', 'Runtime', 'Released_Year', 'IMDB_Rating']]

# Display result
print(shortest_movie)


# In[28]:


# Find the movie with the lowest number of votes
lowest_voted_movie = df.nsmallest(1, 'No_of_Votes')[['Series_Title', 'No_of_Votes', 'Released_Year', 'IMDB_Rating']]

# Display result
print(lowest_voted_movie)


# In[29]:


# Find the movie with the highest number of votes
most_voted_movie = df.nlargest(1, 'No_of_Votes')[['Series_Title', 'No_of_Votes', 'Released_Year', 'IMDB_Rating']]

# Display result
print(most_voted_movie)


# In[30]:


# Find the movie with the highest IMDb rating
highest_rated_movie = df.nlargest(1, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year', 'No_of_Votes']]

# Display result
print(highest_rated_movie)


# In[31]:


# Find the movie with the lowest IMDb rating
lowest_rated_movie = df.nsmallest(1, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year', 'No_of_Votes']]

# Display result
print(lowest_rated_movie)


# In[32]:


# Create a new column for overview length
df['Overview_Length'] = df['Overview'].astype(str).apply(len)

# Find the movie with the longest overview
longest_overview_movie = df.nlargest(1, 'Overview_Length')[['Series_Title', 'Overview', 'Overview_Length', 'IMDB_Rating']]

# Display result
print(longest_overview_movie)


# In[33]:


# Ensure 'Overview' is a string and calculate its length
df['Overview_Length'] = df['Overview'].astype(str).apply(len)

# Find the movie with the shortest overview
shortest_overview_movie = df.nsmallest(1, 'Overview_Length')[['Series_Title', 'Overview', 'Overview_Length', 'IMDB_Rating']]

# Display result
print(shortest_overview_movie)


# In[35]:


# Filter movies where Leonardo DiCaprio is in any of the star columns
leo_movies = df[(df['Star1'] == 'Leonardo DiCaprio') | 
                (df['Star2'] == 'Leonardo DiCaprio') | 
                (df['Star3'] == 'Leonardo DiCaprio') | 
                (df['Star4'] == 'Leonardo DiCaprio')]

# Select relevant columns
leo_movies = leo_movies[['Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes']]

# Sort by release year
leo_movies = leo_movies.sort_values(by='Released_Year')

# Display all Leonardo DiCaprio movies
print(leo_movies)


# In[36]:


plt.figure(figsize=(12, 6))
sns.barplot(y=leo_movies['Series_Title'], x=leo_movies['IMDB_Rating'], palette='Blues_r')

plt.title('Leonardo DiCaprio Movies and Their IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.xlim(7, 10)  # Adjust the x-axis for better visualization

plt.show()


# In[37]:


# Filter movies where Richard Gere is in any of the star columns
gere_movies = df[(df['Star1'] == 'Richard Gere') | 
                 (df['Star2'] == 'Richard Gere') | 
                 (df['Star3'] == 'Richard Gere') | 
                 (df['Star4'] == 'Richard Gere')]

# Select relevant columns
gere_movies = gere_movies[['Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes']]

# Sort by release year
gere_movies = gere_movies.sort_values(by='Released_Year')

# Display all Richard Gere movies
print(gere_movies)


# In[38]:


plt.figure(figsize=(12, 6))
sns.barplot(y=gere_movies['Series_Title'], x=gere_movies['IMDB_Rating'], palette='Reds_r')

plt.title('Richard Gere Movies and Their IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.xlim(7, 10)  

plt.show()


# In[39]:


import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Filter dataset for "Hachi: A Dog's Tale"
hachi_movie = df[df['Series_Title'].str.contains('Hachi: A Dog', case=False, na=False)]

# Check if the movie exists in the dataset
if not hachi_movie.empty:
    # Extract the overview text
    hachi_overview = hachi_movie['Overview'].values[0]

    # Clean the text: remove punctuation & lowercase all words
    cleaned_text = re.sub(r'[^\w\s]', '', hachi_overview.lower())

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate(cleaned_text)

    # Display the Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.title("Word Cloud for 'Hachi: A Dog's Tale' Movie Overview", fontsize=14)
    plt.show()
else:
    print("Movie 'Hachi: A Dog's Tale' not found in the dataset.")


# In[40]:


# Display all column names
print(df.columns)


# In[41]:


df['Decade'] = (df['Released_Year'] // 10) * 10

plt.figure(figsize=(12, 6))
sns.barplot(x=df['Decade'].value_counts().index, y=df['Decade'].value_counts().values, palette='magma')

plt.title('Number of Movies Per Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Movies')

plt.show()


# In[42]:


# Filter movies directed by Christopher Nolan
nolan_movies = df[df['Director'] == 'Christopher Nolan']

# Select relevant columns
nolan_movies = nolan_movies[['Series_Title', 'Released_Year', 'IMDB_Rating']]

# Sort by release year
nolan_movies = nolan_movies.sort_values(by='Released_Year')

# Display the movies
print(nolan_movies)


# In[43]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=nolan_movies['Released_Year'], y=nolan_movies['IMDB_Rating'], marker='o', color='purple', linewidth=2)

plt.xticks(nolan_movies['Released_Year'])  # Show every release year
plt.title('Christopher Nolan IMDb Ratings Over the Years')
plt.xlabel('Release Year')
plt.ylabel('IMDb Rating')
plt.grid(True)

plt.show()


# In[44]:


# Find Nolan's first movie by release year
first_nolan_movie = df[df['Director'] == 'Christopher Nolan'].nsmallest(1, 'Released_Year')

# Display result
print(first_nolan_movie[['Series_Title', 'Released_Year']])


# In[45]:


from itertools import combinations
from collections import Counter

# Create actor pairs from each movie
actor_pairs = []
for i, row in df.iterrows():
    actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
    pairs = combinations(actors, 2)  # Create all 2-actor combinations
    actor_pairs.extend(pairs)

# Count occurrences
pair_counts = Counter(actor_pairs)
top_pairs = pair_counts.most_common(10)

# Convert to DataFrame
top_pairs_df = pd.DataFrame(top_pairs, columns=['Pair', 'Count'])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=top_pairs_df['Pair'].astype(str), x=top_pairs_df['Count'], palette='Blues_r')

plt.title('Top 10 Most Frequent Co-Star Pairs')
plt.xlabel('Number of Movies Together')
plt.ylabel('Actor Pair')

plt.show()


# In[46]:


df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')

top_genres = df_exploded['Genre'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(y=top_genres.index, x=top_genres.values, palette='coolwarm')

plt.title('Most Common Movie Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')

plt.show()


# In[47]:


# Explode genres so each genre has its own row
df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')

# Calculate the average IMDb rating for each genre
genre_avg_rating = df_exploded.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False)

# Get the best and worst rated genres
best_genre = genre_avg_rating.idxmax()
worst_genre = genre_avg_rating.idxmin()

print(f"The highest-rated genre is: {best_genre} with an average rating of {genre_avg_rating.max():.2f}")
print(f"The lowest-rated genre is: {worst_genre} with an average rating of {genre_avg_rating.min():.2f}")


# In[48]:


# Explode genres so each genre has its own row
df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')

# Get unique genres
unique_genres = df_exploded['Genre'].unique()

# Display all genres
print("All unique genres in the dataset:")
print(unique_genres)


# In[49]:


# Count number of movies per genre
genre_counts = df_exploded['Genre'].value_counts()

# Display
print("\nNumber of movies in each genre:")
print(genre_counts)


# In[50]:


plt.figure(figsize=(12, 6))
sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='magma')

plt.title('Number of Movies Per Genre')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')

plt.show()


# In[51]:


# Filter movies where Leonardo DiCaprio is in any of the star columns
leo_movies = df[(df['Star1'] == 'Leonardo DiCaprio') | 
                (df['Star2'] == 'Leonardo DiCaprio') | 
                (df['Star3'] == 'Leonardo DiCaprio') | 
                (df['Star4'] == 'Leonardo DiCaprio')]

# Explode genres so each genre gets its own row
leo_genres = leo_movies.assign(Genre=leo_movies['Genre'].str.split(', ')).explode('Genre')

# Get unique genres
leo_unique_genres = leo_genres['Genre'].unique()

# Display result
print("Genres Leonardo DiCaprio has acted in:")
print(leo_unique_genres)


# In[52]:


# Count movies per genre for Leonardo DiCaprio
leo_genre_counts = leo_genres['Genre'].value_counts()

# Display
print("\nNumber of Leonardo DiCaprio movies in each genre:")
print(leo_genre_counts)


# In[53]:


plt.figure(figsize=(12, 6))
sns.barplot(y=leo_genre_counts.index, x=leo_genre_counts.values, palette='coolwarm')

plt.title('Genres Leonardo DiCaprio Has Acted In')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')

plt.show()


# In[54]:


# Filter movies where Richard Gere is in any of the star columns
gere_movies = df[(df['Star1'] == 'Richard Gere') | 
                 (df['Star2'] == 'Richard Gere') | 
                 (df['Star3'] == 'Richard Gere') | 
                 (df['Star4'] == 'Richard Gere')]

# Explode genres so each genre gets its own row
gere_genres = gere_movies.assign(Genre=gere_movies['Genre'].str.split(', ')).explode('Genre')

# Get unique genres
gere_unique_genres = gere_genres['Genre'].unique()

# Display result
print("Genres Richard Gere has acted in:")
print(gere_unique_genres)


# In[55]:


# Count movies per genre for Richard Gere
gere_genre_counts = gere_genres['Genre'].value_counts()

# Display
print("\nNumber of Richard Gere movies in each genre:")
print(gere_genre_counts)


# In[56]:


plt.figure(figsize=(12, 6))
sns.barplot(y=gere_genre_counts.index, x=gere_genre_counts.values, palette='Blues_r')

plt.title('Genres Richard Gere Has Acted In')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')

plt.show()


# In[57]:


# Count occurrences of each actor in the dataset
actor_counts = all_actors.value_counts().head(10)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(y=actor_counts.index, x=actor_counts.values, palette='viridis')

plt.title('Top 10 Most Frequent Actors in IMDb Top 1000')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')

plt.show()


# In[58]:


df['Gross'] = df['Gross'].str.replace(',', '').astype(float)

top_grossing = df.nlargest(10, 'Gross')[['Series_Title', 'Gross']]

plt.figure(figsize=(12, 6))
sns.barplot(y=top_grossing['Series_Title'], x=top_grossing['Gross'], palette='coolwarm')

plt.title('Top 10 Highest Grossing Movies')
plt.xlabel('Gross Earnings ($)')
plt.ylabel('Movie Title')

plt.show()


# In[59]:


# Filter movies released in the 1990s
movies_90s = df[(df['Released_Year'] >= 1990) & (df['Released_Year'] <= 1999)]

# Find the movie with the most votes
most_popular_90s = movies_90s.nlargest(1, 'No_of_Votes')[['Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes']]

# Display result
print("Most popular movie from the 1990s:")
print(most_popular_90s)


# In[60]:


# Get the top 10 most voted movies from the 1990s
top_90s_movies = movies_90s.nlargest(10, 'No_of_Votes')[['Series_Title', 'No_of_Votes']]

plt.figure(figsize=(12, 6))
sns.barplot(y=top_90s_movies['Series_Title'], x=top_90s_movies['No_of_Votes'], palette='Blues_r')

plt.title('Top 10 Most Popular Movies from the 1990s')
plt.xlabel('Number of Votes')
plt.ylabel('Movie Title')

plt.show()


# In[61]:


# Filter movies where Tom Cruise is in any of the star columns
tom_cruise_movies = df[(df['Star1'] == 'Tom Cruise') | 
                        (df['Star2'] == 'Tom Cruise') | 
                        (df['Star3'] == 'Tom Cruise') | 
                        (df['Star4'] == 'Tom Cruise')]

# Filter only movies from the 1990s
tom_cruise_90s = tom_cruise_movies[(tom_cruise_movies['Released_Year'] >= 1990) & 
                                   (tom_cruise_movies['Released_Year'] <= 1999)]

# Find the highest-rated Tom Cruise movie in the 90s
best_tom_cruise_90s = tom_cruise_90s.nlargest(1, 'IMDB_Rating')[['Series_Title', 'Released_Year', 'IMDB_Rating']]

# Display result
print("Tom Cruise's highest-rated movie in the 1990s:")
print(best_tom_cruise_90s)


# In[62]:


df['Title_Length'] = df['Series_Title'].apply(len)

# Get the longest movie titles
longest_titles = df.nlargest(10, 'Title_Length')[['Series_Title', 'Title_Length']]

# Display
print("Movies with the longest titles:")
print(longest_titles)


# In[63]:


# Create a new column for title length
df['Title_Length'] = df['Series_Title'].apply(len)

# Find the movie with the shortest title
shortest_title_movie = df.nsmallest(1, 'Title_Length')[['Series_Title', 'Released_Year', 'IMDB_Rating', 'Title_Length']]

# Display result
print("Movie with the shortest title:")
print(shortest_title_movie)


# In[64]:


#MACHINE LEARNING
get_ipython().system('pip install mlxtend')


# In[66]:


# APRIORI ALGORITHM
# Frequent Pattern Mining
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Convert 'Genre' column into a list of lists (transactions)
df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')
movie_genres = df_exploded.groupby('Series_Title')['Genre'].apply(list).tolist()

# Convert data into a transaction format
te = TransactionEncoder()
te_ary = te.fit(movie_genres).transform(movie_genres)
df_genres = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori Algorithm to find frequent itemsets
frequent_itemsets = apriori(df_genres, min_support=0.05, use_colnames=True)

# Extract association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display results
print("Frequent itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get top 10 frequent genre pairs
top_frequent = frequent_itemsets.nlargest(10, 'support')

plt.figure(figsize=(12, 6))
sns.barplot(y=top_frequent['itemsets'].apply(lambda x: ', '.join(x)), x=top_frequent['support'], palette='coolwarm')

plt.title('Top 10 Frequent Genre Combinations')
plt.xlabel('Support')
plt.ylabel('Genre Combination')

plt.show()


# In[68]:


df['Decade'] = (df['Released_Year'] // 10) * 10

# Combine decade and genre
df['Decade_Genre'] = df.apply(lambda x: str(x['Decade']) + ', ' + x['Genre'], axis=1)
df['Decade_Genre'] = df['Decade_Genre'].str.split(', ')

# Convert into transaction format
te = TransactionEncoder()
te_ary = te.fit(df['Decade_Genre']).transform(df['Decade_Genre'])
df_decade_genre = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori
frequent_decade_genre = apriori(df_decade_genre, min_support=0.02, use_colnames=True)

# Extract association rules
decade_genre_rules = association_rules(frequent_decade_genre, metric="lift", min_threshold=1)

# Display results
print("Frequent Decade-Genre Combinations:\n", frequent_decade_genre)
print("\nDecade-Genre Association Rules:\n", decade_genre_rules)


# In[69]:


# Convert 'Genre' column into a list of lists
df_exploded = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')
movie_genres = df_exploded.groupby('Series_Title')['Genre'].apply(list).tolist()

# Convert data into a transaction format
te = TransactionEncoder()
te_ary = te.fit(movie_genres).transform(movie_genres)
df_genres = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori for 3-itemsets or more
frequent_genre_sets = apriori(df_genres, min_support=0.02, use_colnames=True)

# Display only sets with 3 or more genres
multi_genre_combinations = frequent_genre_sets[frequent_genre_sets['itemsets'].apply(lambda x: len(x) >= 3)]
print("Frequent Multi-Genre Combinations:\n", multi_genre_combinations)


# In[70]:


pip install pymining


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymining import seqmining

# Load the dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

#  Convert 'Released_Year' to integer format
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce').dropna().astype(int)

# Sort movies by release year
df_sorted = df.sort_values(by='Released_Year')

# Prepare sequences (group genres by year)
yearly_sequences = df_sorted.groupby('Released_Year')['Genre'].apply(lambda x: list(set(', '.join(x).split(', ')))).tolist()

# Apply SPADE Algorithm to find frequent genre sequences
patterns = seqmining.freq_seq_enum(yearly_sequences, 3)  # Min support of 3 occurrences

# Display only the top 10 frequent patterns to avoid Jupyter overload
print("Top 10 Frequent Sequential Patterns in Movie Genres Across Years:\n")
top_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)[:10]

for pattern, freq in top_patterns:
    print(f"Pattern: {pattern}, Frequency: {freq}")

# Convert results into a DataFrame for visualization
df_patterns = pd.DataFrame(top_patterns, columns=['Pattern', 'Frequency'])
df_patterns['Pattern'] = df_patterns['Pattern'].apply(lambda x: ' → '.join(x))  # Format patterns

# Visualize results using a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(y=df_patterns['Pattern'], x=df_patterns['Frequency'], palette='coolwarm')

plt.title('Top 10 Frequent Sequential Genre Patterns')
plt.xlabel('Frequency')
plt.ylabel('Genre Pattern')

plt.show()


# In[72]:


get_ipython().system('pip install scikit-learn')


# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Convert 'Runtime' to numerical format
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)

# Select relevant features (IMDb Rating & Runtime)
X = df[['IMDB_Rating', 'Runtime']].dropna()  

# Standardize the data for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  
clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df.loc[X.index, 'Cluster'] = clusters

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.loc[X.index, 'Runtime'], y=df.loc[X.index, 'IMDB_Rating'], hue=df.loc[X.index, 'Cluster'], palette='viridis', alpha=0.7)

plt.title('DBSCAN Clustering of Movies Based on IMDb Rating & Runtime')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('IMDb Rating')
plt.legend(title="Cluster")
plt.show()

# Display the number of movies in each cluster
print(df['Cluster'].value_counts())


# In[74]:


get_ipython().system('pip install networkx matplotlib')


# In[76]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Create an empty graph
G = nx.Graph()

# Add actor collaborations as edges
for _, row in df.iterrows():
    actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
    actors = [actor for actor in actors if pd.notna(actor)]  # Remove missing values
    
    for i in range(len(actors)):
        for j in range(i + 1, len(actors)):  # Create edges between co-actors
            G.add_edge(actors[i], actors[j], movie=row['Series_Title'])

# Visualize the Most Frequent Actor Collaborations
plt.figure(figsize=(12, 8))
top_actors = [node for node, degree in sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]]  # Top 10 actors
subgraph = G.subgraph(top_actors)

nx.draw(subgraph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
plt.title("Top Actor Collaborations in IMDb Movies")
plt.show()


# In[3]:


#MILESTONE 3 - APPLYING MACHINE LEARNING 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Convert 'Released_Year' to numeric
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce').dropna().astype(int)

# Compute the average IMDb rating per year
ratings_over_time = df.groupby('Released_Year')['IMDB_Rating'].mean()

# Plot IMDb rating trends over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=ratings_over_time.index, y=ratings_over_time.values, marker='o', color='b')

plt.title('IMDb Rating Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.grid(True)
plt.show()


# In[2]:


# Decompose the time series data
decomposition = seasonal_decompose(ratings_over_time, model='additive', period=10)

# Plot the trend, seasonality, and residuals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(decomposition.trend, label='Trend', color='blue')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(decomposition.resid, label='Residual (Anomalies)', color='red')
plt.legend()

plt.suptitle('Time Series Decomposition of IMDb Ratings')
plt.show()


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Convert 'Runtime' to numerical format
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)

# Select features and target variable (Genre)
df['Genre'] = df['Genre'].str.split(', ').str[0]  # Take only the first genre for simplicity

# Encode categorical target variable (Genre)
label_encoder = LabelEncoder()
df['Genre_Label'] = label_encoder.fit_transform(df['Genre'])

# Select features for prediction
X = df[['IMDB_Rating', 'Runtime', 'No_of_Votes']].dropna()
y = df.loc[X.index, 'Genre_Label']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.naive_bayes import GaussianNB

# Train the Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred_nb = nb_model.predict(X_test)

# Evaluate accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f'Naïve Bayes Accuracy: {nb_accuracy:.2f}')


# In[6]:


from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')


# In[7]:


print(f"Naïve Bayes Accuracy: {nb_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

if rf_accuracy > nb_accuracy:
    print("✅ Random Forest performed better!")
else:
    print("✅ Naïve Bayes performed better!")


# In[9]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Train AdaBoost with a weak Decision Tree classifier
adaboost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)

# Predict on test data
y_pred_ada = adaboost_model.predict(X_test)

# Evaluate accuracy
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print(f'AdaBoost Accuracy: {ada_accuracy:.2f}')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Convert 'Runtime' to numerical format
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)

# Select features for clustering
X = df[['IMDB_Rating', 'Runtime']].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN with adjusted parameters
dbscan = DBSCAN(eps=0.3, min_samples=3)  # Adjusted eps and min_samples
clusters = dbscan.fit_predict(X_scaled)

# Assign cluster labels to DataFrame
df.loc[X.index, 'Cluster'] = clusters

# Check if clusters were found
print("Unique Cluster Labels:", np.unique(clusters))
print(df['Cluster'].value_counts())  # See how many movies per cluster

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.loc[X.index, 'Runtime'], y=df.loc[X.index, 'IMDB_Rating'], hue=df.loc[X.index, 'Cluster'], palette='viridis', alpha=0.7)

plt.title('DBSCAN Clustering of Movies Based on IMDb Rating & Runtime')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('IMDb Rating')
plt.legend(title="Cluster")
plt.show()


# In[11]:


import pandas as pd
import numpy as np

# Load dataset
file_path = "C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv"
df = pd.read_csv(file_path)

# Compute Z-scores for IMDb ratings
df['IMDB_Z_Score'] = (df['IMDB_Rating'] - df['IMDB_Rating'].mean()) / df['IMDB_Rating'].std()

# Find movies with extreme ratings (Z-score > 2 or < -2)
outliers = df[(df['IMDB_Z_Score'] > 2) | (df['IMDB_Z_Score'] < -2)]

print("Outlier Movies Based on IMDb Rating:")
print(outliers[['Series_Title', 'IMDB_Rating']])


# In[ ]:




