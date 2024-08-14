import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df_market = pd.read_csv("C:\\Users\\Kimble\\Downloads\\social_marketing.csv")

# Drop the user identifier column
df_market_data = df_market.drop(columns=['Unnamed: 0'])


# Applying K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=12, random_state=42)
df_market['Cluster'] = kmeans.fit_predict(df_market_data)

# Identify the size of each cluster
cluster_sizes = df_market['Cluster'].value_counts()

# Find the largest clusters
largest_clusters = cluster_sizes.index[:11]  # Modify this to show more or fewer clusters

# Visualize the distributions of the features in the largest clusters
for cluster in largest_clusters:
    plt.figure(figsize=(12, 8))
    cluster_data = df_market[df_market['Cluster'] == cluster]
    cluster_mean = cluster_data.drop(columns=['Cluster']).mean()
    cluster_mean.plot(kind='bar')
    plt.title(f'Feature Means for Cluster {cluster} (Size: {cluster_sizes[cluster]})')
    plt.ylabel('Mean Value')
    plt.show()
    
# I created 30 clusters and visually analyzed then to watch for after how many clusters the groups/variables/vairbale interactions start repreating themselves. I found that 11 seemed to be an optimal number. These 12 are outputted below.
# Here we can see some noticable specific trends among the user base: 1. general chatter with a mix of all variables 2. general chatter with shopping and photo sharing 3. Health and Fitness 4. Sports, Relgion and Food (seems cultural) 5. Cooking, fashion/beauty and photosharing 
# 6. More chatter with shopping and photosharing 7. News and Politics 8. Online gaming and University 9. TV and Art 10. Health and Fitness 11. Politics and Travel.
# From these 11 clusters we can see clear segments of the user base defined by the content of their tweets. This can then be used for targeted advertisements
