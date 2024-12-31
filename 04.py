import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load datasets
train_file_path = r"4th_task\twitter_training.csv"  # Update with actual path
val_file_path = r"4th_task\twitter_validation.csv"  # Update with actual path

train_data = pd.read_csv(train_file_path)
val_data = pd.read_csv(val_file_path)

# Rename columns for consistency
train_data.columns = ['ID', 'Category', 'Sentiment', 'Text']
val_data.columns = ['ID', 'Category', 'Sentiment', 'Text']

# Merge datasets
combined_data = pd.concat([train_data, val_data], ignore_index=True)

# Display basic information
print("Combined Dataset Overview:")
print(combined_data.info())
print(combined_data.head())

# Analyze sentiment distribution
sentiment_counts = combined_data['Sentiment'].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Generate a Word Cloud for the entire text data
text_data = " ".join(combined_data['Text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Combined Text Data", fontsize=16)
plt.show()

# Generate sentiment-specific Word Clouds
for sentiment in combined_data['Sentiment'].unique():
    sentiment_text = " ".join(combined_data[combined_data['Sentiment'] == sentiment]['Text'].dropna())
    sentiment_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(sentiment_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Sentiment: {sentiment}", fontsize=16)
    plt.show()
