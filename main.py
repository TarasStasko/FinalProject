import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
from langcodes import Language
from transformers import pipeline

# Побудова візуалізацій може зайняти декілька хвилин.
file = 'data.csv'
data = pd.read_csv(file, nrows=2000000, low_memory=False)

data['sensitive-topic'] = data['sensitive-topic'].fillna('none')
filtered_data = data[data['sensitive-topic'] != 'none']
topic_counts = filtered_data['sensitive-topic'].value_counts()
top_topics = topic_counts.head(10)

plt.figure(figsize=(12, 6))
top_topics.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Топ-10 чутливих тем у російських Telegram-каналах', fontsize=14)
plt.xlabel('Чутлива тема', fontsize=12)
plt.ylabel('Кількість постів', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


data['toxicity'] = data['toxicity'].map({'neutral': 0, 'toxic': 1})
toxic_counts_by_channel = data[data['toxicity'] == 1].groupby("channel")["toxicity"].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
toxic_counts_by_channel.plot(kind="bar", color="salmon", width=0.8)

plt.title("Топ-10 каналів за кількістю токсичних повідомлень")
plt.xlabel("Назва каналу")
plt.ylabel("Кількість токсичних повідомлень")
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

content_distribution = data.groupby(['channel', 'type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
content_distribution.plot(kind='bar', stacked=True, ax=ax)
ax.set_title("Розподіл типів контенту по каналах")
ax.set_xlabel("Канал")
ax.set_ylabel("Кількість постів")
plt.tight_layout()
plt.show()

views_by_channel = data.groupby('channel')['views'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=views_by_channel.values, y=views_by_channel.index, color='skyblue')
plt.title("Топ-10 каналів за загальною кількістю переглядів")
plt.xlabel("Загальна кількість переглядів")
plt.ylabel("Канал")
plt.show()

data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.tz_localize(None)
politics_posts = data[data['sensitive-topic'] == 'politics']
politics_by_month = politics_posts['date'].dt.to_period('M').value_counts().sort_index()
plt.figure(figsize=(12, 6))
politics_by_month.plot(kind='line', marker='o', color='darkblue')
plt.title("Тенденція постів на політичну тему по місяцях")
plt.xlabel("Місяць")
plt.ylabel("Кількість постів")
plt.grid(True)
plt.tight_layout()
plt.show()

media_posts = data[data['type'] != 'text'].copy()
media_posts['duration'] = pd.to_numeric(media_posts['duration'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=media_posts, x='duration', y='views', color='green', alpha=0.5)
plt.title("Залежність між тривалістю медіа та кількістю переглядів")
plt.xlabel("Тривалість медіа (секунди)")
plt.ylabel("Кількість переглядів")
plt.tight_layout()
plt.show()

data['message'] = data['message'].fillna('')
sentiment_analyzer = pipeline('sentiment-analysis', model='blanchefort/rubert-base-cased-sentiment')
sample_data = data.head(500).copy()

sample_data.loc[:, 'sentiment'] = sample_data['message'].apply(
    lambda text: sentiment_analyzer(text[:512])[0]['label'] if len(text) > 0 else 'Neutral'
)

sample_data['sentiment'] = sample_data['sentiment'].str.upper()
sentiment_counts = sample_data['sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, palette="viridis", legend=False)

plt.title('Розподіл тональності повідомлень')
plt.ylabel('Кількість повідомлень')
plt.xlabel('Тональність')
plt.tight_layout()
plt.show()

data['message'] = data['message'].fillna('')
data['date'] = pd.to_datetime(data['date'])
data = data[(data['date'].dt.year >= 2020) & (data['sensitive-topic'] != "none")]
data['month'] = data['date'].dt.to_period('M')

topic_counts = data['sensitive-topic'].value_counts()
top_20_topics = topic_counts.head(20).index
filtered_data = data[data['sensitive-topic'].isin(top_20_topics)]
monthly_topics = filtered_data.groupby(['month', 'sensitive-topic']).size().unstack().fillna(0)

fig, ax = plt.subplots(figsize=(14, 8))
monthly_topics.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)
ax.set_title('Розподіл 20 найпопулярніших тем по місяцях')
ax.set_xlabel('Місяць')
ax.set_ylabel('Кількість повідомлень')
ax.legend(title='Тема', loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data['message'] = data['message'].fillna('')
data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
top_topics = data['sensitive-topic'].value_counts().head(10).index
top_topics_data = data[data['sensitive-topic'].isin(top_topics)]

plt.figure(figsize=(12, 6))
sns.countplot(x='hour', data=data, color='skyblue')
plt.title('Часова діаграма активності каналів (розподіл постів по годинах доби)')
plt.xlabel('Година доби')
plt.ylabel('Кількість постів')
plt.xticks(rotation=45)
plt.show()

subset_data = data.head(20000).copy()

subset_data['language'] = subset_data['message'].astype(str).apply(lambda x: detect(x) if pd.notnull(x) else 'unknown')
subset_data['language_full'] = subset_data['language'].apply(lambda x: Language.get(x).display_name('en') if x != 'unknown' else 'Unknown')
language_distribution = subset_data['language_full'].value_counts().head(10)

plt.figure(figsize=(10, 7))
ax = language_distribution.plot(kind='bar', color=plt.cm.Paired(range(len(language_distribution))))
ax.set_yscale('log')

for i, v in enumerate(language_distribution.values):
    perc = (v / language_distribution.sum()) * 100
    ax.text(i, v + 0.5, f"{perc:.1f}%", ha='center', fontsize=12)

plt.title('Розподіл мов у даних', fontsize=16)
plt.xlabel('Мова', fontsize=14)
plt.ylabel('Кількість постів', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()
