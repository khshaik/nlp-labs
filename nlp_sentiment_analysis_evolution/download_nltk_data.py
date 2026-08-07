"""
Script to download NLTK Movie Reviews Corpus and other required NLTK data
"""
import nltk

print("Downloading NLTK Movie Reviews Corpus...")
nltk.download('movie_reviews')

print("\nDownloading additional NLTK data packages...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

print("\nAll NLTK data packages downloaded successfully!")
print("\nYou can now access the movie reviews corpus using:")
print("from nltk.corpus import movie_reviews")
