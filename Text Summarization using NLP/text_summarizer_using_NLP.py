import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Download stopwords from NLTK
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Tokenize the sentences
    sentences = sent_tokenize(text)
    
    # Load English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and unnecessary characters
    def clean_sentence(sentence):
        # Convert sentence to lowercase and remove stopwords
        tokens = [word for word in nltk.word_tokenize(sentence.lower()) if word.isalnum() and word not in stop_words]
        return ' '.join(tokens)

    clean_sentences = [clean_sentence(sentence) for sentence in sentences]
    return sentences, clean_sentences

def rank_sentences(sentences, clean_sentences):
    # Vectorize the cleaned sentences
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(clean_sentences)

    # Compute cosine similarity between all sentences
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Rank sentences using similarity scores
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in scores.argsort()[::-1]]
    return ranked_sentences

def summarize_text(text, summary_length=3):
    # Preprocess the text
    sentences, clean_sentences = preprocess_text(text)
    
    # Rank sentences
    ranked_sentences = rank_sentences(sentences, clean_sentences)
    
    # Select top-ranked sentences for the summary
    summary = ' '.join(ranked_sentences[:summary_length])
    return summary

if __name__ == "__main__":
    sample_text = """
Chennai, the vibrant capital city of Tamil Nadu, is a captivating blend of tradition and modernity. Located on the southeastern coast of India along the Bay of Bengal, Chennai, formerly known as Madras, is one of the country's largest metropolises and a vital cultural, economic, and educational hub in South India. With a rich history dating back over 2,000 years, Chennai is a city that seamlessly weaves its past into its present, offering a unique experience to both residents and visitors.
Known for its deep-rooted traditions, Chennai is often referred to as the cultural capital of South India. It is renowned for its classical dance forms, such as Bharatanatyam, and Carnatic music, which have flourished here for centuries. The city hosts the annual Margazhi Music and Dance Festival, a month-long celebration that attracts performers and art enthusiasts from around the world. The music and dance season, as it is fondly known, transforms Chennai into a vibrant cultural epicenter, showcasing the region's rich artistic heritage.
Chennai is also home to some of India’s most iconic landmarks. The famous Marina Beach, one of the longest urban beaches in the world, stretches along the city’s coastline and is a popular spot for locals and tourists alike. Visitors can enjoy a leisurely stroll along its sandy shores or witness a mesmerizing sunrise over the Bay of Bengal. The city also boasts several historic structures, such as Fort St. George, the first British fort in India, built in 1644, which now serves as the headquarters of the Tamil Nadu government. The stunning Kapaleeshwarar Temple, dedicated to Lord Shiva, and the San Thome Basilica, a significant Christian pilgrimage site built over the tomb of St. Thomas the Apostle, reflect the city's religious diversity and architectural splendor.
Economically, Chennai is a bustling center for industries like automobile manufacturing, IT services, and healthcare. Often referred to as the "Detroit of India," it is a major hub for the automotive industry, hosting factories for major car manufacturers like Hyundai, Ford, and BMW. The city is also a prominent IT and software services center, with several multinational companies establishing their offices here. This economic dynamism has led to a rapid growth in infrastructure, with new residential complexes, shopping malls, and office spaces continually reshaping the city's skyline.
Despite its rapid modernization, Chennai has managed to retain its traditional charm. The city is known for its warm hospitality, delectable South Indian cuisine, and vibrant festivals, such as Pongal, which is celebrated with great enthusiasm. From the aromatic flavors of filter coffee to the intricate sarees of Kanchipuram, Chennai offers a glimpse into the soul of South India. With its perfect mix of culture, history, and modernity, Chennai stands as a testament to India's diversity, a city where every street, monument, and dish has a story to tell.
    """


    summary = summarize_text(sample_text, summary_length=3)
    print("Summary:")
    print(summary)
