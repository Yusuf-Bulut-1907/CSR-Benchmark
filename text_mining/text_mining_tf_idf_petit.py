import re
import spacy
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from spacy.lang.en.stop_words import STOP_WORDS

from stats_and_cleaning import get_cleaned_corpus
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

#nltk.download("stopwords")
#nltk.download("punkt")
custom_stopwords = {
    "company", "business", "report", "year", "page", "website", "provide", 
    "include", "information", "service", "client", "group", "pdf", "site", 
    "download", "consent", "cookie", "use", "data", "also", "may", "one", 
    "new", "us", "cookies", "cooky", "privacy", "device", "browser",
    "collect", "processing", "purpose", "request", "contact", "law", 
    "advertising", "third", "party", "personal", "identify", "storage", 
    "que", "und", "para", "die", "siemens", "les", "des", "von", "com", "der",
    "gdpr", "privacy", "visit","read", "learn", "access", "store", "session","legal","datum",
    "opt","applicable","necessary","interest","notice","email","performance","functionality",
    "adresse", "linkedin","facebook","instagram","twitter","setting", "settings",
    

}
#custom_stopwords = set(nltk.corpus.stopwords.words("english")).union(custom_stopwords)
STOP_WORDS.update(custom_stopwords)
BUSINESS_STOPWORDS = {
    "company","group","business","corporate","organization",
    "report","annual","financial","strategy","approach",
    "management","value","activity","operation",
    "stakeholder","objective","commitment","initiative",
    "policy","framework","implementation",
    "page","website","information","content","document",
    "download","read","visit","contact"
}

STOP_WORDS.update(BUSINESS_STOPWORDS)
CSR_LEXICON = {
    # Environment
    "climate","carbon","emission","co2","biodiversity",
    "energy","renewable","pollution","waste","recycling",
    "water","net_zero","climate_change",

    # Social
    "diversity","inclusion","equality","gender",
    "human_rights","health","safety","training",
    "employee","community",

    # Governance
    "ethics","compliance","transparency","governance",
    "anti_corruption","board","risk_management",

    # Reporting
    "csr","esg","sustainability","sustainable",
    "gri","sdg","un_global_compact"
}
N_GRAM_SUPRESS = {"ip","address", "legitimate","interest", "google","analytic", "similar_technology", "marketing" "communication",
     "mail","personal", "datum","privacy"}
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
# =====================
# LOAD CORPUS
# =====================
documents, metadata = get_cleaned_corpus()

df = pd.DataFrame({
    "text": documents,
    "company": [m.get("company") for m in metadata]
})

print(f"📄 Files loaded : {len(df)}")
print(f"🏢 Unique companies : {df['company'].nunique()}")

# =====================
# CLEANING + LEMMATIZATION (SPACY)
# =====================
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
def extract_noun_trigrams(doc):
    """
    Extract NOUN-NOUN-NOUN or PROPN-NOUN-NOUN trigrams from a spaCy Doc.
    Returns a list of lemmatized trigrams joined with '_'.
    """
    trigrams = []

    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i+1], doc[i+2]

        if (
            t1.pos_ in {"NOUN", "PROPN"} and
            t2.pos_ in {"NOUN", "PROPN"} and
            t3.pos_ in {"NOUN", "PROPN"}
        ):
            if not (t1.is_stop or t2.is_stop or t3.is_stop):
                trigram = f"{t1.lemma_}_{t2.lemma_}_{t3.lemma_}"
                trigrams.append(trigram)

    return trigrams
def extract_filtered_bigrams(doc):
    """
    Extract linguistically meaningful bigrams:
    - ADJ + NOUN
    - NOUN + NOUN
    - PROPN + NOUN
    """
    bigrams = []

    for i in range(len(doc) - 1):
        t1, t2 = doc[i], doc[i+1]

        if (
            not t1.is_stop and not t2.is_stop and
            t1.is_alpha and t2.is_alpha and
            (
                (t1.pos_ == "ADJ" and t2.pos_ == "NOUN") or
                (t1.pos_ == "NOUN" and t2.pos_ == "NOUN") or
                (t1.pos_ == "PROPN" and t2.pos_ == "NOUN")
            )
        ):
            bigram = f"{t1.lemma_}_{t2.lemma_}"
            bigrams.append(bigram)

    return bigrams
def clean_and_lemmatize(text):
    # nettoyage physique
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token.lemma_) > 2
        and any(kw in token.lemma_ for kw in CSR_LEXICON)
        and not any(kw in token.lemma_ for kw in N_GRAM_SUPRESS)
    ]
    nouns_bigrams = extract_filtered_bigrams(doc)
    nouns_trigrams = extract_noun_trigrams(doc)
    return " ".join(tokens +nouns_bigrams + nouns_trigrams)

print("🚿 Cleaning + lemmatization processing ...")
df["text_processed"] = df["text"].apply(clean_and_lemmatize)

# =====================
# GROUP BY COMPANY
# =====================
df_company = (
    df.groupby("company")["text_processed"]
      .apply(lambda x: " ".join(x))
      .reset_index()
)

n_companies = len(df_company)
print(f"✅ Agregated Corpus : {n_companies} entreprises ")

# =====================
# TDM : UNIGRAMS + BIGRAMS
# =====================

# ---- Unigrams ----
cv_uni = CountVectorizer(
    ngram_range=(1, 2),
    min_df=0.05,   # ≥ 3% of companies 
    max_df=0.85,
    #max_features=3000
)

X_uni = cv_uni.fit_transform(df_company["text_processed"])
uni_features = cv_uni.get_feature_names_out()

# ---- Bigrams ----
'''cv_bi = CountVectorizer(
    ngram_range=(2, 2),
    min_df=0.04,   # ≥ 4% of companies → strong relevance
    max_df=0.85,
    #max_features=2000
)

X_bi = cv_bi.fit_transform(df_company["text_processed"])
bi_features = cv_bi.get_feature_names_out()'''

'''cv_tri = CountVectorizer(
    ngram_range=(3, 3),
    min_df=0.05,   # ≥ 5% of companies → very strong relevance
    max_df=0.85,
    #max_features=1000
)
X_tri = cv_tri.fit_transform(df_company["text_processed"])
tri_features = cv_tri.get_feature_names_out()'''

# ---- Fusion ----
X_tdm = hstack([X_uni])#, X_bi, X_tri])
features = np.concatenate([uni_features])#, bi_features, tri_features])

print("📐 TDM shape :", X_tdm.shape)

# =====================
# EXPORT TDM CSV
# =====================
df_tdm = pd.DataFrame(
    X_tdm.toarray(),
    index=df_company["company"],
    columns=features
)

df_tdm.to_csv("data/TDM_unigram_bigram_trigram.csv")
print("💾 TDM exportée (CSV)")

# =====================
# TF-IDF COMPUTATION
# =====================
tfidf = TfidfVectorizer(
    vocabulary=features
)

X_tfidf = tfidf.fit_transform(df_company["text_processed"])
# =====================
# LSA / SVD REDUCTION
# =====================
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

svd = TruncatedSVD(n_components=100, random_state=42)
normalizer = Normalizer(copy=False)

lsa = make_pipeline(svd, normalizer)
X_lsa = lsa.fit_transform(X_tfidf)
df_lsa = pd.DataFrame(
    X_lsa,
    index=df_company["company"],
    columns=[f"LSA_{i}" for i in range(X_lsa.shape[1])]
)

df_lsa.to_csv("data/TFIDF_LSA_100.csv")
print("💾 LSA matrix exported (CSV)")

df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    index=df_company["company"],
    columns=features
)

uni_tfidf = TfidfVectorizer(
    vocabulary=uni_features)
X_uni_tfidf = uni_tfidf.fit_transform(df_company["text_processed"])
df_uni_tfidf = pd.DataFrame(
    X_uni_tfidf.toarray(),
    index=df_company["company"],
    columns=uni_features
)
df_uni_tfidf.to_csv("data/TFIDF_unigram.csv")
print("💾 TF-IDF Unigrams exported (CSV)")

df_tfidf.to_csv("data/TFIDF_unigram_bigram_trigram.csv")
print("💾 TF-IDF exported (CSV)")

print("🎉 Pipeline done with success!")