import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random
from onnx_operations import OnnxConverter

def random_deletion(words):
    """Randomly delete a word in the paragraph."""
    word_tokens = words.split()
    if len(word_tokens) == 1:  # return if single word
        return words
    random_word = random.choice(word_tokens)
    new_words = words.replace(random_word, "")
    return new_words

def random_swap(words):
    """Randomly swap two words in the paragraph."""
    word_tokens = words.split()
    if len(word_tokens) < 2:  # return if less than 2 words
        return words
    random_index_1, random_index_2 = random.sample(range(0, len(word_tokens)), 2)
    word_tokens[random_index_1], word_tokens[random_index_2] = word_tokens[random_index_2], word_tokens[random_index_1]
    return ' '.join(word_tokens)

def augment_paragraph(paragraph):
    """Apply a random augmentation technique to the paragraph."""
    techniques = [random_deletion, random_swap]
    technique = random.choice(techniques)
    return technique(paragraph)

def data_augmentation(df):
    """Augment the data by applying random textual data augmentation techniques."""
    augmented_data = df.copy()
    augmented_data["Four-word Paragraph"] = df["Four-word Paragraph"].apply(augment_paragraph)
    return pd.concat([df, augmented_data])

# Modify the load_data function
def load_data(filepath):
    """Load the data from a CSV file, augment it, and split it into training and testing datasets."""
    df = pd.read_csv(filepath)
    df = data_augmentation(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df
    
def preprocess_data(train_df, test_df=None):
    """Preprocess the data using TF-IDF vectorization."""
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df["Four-word Paragraph"])
    y_train = train_df["Is Positive (1/0)"]
    
    if test_df is not None:
        X_test = vectorizer.transform(test_df["Four-word Paragraph"])
        y_test = test_df["Is Positive (1/0)"]
    else:
        X_test, y_test = None, None
    
    return X_train, y_train, X_test, y_test, vectorizer

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance and print the results."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_rep)


if __name__ == '__main__':
    # Load and preprocess data
    train_df, test_df = load_data("demonstration_file.csv")
    X_train, y_train, X_test, y_test, vectorizer = preprocess_data(train_df, test_df)
    
    # Initialize model
    model = LogisticRegression(random_state=42)

    # Train and evaluate the model 6 times
    for i in range(16):
        print(f"Training iteration {i+1}...")
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        
        # Augment the training data after each iteration except the last one
        if i < 5:
            train_df = data_augmentation(train_df)
            X_train, y_train = preprocess_data(train_df, None)[:2]
    
    # Convert the model to ONNX format and save it
    converter = OnnxConverter()
    onnx_model = converter.convert_to_onnx(model, X_train)
    converter.save_onnx_model(onnx_model)
    
    # Save the TF-IDF vectorizer's vocabulary
    converter.save_vectorizer_vocabulary(vectorizer)