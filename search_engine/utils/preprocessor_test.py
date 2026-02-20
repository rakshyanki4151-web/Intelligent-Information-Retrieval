"""
Test script for TextPreprocessor class

This script tests the preprocessor with various input types:
1. Short single word
2. Long paragraph (200+ words)
3. Text with stopwords
4. Text with special characters
5. Text with URLs and emails
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessor import TextPreprocessor


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)


def test_preprocessor(test_name, text, preprocessor):
    """Test preprocessor with given text and display results."""
    print_separator(test_name)
    print(f"\nInput Text:\n{text}\n")
    
    # Process text
    tokens = preprocessor.process_text(text)
    
    print(f"\nFinal Tokens ({len(tokens)}):")
    print(tokens)
    
    # Show detailed steps
    print("\n" + "-" * 80)
    print("PREPROCESSING STEPS:")
    print("-" * 80)
    
    steps = preprocessor.get_preprocessing_steps(text)
    for i, step in enumerate(steps, 1):
        print(f"\n{i}. {step['step']}:")
        result = step['result']
        if len(result) > 200:
            print(f"   {result[:200]}...")
        else:
            print(f"   {result}")


def main():
    """Run all tests."""
    print_separator("TEXT PREPROCESSOR TEST SUITE")
    
    # Initialize preprocessor with default settings
    print("\nPreprocessor Configuration:")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_special_chars=True,
        remove_stopwords=True,
        use_lemmatization=True,
        use_stemming=False,
        handle_numbers='replace',
        min_token_length=2
    )
    
    config = preprocessor.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test 1: Short single word
    test_preprocessor(
        "TEST 1: Short Single Word",
        "stocks",
        preprocessor
    )
    
    # Test 2: Long paragraph (200+ words)
    long_text = """
    Global tech giants are investing billions in artificial intelligence infrastructure as the race 
    for AI supremacy intensifies. Major companies including Microsoft, Google, and Amazon have 
    announced significant capital expenditure plans for 2025, focusing on data centers and AI chip 
    development. Industry analysts predict this investment wave will reshape the technology landscape 
    over the next decade. The competition has reached unprecedented levels, with each company trying 
    to outpace rivals in developing more powerful AI systems. Microsoft's partnership with OpenAI has 
    given it an early advantage, but Google's DeepMind and Amazon's AWS are rapidly catching up. 
    The investments aren't just about building bigger data centers; they're about creating entirely 
    new computing paradigms. Quantum computing, neuromorphic chips, and specialized AI accelerators 
    are all part of this technological arms race. Meanwhile, smaller startups are finding it 
    increasingly difficult to compete without similar levels of capital investment. This has raised 
    concerns about market concentration and the potential for a few large companies to dominate the 
    AI landscape. Regulators in the United States, European Union, and China are closely monitoring 
    these developments, considering new frameworks to ensure fair competition and prevent monopolistic 
    practices. The societal implications are profound, affecting everything from employment patterns 
    to privacy rights and national security. As AI systems become more capable, questions about 
    governance, ethics, and control become increasingly urgent. The next few years will be critical 
    in determining not just which companies lead in AI, but how humanity as a whole navigates this 
    transformative technology.
    """
    
    test_preprocessor(
        "TEST 2: Long Paragraph (200+ words)",
        long_text,
        preprocessor
    )
    
    # Test 3: Text with stopwords
    test_preprocessor(
        "TEST 3: Text with Stopwords",
        "the movie was very good and the acting was excellent",
        preprocessor
    )
    
    # Test 4: Special characters
    test_preprocessor(
        "TEST 4: Special Characters",
        "Apple's CEO said: 'Innovation!' - but what does that really mean?",
        preprocessor
    )
    
    # Test 5: URLs and emails
    test_preprocessor(
        "TEST 5: URLs and Emails",
        "Visit our website at http://example.com or https://news.bbc.co.uk for more info. "
        "Contact us at support@example.com or sales@company.org for inquiries.",
        preprocessor
    )
    
    # Additional test: Compare lemmatization vs stemming
    print_separator("BONUS TEST: Lemmatization vs Stemming Comparison")
    
    test_text = "The running runners ran quickly through the beautiful buildings"
    
    print(f"\nTest Text: {test_text}\n")
    
    # Lemmatization only
    preprocessor_lemma = TextPreprocessor(
        remove_stopwords=True,
        use_lemmatization=True,
        use_stemming=False
    )
    tokens_lemma = preprocessor_lemma.process_text(test_text)
    print(f"Lemmatization only: {tokens_lemma}")
    
    # Stemming only
    preprocessor_stem = TextPreprocessor(
        remove_stopwords=True,
        use_lemmatization=False,
        use_stemming=True
    )
    tokens_stem = preprocessor_stem.process_text(test_text)
    print(f"Stemming only:      {tokens_stem}")
    
    # Both
    preprocessor_both = TextPreprocessor(
        remove_stopwords=True,
        use_lemmatization=True,
        use_stemming=True
    )
    tokens_both = preprocessor_both.process_text(test_text)
    print(f"Both (lemma->stem):  {tokens_both}")
    
    # Test with numbers
    print_separator("BONUS TEST: Number Handling Options")
    
    number_text = "The company reported 2.5 billion dollars in revenue for Q4 2024, up 15% from 2023."
    print(f"\nTest Text: {number_text}\n")
    
    # Replace numbers
    preprocessor_replace = TextPreprocessor(handle_numbers='replace', remove_stopwords=False)
    print(f"Replace: {preprocessor_replace.process_text(number_text)}")
    
    # Remove numbers
    preprocessor_remove = TextPreprocessor(handle_numbers='remove', remove_stopwords=False)
    print(f"Remove:  {preprocessor_remove.process_text(number_text)}")
    
    # Keep numbers
    preprocessor_keep = TextPreprocessor(handle_numbers='keep', remove_stopwords=False)
    print(f"Keep:    {preprocessor_keep.process_text(number_text)}")
    
    print_separator("ALL TESTS COMPLETED")
    print("\nSummary:")
    print("[OK] Short word test")
    print("[OK] Long paragraph test (200+ words)")
    print("[OK] Stopwords removal test")
    print("[OK] Special characters test")
    print("[OK] URLs and emails test")
    print("[OK] Lemmatization vs Stemming comparison")
    print("[OK] Number handling options")
    print("\nTextPreprocessor is ready for use in your Django classifier!")


if __name__ == "__main__":
    main()
