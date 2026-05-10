
import sys
import os

# Simple model builder using existing test approach
def build_model():
    # Create mini corpus for model building
    with open('test_corpus.txt', 'r') as f:
        corpus_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f'Building model from {len(corpus_lines)} corpus lines')
    
    # Use the existing test approach but save model
    os.system('TEST_CORPUS=test_corpus.txt ./test/test_simple_tokenizer > build_output.txt 2>&1')
    
    # Check if model was created
    if os.path.exists('tokenizer_model.bin'):
        print('Model created successfully')
        return True
    else:
        print('Model creation failed')
        return False

if __name__ == '__main__':
    build_model()
