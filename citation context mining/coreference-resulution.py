

import spacy
import neuralcoref

# Load English model in SpaCy
nlp = spacy.load('en')

# Add neuralcoref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

def has_coreference(sentence1, sentence2):
    # Combine the sentences into one document
    doc = nlp(sentence1 + " " + sentence2)
    
    # Check if there is a coreference in the document
    if doc._.has_coref:
        # If there is a coreference, check if it involves both sentences
        for cluster in doc._.coref_clusters:
            spans_in_sentence1 = [span for span in cluster if span.start_char < len(sentence1)]
            spans_in_sentence2 = [span for span in cluster if span.start_char >= len(sentence1)]
            
            if spans_in_sentence1 and spans_in_sentence2:
                return True

    return False

# Test the function
sentence1 = "More recently, in Kaplan et al. (2016) they define a similar problem of citation block determination"
sentence2 = "The main difference is that they do not consider the implicit citations that might be non-contiguous to the citing sentence."
print(has_coreference(sentence1, sentence2))  # Should print True
