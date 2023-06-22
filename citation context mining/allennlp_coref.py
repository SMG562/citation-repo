from allennlp.predictors.predictor import Predictor

import allennlp_models.coref

def has_coreference(sent1, sent2):
    # Load the predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

    # Combine the sentences
    text = sent1 + " " + sent2

    # Predict the clusters using the predictor
    result = predictor.predict(document=text)

    # Return whether any clusters are found
    return len(result['clusters']) > 0


# test
sent1 = "Anna has a dog. She loves him."
sent2 = "Her dog's name is Bruno."

print(has_coreference(sent1, sent2))  # should print True

# test on sdp-act data
import pandas as pd

sdp_act_data = pd.read_csv('/Users/hanliu/Downloads/train.txt', sep="\t")
sdp_act_data

citing_sentence = sdp_act_data.loc[1, 'citation_context']
citing_sentence

citing_context = eval(sdp_act_data.loc[1, 'cite_context_paragraph'])
citing_context

for i in citing_context:
  if has_coreference( citing_sentence, i):
    print(i)
