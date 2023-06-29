import pickle
with open('hands_version/dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
print(len(dataset["arrays"][0]))