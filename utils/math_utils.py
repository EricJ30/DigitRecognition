import numpy as np

#Compute crossentropy from logits[batch,n_classes] and ids of correct answers
def softmax_crossentropy_with_logits(logits, reference_answers):
    
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))    
    return xentropy
#Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
def grad_softmax_crossentropy_with_logits(logits, reference_answers):

    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)    
    return (- ones_for_answers + softmax) / logits.shape[0]

# Normalize to 0-1
def normalize(X):
    X_normalize = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X_normalize   

#Convert array of labels to one-hot encoded format
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])