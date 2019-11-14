# Nishant Nikhil (i.nishantnikhil@gmail.com)
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].
# [1] Shen, Y., He, X., Gao, J., Deng, L., and Mesnil, G. 2014. A latent semantic model
#         with convolutional-pooling structure for information retrieval. In CIKM, pp. 101-110.
#         http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf
# [2] http://research.microsoft.com/en-us/projects/dssm/
# [3] http://research.microsoft.com/pubs/238873/wsdm2015.v3.pdf

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K, L)
        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)
    def forward(self, q, pos, negs):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = q.transpose(1,2)
        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c • l_Q + b_c). Note: the paper does not include bias units.
        q_c = F.tanh(self.query_conv(q))
        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1,2)
        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
        # the paper does not include bias units.
        q_s = F.tanh(self.query_sem(q_k))
        q_s = q_s.resize(L)
        # # The document equivalent of the above query model for positive document
        pos = pos.transpose(1,2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(L)
        # # The document equivalent of the above query model for negative documents
        negs = [neg.transpose(1,2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1,2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(L) for neg_s in neg_ss]
        # Now let us calculates the cosine similarity between the semantic representations of
        # a queries and documents
        # dots[0] is the dot-product for positive document, this is necessary to remember
        # because we set the target label accordingly
        dots = [q_s.dot(pos_s)]
        dots = dots + [q_s.dot(neg_s) for neg_s in neg_ss]
        # dots is a list as of now, lets convert it to torch variable
        dots = torch.stack(dots)
        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        with_gamma = self.learn_gamma(dots.resize(J+1, 1, 1))
        # You can use the softmax function to calculate P(D+|Q), but here we return the logits for the CrossEntropyLoss
        # prob = F.softmax(with_gamma)
        return with_gamma

model = CDSSM()

# Build a random data set.
import numpy as np
sample_size = 10
l_Qs = []
pos_l_Ds = []

(query_len, doc_len) = (5, 100)

for i in range(sample_size):
    query_len = np.random.randint(1, 10)
    l_Q = np.random.rand(1, query_len, WORD_DEPTH)
    l_Qs.append(l_Q)
    
    doc_len = np.random.randint(50, 500)
    l_D = np.random.rand(1, doc_len, WORD_DEPTH)
    pos_l_Ds.append(l_D)

neg_l_Ds = [[] for j in range(J)]
for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, J, replace = False)
    for j in range(J):
        negative = negatives[j]
        neg_l_Ds[j].append(pos_l_Ds[negative])

# Till now, we have made a complete numpy dataset
# Now let's convert the numpy variables to torch Variable

for i in range(len(l_Qs)):
    l_Qs[i] = Variable(torch.from_numpy(l_Qs[i]).float())
    pos_l_Ds[i] = Variable(torch.from_numpy(pos_l_Ds[i]).float())
    for j in range(J):
        neg_l_Ds[j][i] = Variable(torch.from_numpy(neg_l_Ds[j][i]).float())


# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# output variable, remember the cosine similarity with positive doc was at 0th index
y = np.ndarray(1)
# CrossEntropyLoss expects only the index as a long tensor
y[0] = 0
y = Variable(torch.from_numpy(y).long())

for i in range(sample_size):
    y_pred = model(l_Qs[i], pos_l_Ds[i], [neg_l_Ds[j][i] for j in range(J)])
    loss = criterion(y_pred.resize(1,J+1), y)
    print (i, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
