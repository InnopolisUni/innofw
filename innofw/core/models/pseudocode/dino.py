"""
Name: Emerging Properties in Self-Supervised Vision Transformers
Paper Link: https://arxiv.org/pdf/2104.14294.pdf
Code Link: https://github.com/facebookresearch/dino


Description: DINO stands for "self DIstillation with NO labels", the idea is to see whether using supervised
learning was preventing transformers from showing the same kind of results in CV as they demonstrated in the
NLP world (where we use self-supervised learning objectives such as (masked) language modeling).

It turns out some nice properties emerge such as:
* DINO-ViT learns to predict segmentation masks
* features are especially of high quality for the k-NN classification


Pseudocode:
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views
    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate
    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)
def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
"""
