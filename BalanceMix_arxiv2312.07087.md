# Paper: Toward Robustness in Multi-label Classification

**Original Title:** Toward Robustness in Multi-label Classification: A Data Augmentation Strategy against Imbalance and Noise

**Authors:** Hwanjun Song, Minseok Kim, Jae-Gil Lee

**Source:** [arXiv:2312.07087](https://arxiv.org/abs/2312.07087)

---

```
Toward Robustness in Multi-label Classication:
A Data Augmentation Strategy against Imbalance and Noise
Hwanjun Song
1
, Minseok Kim
2
, Jae-Gil Lee
1
1
KAIST,
2
Amazon
f
songhwanjun, jaegil
g
@kaist.ac.kr, kminseok@amazon.com
Abstract
Multi-label classication poses challenges due to imbalanced
and noisy labels in training data. We propose a unied data
augmentation method, named BalanceMix, to address these
challenges. Our approach includes two samplers for imbal-
anced labels, generating minority-augmented instances with
high diversity. It also renes multi-labels at the label-wise
granularity, categorizing noisy labels as clean, re-labeled, or
ambiguous for robust optimization. Extensive experiments on
three benchmark datasets demonstrate that BalanceMix out-
performs existing state-of-the-art methods. We release the
code at https://github.com/DISL- Lab/BalanceMix.
Introduction
The issue of data-label quality emerges as a major concern
in the practical use of deep learning, potentially resulting in
catastrophic failures when deploying models in real-world
test scenarios (Whang et al. 2021). This concern is magni-
ed in multi-label classication, where instances can be as-
sociated with multiple labels simultaneously. In this context,
AI system robustness is at risk due to diverse types of data-
label issues, although the task can reect the complex rela-
tionships present in real-world data (Bello et al. 2021).
The presence of
class imbalance
occurs when a few
majority classes occupy most of the positive labels, and
positive-negative imbalance
arises due to instances typically
having fewer positive labels but numerous negative labels.
Such imbalanced labels can dominate the optimization pro-
cess and lead to underemphasizing the gradients from mi-
nority classes or positive labels. Additionally, the presence
of
noisy labels
stems from the costly and time-consuming
nature of meticulous annotation (Song et al. 2022). Labels
can be corrupted by adversaries or system failures (Zhang
et al. 2020). Notably, instances have both clean and incorrect
labels, therefore resulting in diverse cases of noisy labels.
Three distinct types of noisy labels arise in multi-label
classication, as illustrated in Fig. 1:
mislabeling
, where a
visible object in the image is labeled incorrectly by a human
or machine annotator, such as a dog being labeled as a cat;
random ipping
, where labels are randomly ipped by an
adversary regardless of the presence of other class objects,
Copyright © 2024, Association for the Advancement of Articial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: Examples of noisy labels in multi-label classica-
tion.
1
and
0
indicate positive and negative labels, symboliz-
ing the existence of an object class.
such as negative labels for a cat and a bowl being ipped
independently to positive labels; and
(partially) missing la-
bels
, where even humans cannot nd all applicable class la-
bels for each image, and it is more difcult to detect their
absence than to detect their presence (Cole et al. 2021).
Ensuring the robustness of AI systems calls for a
holis-
tic
approach that effectively operates within the following
settings: clean, noisy, missing, and imbalanced labels at the
same time. However, this task is non-trivial given that minor-
ity and noisy labels have similar behavior in learning, e.g.,
larger gradients, making the task even more complicated
and challenging. As a result, prior studies have addressed
these two problems
separately
in different setups, assum-
ing either clean or well-balanced training dataŠi.e., imbal-
anced clean labels (Lin et al. 2017; Ben-Baruch et al. 2021;
Du et al. 2023) and well-balanced noisy labels (Zhao and
Gomes 2021; Ferreira, Costeira, and Gomes 2021; Shikun
et al. 2022; Wei et al. 2023).
We address this challenge using a new data augmentation
method,
BalanceMix
, without complex data preprocessing
and architecture change. First, for imbalanced multi-labels,
we maintain an additional batch sampler called a
minority
sampler
, which samples the instances containing minority
labels with high probability, as illustrated in Fig. 2(a). To
counter the limited diversity in oversampling, we interpolate
the instances sampled from the minority sampler with those
sampled from a random sampler using the Mixup (Zhang
et al. 2018) augmentation. By mixing with a higher weight
to the random instances, the sparse context of the oversam-
pled instances literally
percolates
through the majority of
training data without losing diversity. Minority sampling in
Fig. 2(a) followed by the Mixup augmentation in Fig. 2(c) is
called
minority-augmented mixing
.
arXiv:2312.07087v1  [cs.LG]  12 Dec 2023

(a) Minority Sampling (w. Noisy Multi-labels). (b) Fine-grained Label-wise Management. (c) Mixing Augmentation.
Figure 2: Overview of BalanceMix. Here with MS-COCO, ﬁpersonﬂ and ﬁcarﬂ are the most frequently observed (majority)
classes, whereas ﬁhair dryer,ﬂ ﬁscissor,ﬂ and ﬁtoasterﬂ are the least frequently observed (minority) classes in the training data.
Then, for noisy multi-labels, we incorporate
ne-grained
label-wise management
to feed high-quality multi-labels
into the augmentation process. Unlike existing robust learn-
ing methods such as Co-teaching (Han et al. 2018) which
consider each instance as a candidate for selection or correc-
tion, we should move to a ner granularity and consider
each
label
as a candidate. As illustrated in Fig. 2(b), the label-
wise management step categorizes the entire set of noisy la-
bels into three subsets:
clean
labels which are expected to be
correct with high probability;
re-labeled
labels whose ip-
ping is corrected with high condence; and
ambiguous
la-
bels which need to be downgraded in optimization. Putting
our solutions for imbalanced and noisy labels together, Bal-
anceMix is completed, as illustrated in Fig. 2.
Our technical innovations are successfully incorporated
into the well-established techniques of oversampling and
Mixup, enabling easy integration into the existing training
pipeline. Our contributions are threefold: (1) BalanceMix
serves as a versatile data augmentation technique, demon-
strating reliable performance across clean, noisy, missing,
and imbalanced labels. (2) BalanceMix avoids overtting
to minority classes and incorrect labels thanks to minority-
augmented mixing with ne-grained label management. (3)
BalanceMix outperforms existing prior arts and reaches
91.7mAP on the MS-COCO data, which is the state-of-the-
art performance with the ResNet backbone.
Related Work
Multi-label with Imbalance.
One of the main trends in
this eld is solving long-tail class imbalance and positive-
negative label imbalance. There have been classical resam-
pling approaches (Wang, Minku, and Yao 2014; Loyola-
Gonz
´
alez et al. 2016) for imbalance, but they are mostly
designed for a single-label setup. A common solution for
class imbalance with multi-labels is the focal loss (Lin
et al. 2017), which down-weights the loss value of each la-
bel gradually as a model's prediction condence increases,
highlighting difcult-to-learn minority class labels; how-
ever, it can lead to overtting to incorrect labels. The asym-
metric focal loss (ASL) (Ben-Baruch et al. 2021) modies
the focal loss to operate differently on positive and negative
labels for the imbalance. (Yuan et al. 2023) proposed a bal-
ance masking strategy using a graph-based approach.
Multi-label with (Partially) Missing Labels.
Annotation
in the multi-label setup becomes harder as the number of
classes increases. Subsequently, the need to handle missing
labels has recently gained a lot of attention. A simple so-
lution is regarding all the missing labels as negative labels
(Wang et al. 2014), but it leads to overtting to incorrect neg-
ative ones. There have been several studies with deep neu-
ral networks (DNNs). (Durand, Mehrasa, and Mori 2019)
adopted curriculum learning for pseudo-labeling based on
model predictions. (Huynh and Elhamifar 2020) used the
inferred dependencies among labels and images to prevent
overtting. Recently, (Cole et al. 2021) and (Kim et al. 2022,
2023) addressed the hardest version, where only a single
positive label is provided for each instance. They proposed
multiple solutions including label smoothing, expected pos-
itive regularization, and label selection and correction. How-
ever, the imbalance problem is overlooked, and all the labels
are simply assumed to be clean.
Classication with Noisy Labels.
For
single
-label clas-
sication, learning with noisy labels has established multi-
ple directions. Most approaches are based on the memoriza-
tion effect of DNNs, in which simple and gen eralized pat-
terns are prone to be learned before the overtting to noisy
patterns (Arpit et al. 2017). More specically, instances
with small losses or consistent predictions are treated as
clean instances, as in Co-teaching (Han et al. 2018), O2U-
Net (Huang et al. 2019), and CNLCU (Xia et al. 2022); in-
stances are re-labeled based on a model's predictions for la-
bel correction, as in SELFIE (Song, Kim, and Lee 2019) and
SEAL (Chen et al. 2021). A considerable effort has also been
made to use semi-supervised learning, as in DivideMix (Li,
Socher, and Hoi 2020) and PES (Bai et al. 2021). In addi-
tion, a few studies have addressed class imbalance in the
noisy single-label setup (Wei et al. 2021; Ding et al. 2022),
but they cannot be immediately applied to the multi-label
setup owing to their inability to handle the various types of
label noise caused by the nature of having both clean and
incorrect labels in one instance.
For
multi
-label classication with noisy labels, there has
yet to be studied actively owing to the inherent complex-

ity including diverse types of label noise and imbalance.
CbMLC (Zhao and Gomes 2021) addresses label noise by
proposing a context-based classier, but its architecture is
conned to graph neural networks and requires large pre-
trained word embeddings. A method by Hu et al.(Hu et al.
2018) utilizes a teacher-student network with feature trans-
formation. SELF-ML (Ferreira, Costeira, and Gomes 2021)
re-labels an incorrect label using a combination of clean la-
bels, but it works only when multi-labels can be dened
as attributes associated with each other. ASL (Ben-Baruch
et al. 2021) solves the problem of mislabeling by shift-
ing the prediction probability of low-condence negative la-
bels, making their losses close to zero in optimization. T-
estimator (Shikun et al. 2022) solves the estimation problem
of the noise transition matrices in the multi-label setting.
Oversampling with Mixup.
Prior studies have applied
Mixup to address class imbalance (Guo and Wang 2021; Wu
et al. 2020; Galdran, Carneiro, and Gonz
´
alez Ballester 2021;
Park et al. 2022). Yet, they mainly focus on
single
-label
classication, overlooking positive-negative imbalances and
noisy labels. We propose the rst approach that uses predic-
tive condence to dynamically adjust the degree of oversam-
pling for both types of imbalance while employing label-
wise management for noisy labels.
Problem Denition
A multi-label multi-class classication problem requires
training data
D
, a set of two random variables (
x
,
y
) which
consists of an instance (
d
-dimensional feature)
x
2 X
(
ˆ
R
d
)
and its multi-label
y
2 f
0
;
1
g
K
, where
K
is the num-
ber of applicable classes. However, in the presence of la-
bel noise, the noisy multi-label
~
y
2 f
0
;
1
g
K
possibly con-
tains incorrect labels originated from mislabeling, random
ipping, and missing labels; that is, a noisy label
~
y
k
2
~
y
may not be equal to the true label
y
k
2
y
. Thus, let
~
D
=
f
(
x
n
;
~
y
n
)
g
N
n
=1
be the noisy training data of size
N
.
Label Noise Modeling.
We dene three types of label
noise. From the statistical point of view, (1)
mislabeling
is
dened as
class-dependent
label noise, where a class ob-
ject in the image is incorrectly labeled as another class ob-
ject that may not be visible. The ratio of a class
c
1
be-
ing mislabeled as
c
2
is formulated by
ˆ
c
1
!
c
2
=
p
( ~
y
c
1
=
0
;
~
y
c
2
= 1
j
y
c
1
= 1
; y
c
2
= 0)
. In contrast, (2)
random ipping
is
class-independent
label noise, where the presence (or ab-
sence) of a class
c
is randomly ipped with a probability of
ˆ
c
=
p
( ~
y
c
= 1
j
y
c
= 0) =
p
( ~
y
c
= 0
j
y
c
= 1)
, which is inde-
pendent of the presence of other classes. This scenario can
be caused by an adversary's attack or a system failure. Last,
(3)
missing labels
from partial labeling can be considered as
a type of label noise, where all missing labels are treated as
negative ones.
Optimization.
To deal with multi-labels in optimization,
the most widely-used approach is solving
K
binary classi-
cation problems using the binary cross-entropy (BCE) loss.
Given a DNN parameterized by

, the DNN is updated via
stochastic gradient descent to minimize the expected BCE
(a) Prediction Condence. (b) Average Precision (AP).
Figure 3: AP and Prediction condence in COCO using the
BCE loss at the 40
%
of training epochs, where 80 classes
are partitioned into ten groups in the descending order of
positive label frequency. The Pearson correlation coefcient
is computed between ten class groups.
loss on the mini-batch
B
ˆ
~
D
,
L
(
B
; ) =
1
j
B
j
X
(
x
;
~
y
)
2
B
K
X
k
=1
BCE(
f
(
x
;
~
y
k
)
)
;
where
BCE(
f
(
x
;
~
y
k
)
) =

~
y
k

log (
f
(
x
;
~
y
k
)
)

(1

~
y
k
)

log (1

f
(
x
;
~
y
k
)
)
:
(1)
Given the instance
x
,
f
(
x
;
~
y
k
)
and
1

f
(
x
;
~
y
k
)
are the con-
dence in presence and absence, respectively, for the
k
-th
class by the model

. BalanceMix is built on top of this
standard optimization pipeline for multi-label classication.
Methodology: BalanceMix
Our primary idea is to generate minority-augmented in-
stances and their reliable multi-labels through
data aug-
mentation
. We now detail the two main components, which
achieve balanced and robust optimization by
minority-
augmented mixing
and
label-wise management
. The pseu-
docode of BalanceMix is provided in Appendix A.
Minority-augmented Mixing
To relieve the class imbalance problem, prior studies either
oversample the minority class labels or adjust their loss val-
ues (Tarekegn, Giacobini, and Michalak 2021). These meth-
ods are intuitive but rather intensify the overtting problem
since they rely on a few minority instances with limited di-
versity (Guo and Wang 2021). On the other hand, we lever-
age random instances to increase the diversity of minority
instances by
separately
maintaining two samplers in Fig. 2.
Condence-based Minority Sampling.
Prior oversam-
pling methods that rely on the frequency of positive labels
face two key limitations. First, this frequency alone does not
identify the minority multi-labels with low AP values; as il-
lustrated in Fig. 3(a), the class group with few positive labels
does not always have lower AP values due to the complexity
of the two types of imbalance in the multi-label setup. Sec-
ond, there is a risk of overtting because of sticking to the
same oversampling policy during the entire training period.
To address these limitations, we rst propose to employ
the prediction condence
f
(
x
;
~
y
k
)
, which exhibits a strong
correlation with the AP, as shown in Fig. 3(b). We opt to
oversample the instances with low prediction condence in
their multi-labels, as they are expected to contribute the most
signicant increase in the AP. Initially, We dene two con-

dence scores for a
specic
class
k
,
P(
k
) =
1
jP
k
j
X
(
x
;
~
y
)
2P
k
f
(
x
;
~
y
k
)
;
A(
k
) =
1
jA
k
j
X
(
x
;
~
y
)
2A
k
(1

f
(
x
;
~
y
k
)
)
;
s
:
t
:
P
k
=
f
(
x
;
~
y
)
2
~
D
: ~
y
k
= 1
g
;
A
k
=
f
(
x
;
~
y
)
2
~
D
: ~
y
k
= 0
g
;
(2)
which are the expected prediction condences, respectively,
for the presence (P) and absence (A) of the
k
-th class. Next,
the condence score of an instance
(
x
;
~
y
)
is dened by ag-
gregating Eq. (2) for
all
the classes,
Score
(
x
;
~
y
) =
K
X
k
=1
1
[ ~
y
k
=1]
P(
k
) +
1
[ ~
y
k
=0]
A(
k
)
:
(3)
Then, the sampling probability of
(
x
;
~
y
)
is formulated as
p
sampling

(
x
;
~
y
);
~
D

=
1
=
Score
(
x
;
~
y
)
P
(
x
0
;
~
y
0
)
2
~
D
1
=
Score(
x
0
;
~
y
0
)
:
(4)
By doing so, we consider positive-negative imbalance to-
gether with class imbalance by relying on the prediction con-
dence, which marks a signicant difference from existing
methods (Galdran, Carneiro, and Gonz
´
alez Ballester 2021;
Park et al. 2022) that considers only the class imbalance of
positive labels. Further, our condence-based sampling dy-
namically adjusts the degree of oversampling over a training
period, mitigating the risk of overtting. Minority instances
are initially oversampled with a high probability, but the de-
gree of oversampling gradually decreases as the imbalance
problem gets resolved (see Figure 6 in Appendix B).
Mixing Augmentation.
To mix the instances from the two
samplers. We adopt the Mixup (Zhang et al. 2018) augmen-
tation because it can mix two instances even when multi-
labels are assigned to them. Let
(
x
R
;
~
y
R
)
and
(
x
M
;
~
y
M
)
be
the instances sampled from the random and minority sam-
plers, respectively. The minority-augmented instance is gen-
erated by their interpolation,
x
mix
=

x
R
+ (1


)
x
M
;
~
y
mix
=

~
y
R
+ (1


)
~
y
M
where

= max(

0
;
1


0
)
;
(5)
and

0
2
[0
;
1]
˘
Beta(
 ; 
)
. By the second row of Eq. (5),

becomes greater than or equal to
0
:
5
; thus, the instance of
the random sampler amplies diversity, while that of the mi-
nority sampler adds the context of minority classes.
Mixing
one random instance and one controlled (minor) instance
,
instead of mixing two random instances, is a simple yet ef-
fective strategy, as shown in the evaluation.
Fine-grained Label-wise Management
Before mixing the two instances by Eq. (5), to make noisy
multi-labels reliable in support of robust optimization, we
perform
label-wise
renement.
Clean Labels.
To relieve the imbalance pro blem in label
selection, we separately identify clean labels for each class.
Let
L
( ~
y
k
=1)
and
L
( ~
y
k
=0)
be the sets of the BCE losses of the
positive and negative labels of the
k
-th class,
L
( ~
y
k
=
l
)
=
f
BCE(
f
(
x
;
~
y
k
=
l
)
)
j
(
x
;
~
y
)
2
~
D ^
~
y
k
2
~
y
^
~
y
k
=
l
g
;
(6)
where
l
is 1 or 0 for the positive or negative label.
Clean labels exhibit loss values smaller than noise ones
due to the memorization effect of DNNs (Li, Socher, and Hoi
2020). Hence, we t a bi-modal univariate Gaussian mixture
model (GMM) to each set of the BCE losses in using the
expectation-maximization (EM) algorithm, returning
2

K
GMM models for positive and negative labels of
K
classes,
p
G
=
f
(
G
( ~
y
k
=1)
;
G
( ~
y
k
=0)
)
g
K
k
=1
:
(7)
Given the BCE loss of
x
for the
k
-th positive or negative
label, its clean-label probability is obtained by the posterior
probability of the corresponding GMM,
p
G
(
x
;
~
y
k
=
l
) =
G
( ~
y
k
=
l
)
(BCE(
f
(
x
;
~
y
k
=
l
)
)
j
g
)
 G
( ~
y
k
=
l
)
(
g
)
G
( ~
y
k
=
l
)
(BCE(
f
(
x
;
~
y
k
=
l
)
))
;
(8)
where
g
denotes a modality for the small-loss (clean) label.
Thus, a label with
p
G
>
0
:
5
is marked as being clean.
The time complexity of GMM modeling is
O
(
N GD
) =
O
(
N
)
and thus linear to the number of instances
N
, where
the number of modalities
G
= 2
and the number of dimen-
sions
D
= 1
(see (Trivedi et al. 2017) for the proof of the time
complexity). Since we model the GMMs once per epoch,
the cost involved is expected to be small compared with the
training steps of a complex DNN.
Re-labeled Labels.
Before the overtting to noisy labels,
a model's prediction delivers useful underlying information
on correct labels (Song, Kim, and Lee 2019). Therefore, we
modify the given label if it is not selected as a clean one
but the model exhibits high condence in predictions. To
obtain a stable condence from the model, we ensemble
the prediction condences on two augmented views cre-
ated by RandAug (Cubuk et al. 2020). Given two differently-
augmented instances from the original instance
x
whose
p
G
(
x
;
~
y
k
)

0
:
5
, the
k
-th label is re-labeled by
1
=
2

(
f
(aug
1
(
x
)
;
~
y
k
)
+
f
(aug
2
(
x
)
;
~
y
k
)
)
> 
=
)
~
y
k
= 1
;
1
=
2

(
f
(aug
1
(
x
)
;
~
y
k
)
+
f
(aug
2
(
x
)
;
~
y
k
)
)
<
1


=
)
~
y
k
= 0
;
(9)
where

is the condence threshold for re-labeling.
Ambiguous Labels.
The untouched lab els, which are nei-
ther clean nor re-labeled, are regarded as being ambiguous.
These labels are potentially incorrect, but they could hold
meaningful information in learning with careful treatment.
To squeeze the meaningful information and reduce the po-
tential risk, for these ambiguous labels, we execute
impor-
tance reweighting
which decays a loss based on the clean-
label probability estimated by Eq. (8).
Optimization with BalanceMix
Given two instances sampled from the random and minor-
ity samplers, their multi-labels are rst rened by the label-
wise management setup. The
reliability
of each rened la-
bel is stored as
C
for clean labels,
R
for re-labeled labels,
and
U
for ambiguous labels. Then, a minority-augmented
instance is generated by Mixup with the mixed multi-labels.
The reliability of each label of the augmented instance fol-
lows that of the instance selected by the random sampler, be-
cause it always dominates in mixing by


0
:
5
in Eq. (5).

(a) Default. (b) Methods for Noisy Labels. (c) Methods for Missing Labels. (d) Ours.
Figure 4: Performance rankin g (1Œ7) from ve different perspectives. BalanceMix is the
most versatile
to handle diverse types
of label issues in multi-label classication. A number in a pentagon is its area, roughly meaning the overall performance.
Class Group
All
Many-shot
Medium-shot
Few-shot
Category
Method
0% 20% 40%
0% 20% 40%
0% 20% 40%
0% 20% 40%
Default
BCE Loss
83.4 73.1 63.8
86.9 78.4 68.4
84.5 74.0 64.3
64.3 55.5 48.0
Noisy Labels
Co-teaching
82.8 82.5 78.6
87.2 87.0 82.7
84.0 83.7 79.7
61.2 60.9 54.5
ASL
85.0
82.8
80.3
88.4
87.4
85.4
86.3
84.0
81.8
67.5
62.5 55.6
T-estimator
84.3 82.2 80.5
87.5 86.3 85.3
85.4 83.6 81.3
67.0 59.7 61.0
Missing Labels
LL-R
82.5 80.7 75.3
81.0 83.9 79.8
83.8 81.8 76.7
65.7 63.1 52.6
LL-Ct
79.4 81.3 77.4
72.3 78.8 79.3
80.5 82.6 78.8
67.0 64.3
55.9
Proposed
BalanceMix
85.2 84.3 81.6
88.4 87.7 85.5
86.1
85.2 82.6
70.2 68.7 63.1
Table 1: Last mAPs on MS-COCO with
mislabeling
of
0
Œ
40%
. The 1st and 2nd best values are in bold and underlined.
The loss function of BalanceMix is dened on the minority-
augmented mini-batch
B
mix
by
L
ours
(
B
mix
; ) =
1
j
B
mix
j
X
(
x
mix
;
~
y
mix
)
2
B
mix
`
(
x
mix
;
~
y
mix
)
;
where
`
(
x
mix
;
y
mix
) =
X
k
2
C
[
R
BCE(
f
(
x
mix
;
~
y
mix
k
)
)
+
X
k
2
U
p
G
(
x
mix
;
~
y
mix
k
)

BCE(
f
(
x
mix
;
~
y
mix
k
)
)
:
(10)
We perform standard training for warm-up epochs and then
apply the proposed loss function in Eq. (10).
Evaluation
Datasets.
Pascal-VOC (Everingham et al. 2010) and MS-
COCO (Lin et al. 2014) are the most widely-used datasets
with well-curated labels of 20 and 80 common classes. In
contrast, DeepFashion (Liu et al. 2016) is a real-world in-
shopping dataset with noisy weakly-annotated labels for
1
;
000
descriptive attributes.
Imbalanced and Noisy Labels.
The three datasets con-
tain different levels of natural imbalance. Pascal-VOC, MS-
COCO, and DeepFashion have the class imbalance ratios
1
of
14
,
339
, and
239
, and the positive-negative imbalance ra-
tios of
13
,
27
, and
3
, respectively. See the detailed analysis
of the imbalance in Appendix C. We articially contami-
nate Pascal-VOC and MS-COCO to add three types of label
noise. First, for mislabeling, we inject class-dependent label
noise. Given a noise rate
˝
, the presence of the
i
-th class
is mislabeled as that of the
j
-th class with a probability of
1
The ratio of the number of the instances in the most frequent
class to that of the instances in the least frequent class.
ˆ
i
!
j
; we follow the protocol used for a long-tail noisy label
setup (Wei et al. 2021). For the two different classes
i
and
j
,
ˆ
i
!
j
=
p
( ~
y
i
= 0
;
~
y
j
= 1
j
y
i
= 1) =
˝

N
j
=
(
N

N
i
)
;
(11)
where
N
i
is the number of positive labels for the
i
-th class.
Second, for random ipping, all positive and negative la-
bels are ipped independently with the probability of
˝
.
Third, for missing labels, we follow the single positive label
setup (Kim et al. 2022), where one positive label is selected
at random and the other positive labels are dropped.
Algorithms.
We use the ResNet-50 backbone pre-trained
on ImageNet-1K and ne-tune using SGD with a mo-
mentum of 0.9 and resolution of
448

448
. We com-
pare BalanceMix with a standard method using the BCE
loss (Default) and
ve
state-of-the-art methods, categorized
into two groups. The former is to handle
noisy
labels based
on instance-level selection, loss reweighting, and noise tran-
sition matrix estimationŠCo-teaching (Han et al. 2018),
ASL (Ben-Baruch et al. 2021), and T-estimator (Shikun et al.
2022). The latter is to handle
missing
labels based on la-
bel rejection and correctionŠLL-R and LL-Ct (Kim et al.
2022). For data augmentation, we apply RandAug and
Mixup to all methods, except Default using only RandAug.
The results of Default with Mixup are presented in Table 5.
As for our hyperparameters, the coefcient

for Mixup is
set to be
4
:
0
; and the condence threshold

for re-labeling
is set to be
0
:
975
for the standard, mislabeling, and random
ipping settings with
multiple
positive labels, but it is set
to be
0
:
550
for the missing label setting with a
single
posi-
tive label. More details of conguration and hyperparameter
search can be found in Appendices D and E.
Evaluation Metric.
We report the overall validation (or
test) mAP at the last epoch over three disjoint class sub-

Class Group
All
Many-shot
Medium-shot
Few-shot
Category
Method
0% 20% 40%
0% 20% 40%
0% 20% 40%
0% 20% 40%
Default
BCE Loss
83.4 59.8 43.5
86.9 71.2 65.1
84.5 60.6 43.2
64.3 38.9 30.6
Noisy Labels
Co-teaching
82.8 65.5 43.6
87.2 76.1 61.9
84.0 66.8 43.8
61.2 38.3 25.9
ASL
85.0
75.0
66.2
88.4
84.4
82.8
86.3
77.0
67.7
67.5
39.2 30.8
T-estimator
84.3 74.3 69.9
87.5 82.6 80.8
85.4 76.0 71.5
67.4 43.6 39.1
Missing Labels
LL-R
82.5 74.0 69.3
81.0 77.2 76.6
83.8 75.8 71.0
65.7 46.0
38.6
LL-Ct
79.4 73.2 70.1
72.3 75.5 76.5
80.5 75.0 71.8
67.0 45.5 41.1
Proposed
BalanceMix
85.2 76.5 74.5
88.4 84.5
81.3
86.1
78.2 76.3
70.2 46.1 43.0
Table 2: Last mAPs on MS-COCO with
random ipping
of
0
Œ
40%
. The 1st and 2nd best values are in bold and underlined.
Datasets
MS-COCO
Pascal-VOC
Category
Method
All Many Medium Few
All Medium Few
Default
BCE Loss
69.7 71.7 70.6 54.4
85.7 89.2 84.2
Noisy Labels
Co-teaching
68.1 61.5 69.2 59.1
80.9 87.2 78.1
ASL
73.3
77.7
74.7 49.2
86.8 82.1 88.8
T-estimator
16.8 43.2 16.3 3.3
86.2 88.9 85.0
Missing Labels
LL-R
74.2 75.4 75.3 58.7
89.1 91.5
88.1
LL-Ct
76.9
77.4
78.2
57.6
89.3
91.5
88.3
Proposed
BalanceMix
77.4
76.2
78.5 61.3
92.6 94.5 91.8
Table 3: Last mAPs on MS-COCO and Pascal-VOC in the
missing label (single positive label)
setup.
sets: many-shot (more than 10,000 positive labels), medium-
shot (from 1,000 to 10,000 positive labels), and few-shot
(less than 1,000 positive labels) classes. The result at the
last epoch is commonly used in the literature on robustness
to label noise (Han et al. 2018). We repeat every task thrice,
and see Appendix F for the standard error.
Overall Analysis on Five Perspectives
Fig. 4 shows the overall performance rankings aggregated
2
on Pascal-VOC and MS-COCO for ve different perspec-
tives: ﬁCleanﬂ for when label noise is not injected, ﬁMisla-
belﬂ for when labels are mislabeled with the noise ratio of
20
Œ
40%
, ﬁRand. Flipﬂ for when labels are randomly ipped
with the noise ratio of
20
Œ
40%
, ﬁMissingﬂ for when the sin-
gle positive label setup is used, and ﬁImbal.ﬂ for when few-
shot classes without label noise are used.
Only BalanceMix operates in all scenarios with high per-
formance: its minority-augmented mixing overcomes the
problem of
imbalanced labels
while its ne-grained label-
wise management adds robustness to
diverse types of label
noise
. Except BalanceMix, the ve existing methods have
pros and cons. The three methods of handling noisy labels in
Fig. 4(b) generally perform better for mislabeling and ran-
dom ipping than the others; but the instance-level selec-
tion of Co-teaching is not robust to random ipping where a
signicant number of negative labels are ipped to positive
ones. In contrast, the two methods of handling missing la-
bels in Fig. 4(c) perform better with the existence of missing
labels than Co-teaching, ASL, and T-estimator. LL-Ct (label
correction) is more suitable than LL-R (label rejection) for
mislabeling and random ipping since label correction has a
2
For each perspective, we respectively compute the ranking on
each dataset and then sum up the rankings to get the nal one.
Class Group
All
Many Medium Few
BCE Loss
75.2
93.4 84.4 53.4
Co-teaching
66.8
90.7 81.3 32.8
ASL
76.4
94.4 85.2 55.4
T-estimator
75.4
94.7 84.8 53.1
LL-R
75.3
93.3 84.2 53.8
LL-Ct
75.2
92.6 84.2 53.8
BalanceMix
77.0
95.2 85.6 56.4
Table 4: mAPs on DeepFashion with
real-world
noisy
multi-labels using seven multi-label classication methods.
potential to re-label some of incorrect labels. For the imbal-
ance, ASL shows reasonable performance on the few-shot
subset by adopting a modied focal loss.
Results on Imbalanced and Noisy Labels
We evaluate the performance of BalanceMix on MS-COCO
with three types of synthetic label noise and on DeepFashion
with
real-world
label noise. We defer the results on Pascal-
VOC to Appendix F for the sake of space.
Mislabeling (Table 1).
BalanceMix achieves not only the
best overall mAP (see the ﬁAllﬂ column) with varying misla-
beling ratios, but also the best mAP on few-shot classes (see
the ﬁFew-shotﬂ column). It shows higher robustness even
compared with the three methods designed for noisy labels.
ASL performs well among the compared methods, but its
weighting scheme of pushing higher weights to difcult-
to-learn labels could lead to overtting to difcult incor-
rect labels; hence, when the noise ratio increases, its per-
formance rapidly degrades from
67
:
5%
to
55
:
6%
in the few-
shot classes. Both methods for missing labels perform bet-
ter than the default method (BCE), but are still vulnerable to
mislabeling.

Component
Clean Label
Mislabel 40%
Rand Flip 40%
Missing Label
Overall (Mean)
Default (BCE Loss)
83.4
63.3
43.0
72.6
65.6
+
Random Sampler (
ˇ
Mixup)
84.2 (
+
0.8)
67.4 (
+
3.3)
64.9 (
+
21.9)
73.3 (
+
0.7)
72.5 (
+
6.9)
+
Minority Sampler (in Eq. (4))
85.1 (
+
1.7)
70.2 (
+
6.9)
67.2 (
+
24.2)
74.2 (
+
1.6)
74.2 (
+
8.6)
+
Clean Labels (in Eq. (8))
84.9 (

0.2)
76.1 (
+
5.9)
74.9
(
+
7.7)
74.6 (
+
0.4)
77.6 (
+
3.4)
+
Re-labeled Labels (in Eq. (9))
85.3
(
+
0.4)
80.2 (
+
3.9)
74.9
(
+
0.0)
76.1 (
+
1.5)
79.1 (
+
1.5)
+
Ambiguous Labels (in Eq. (10))
85.3
(
+
0.0)
81.6
(
+
1.4)
74.5 (

0.4)
77.4
(
+
1.3)
79.7
(
+
0.6)
Table 5: Component analysis of BalanceMix on MS-COCO. The values in parentheses are the gain caused by each component.
Random Flipping (Table 2).
This is more challenging
than mislabeling noise, in considering that even negative la-
bels are ipped by a given noise ratio. Accordingly, the mAP
of Co-teaching and ASL drops signicantly when the noise
ratio reaches
40%
(see the ﬁAllﬂ column), which implies
that instance selection in Co-teaching and loss reweight-
ing in ASL are ineffective to overcome random ipping.
T-estimator shows a better result at the noise ratio of
40%
than ASL by estimating the noise transition matrix per class.
Overall, BalanceMix achieves higher robustness against a
high ipping ratio of
40%
with ne-grained label-wise man-
agement; its performance drops by only
10
:
7%
p
, which is
much smaller than
39
:
2%
p
,
18
:
8%
p
, and
14
:
4%
p
of Co-
teaching, ASL, and T-estimator, respectively. Thus, it main-
tains the best mAP for all class subsets in general.
Missing Labels (Table 3).
Unlike the mislabeling and ran-
dom ipping, LL-R and LL-Ct generally show higher mAPs
than the methods for noisy labels, because LL-R and LL-Ct
are designed to reject or re-label unobserved positive labels
that are erroneously considered as negative ones. Likewise,
the label-wise management of BalanceMix includes the re-
labeling process, xing incorrect positive and negative labels
to be correct. In addition, it shows higher mAP in the few-
shot classes than LL-Ct due to the consideration of imbal-
anced labels. Thus, it consistently maintains its performance
dominance. Meanwhile, T-estimator performs badly in MS-
COCO due to the complexity of transition matrix estimation.
Real-world Noisy Labels (Table 4).
A real-world noisy
dataset, DeepFashion, likely contains
all
the label noisesŠ
mislabeling, random ipping, and missing labelsŠalong
with class imbalance. Therefore, our motivation for a holis-
tic approach is of importance for real use cases.
The relatively small performance gain is attributed to a
small percentage (around 8%) of noise labels (Song et al.
2022) in DeepFashion, because its ne-grained labels were
annotated via a crowd-sourcing platform which can be rel-
atively reliable. The performance gain will increase for
datasets with a higher noise ratio.
Component Ablation Study
We conduct a component ablation study by adding the main
components one by one on top of the default method. Table
5 summarizes the mAP and average performance of each of
ve variants. The rst variant of using only a random sam-
pler is equivalent to the original Mixup.
First, using only a random sampler like Mixup does not
sufciently improve the model performance, but adding the
minority sampler achieves sufcient improvement because
it takes imbalanced labels into account. Second, exploiting
Method
Backbone
Resolution
mAP (All)
MS-CMA
ResNet-101
448

448
83.8
ASL
ResNet-101
448

448
85.0
ML-Decoder
ResNet-101
448

448
87.1
BalanceMix
ResNet-101
448

448
87.4 (+0.3)
ASL
TResNet-L
448

448
88.4
Q2L
TResNet-L
448

448
89.2
ML-Decoder
TResNet-L
448

448
90.0
BalanceMix
TResNet-L
448

448
90.5 (+0.5)
ML-Decoder
TResNet-L
640

640
91.1
ML-Decoder
TResNet-XL
640

640
91.4
BalanceMix
TResNet-L
640

640
91.7 (+0.6)
Table 6:
State-of-the-art comparison
on MS-COCO. The
values in parentheses are the improvements over the latest
method using the same backbone.
only the selected clean labels increases the mAP when pos-
itive labels are corrupted with mislabeling and random ip-
ping. However, this approach is not that benecial in the
clean and missing label setups, where all positive labels are
regarded as being clean; it also simply discards all (expect-
edly) unclean negative labels without any treatment. Third,
re-labeling complements the limitation of clean label se-
lection, providing additional mAP gains in most scenarios.
Fourth, using ambiguous labels adds further mAP improve-
ment except for the random ipping setup.
In summary, since all the components in BalanceMix gen-
erally add a synergistic effect, leveraging all of them is rec-
ommended for use in practice. In Appendix G, we (1) ana-
lyze the impact of minority-augmented mixing on diversity
changes, (2) provide the pure effect of label-wise manage-
ment, and (3) report its accuracy in selecting clean labels
and re-labeling incorrect labels.
State-of-the-art Comparison on MS-COCO
We compare BalanceMix with several methods showing
the state-of-the-art performance with a ResNet backbone
on MS-COCO. The results are borrowed from Ridnik et
al. (Ridnik et al. 2023), and we follow exactly the same set-
ting in backbones, image resolution, and data augmentation.
BalanceMix is implemented on top of ML-Decoder for com-
parison. All backbones are pre-trained on ImageNet. The
compared methods are developed without consideration of
label noise, but we nd out that MS-COCO originally has
noisy labels (see Appendix H for examples).
Table 6 summarizes the best mAP on MS-COCO with-
out synthetic noise injection. For the
448

448
resolution,
BalanceMix improves the mAP by
0
:
3
Œ
0
:
5%
p
when using
ResNet-101 and TResNet-L. For the
640

640
resolution, its

improvement over ML-Decoder becomes
0
:
6%
p
when using
TResNet-L. The
91
:
7
mAP of BalanceMix with TResNet-
L is even higher than the 91.4mAP of ML-Decoder with
TResNet-XL.
Conclusion
We propose BalanceMix, which can handle imbalanced
labels and diverse types of label noise. The minority-
augmented mixing allows for adding sparse context in mi-
nority classes to majority classes without losing diversity.
The label-wise management realizes a robust way of ex-
ploiting noisy multi-labels without overtting. Through ex-
periments using real-world and synthetic noisy datasets, we
verify that BalanceMix outperforms state-of-the-art methods
in each setting of mislabeling, ipping, and missing labels,
with the co-existence of severe class imbalance. Overall, this
work will inspire subsequent studies to handle imbalanced
and noisy labels in a holistic manner.
Acknowledgements
This work was supported by Institute of Information &
Communications Technology Planning & Evaluation (IITP)
grant funded by the Korea government (MSIT) (No. 2020-
0-00862, DB4DL: High-Usability and Performance In-
Memory Distributed DBMS for Deep Learning). Addition-
ally, this work was partly supported by the FOUR Brain Ko-
rea 21 Program through the National Research Foundation
of Korea (NRF-5199990113928).
References
Arpit, D.; Jastrzebski, S.; Ballas, N.; Krueger, D.; Bengio,
E.; Kanwal, M. S.; Maharaj, T.; Fischer, A.; Courville, A.;
Bengio, Y.; et al. 2017. A closer look at memorization in
deep networks. In
ICML
, 233Œ242.
Bai, Y.; Yang, E.; Han, B.; Yang, Y.; Li, J.; Mao, Y.; Niu,
G.; and Liu, T. 2021. Understanding and improving early
stopping for learning with noisy labels. In
NeurIPS
, 24392Œ
24403.
Bello, M.; N
´
apoles, G.; Vanhoof, K.; and Bello, R. 2021.
Data quality measures based on granular computing for
multi-label classication.
Information Sciences
, 560: 51Œ67.
Ben-Baruch, E.; Ridnik, T.; Zamir, N.; Noy, A.; Friedman,
I.; Protter, M.; and Zelnik-Manor, L. 2021. Asymmetric loss
for multi-label classication. In
ICCV
, 82Œ91.
Chen, P.; Ye, J.; Chen, G.; Zhao, J.; and Heng, P.-A. 2021.
Beyond class-conditional assumption: A primary attempt to
combat instance-dependent label noise. In
AAAI
, 11442Œ
11450.
Cole, E.; Mac Aodha, O.; Lorieul, T.; Perona, P.; Morris, D.;
and Jojic, N. 2021 . Multi-label learning from single positive
labels. In
CVPR
, 933Œ942.
Cubuk, E. D.; Zoph, B.; Shlens, J.; and Le, Q. V. 2020. Ran-
daugment: Practical automated data augmentation with a re-
duced search space. In
CVPRW
, 702Œ703.
Ding, Y.; Zhou, T.; Zhang, C.; Luo, Y.; Tang, J.; and Gong,
C. 2022. Multi-class Label Noise Learning via Loss Decom-
position and Centroid Estimation. In
SDM
, 253Œ261.
Du, Y.; Shen, J.; Zhen, X.; and Snoek, C. G. 2023. Su-
perDisco: Super-Class Discovery Improves Visual Recog-
nition for the Long-Tail. In
CVPR
, 19944Œ19954.
Durand, T.; Mehrasa, N.; and Mori, G. 2019. Learning a
deep convnet for multi-label classication with partial la-
bels. In
CVPR
, 647Œ657.
Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.;
and Zisserman, A. 2010. The PASCAL visual object classes
(VOC) challenge.
International Journal of Computer Vision
,
88(2): 303Œ338.
Ferreira, B. Q.; Costeira, J. P.; and Gomes, J. P. 2021. Ex-
plainable Noisy Label Flipping for Multi-Label Fashion Im-
age Classication. In
CVPRW
, 3916Œ3920.
Galdran, A.; Carneiro, G.; and Gonz
´
alez Ballester, M. A.
2021. Balanced-mixup for highly imbalanced medical im-
age classication. In
MICCAI
, 323Œ333.
Guo, H.; and Wang, S. 2021. Long-tailed multi-label vi-
sual recognition by collaborative training on uniform and
re-balanced samplings. In
CVPR
, 15089Œ15098.
Han, B.; Yao, Q.; Yu, X.; Niu, G.; Xu, M.; Hu, W.; Tsang,
I.; and Sugiyama, M. 2018. Co-teaching: Robust training
of deep neural networks with extremely noisy labels. In
NeurIPS
, 8536Œ8546.
Hu, M.; Han, H.; Shan, S.; and Chen, X. 2018. Multi-label
learning from noisy labels with non-linear feature transfor-
mation. In
ACCV
, 404Œ419.
Huang, J.; Qu, L.; Jia, R.; and Zhao, B. 2019. O2u-net: A
simple noisy label detection approach for deep neural net-
works. In
ICCV
, 3326Œ3334.
Huynh, D.; and Elhamifar, E. 2020. Interactive multi-label
cnn learning with partial labels. In
CVPR
, 9423Œ9432.
Kim, Y.; Kim, J. M.; Akata, Z.; and Lee, J. 2022. Large Loss
Matters in Weakly Supervised Multi-Label Classication. In
CVPR
, 14156Œ14165.
Kim, Y.; Kim, J. M.; Jeong, J.; Schmid, C.; Akata, Z.; and
Lee, J. 2023. Bridging the Gap between Model Explanations
in Partially Annotated Multi-label Classication. In
CVPR
,
3408Œ3417.
Lanchantin, J.; Wang, T.; Ordonez, V.; and Qi, Y. 2021. Gen-
eral multi-label image classication with transformers. In
CVPR
, 16478Œ16488.
Li, J.; Socher, R.; and Hoi, S. C. 2020. DivideMix: Learning
with noisy labels as semi-supervised learning. In
ICLR
.
Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Doll
´
ar, P.
2017. Focal loss for dense object detection. In
CVPR
, 2980Œ
2988.
Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ra-
manan, D.; Doll
´
ar, P.; and Zitnick, C. L. 2014. Microsoft
COCO: Common objects in context. In
ECCV
, 740Œ755.
Liu, S.; Zhang, L.; Yang, X.; Su, H.; and Zhu, J. 2021.
Query2label: A simple transformer way to multi-label clas-
sication.
arXiv preprint arXiv:2107.10834
.
Liu, Z.; Luo, P.; Qiu, S.; Wang, X.; and Tang, X. 2016.
DeepFashion: Powering Robust Clothes Recognition and
Retrieval with Rich Annotations. In
CVPR
, 1096Œ1104.

Loyola-Gonz
´
alez, O.; Mart
´
nez-Trinidad, J. F.; Carrasco-
Ochoa, J. A.; and Garc
´
a-Borroto, M. 2016. Study of the im-
pact of resampling methods for contrast pattern based clas-
siers in imbalanced databases.
Neurocomputing
, 175: 935Œ
947.
Park, S.; Hong, Y.; Heo, B.; Yun, S.; and Choi, J. Y. 2022.
The Majority Can Help The Minority: Context-rich Minor-
ity Oversampling for Long-tailed Classication. In
CVPR
,
6887Œ6896.
Ridnik, T.; Sharir, G.; Ben-Cohen, A.; Ben-Baruch, E.; and
Noy, A. 2023. ML-Decoder: Scalable and versatile classi-
cation head. In
WACV
, 32Œ41.
Shikun, L.; Xiaobo, X.; Hansong, Z.; Yibing, Z.; Shim-
ing, G.; and Tongliang, L. 2022. Estimating Noise Tran-
sition Matrix with Label Correlations for Noisy Multi-Label
Learning. In
NeurIPS
.
Song, H.; Kim, M.; and Lee, J.-G. 2019. SELFIE: Refur-
bishing unclean samples for robust deep learning. In
ICML
,
5907Œ5915.
Song, H.; Kim, M.; Park, D.; Shin, Y.; and Lee, J.-G. 2022.
Learning from noisy labels with deep neural networks: A
survey.
IEEE TNNLS
.
Tarekegn, A. N.; Giacobini, M.; and Michalak, K. 2021. A
review of methods for imbalanced multi-label classication.
Pattern Recognition
, 118: 107965.
Trivedi, S. K.; Dey, S.; Kumar, A.; and Panda, T. K. 2017.
Handbook of research on advanced data mining techniques
and applications for business intelligence
. IGI Global.
Wang, Q.; Shen, B.; Wang, S.; Li, L.; and Si, L. 2014. Binary
codes embedding for fast image tagging with incomplete la-
bels. In
ECCV
, 425Œ439.
Wang, S.; Minku, L. L.; and Yao, X. 2014. Resampling-
based ensemble methods for online class imbalance learn-
ing.
IEEE Transactions on Knowledge and Data Engineer-
ing
, 27(5): 1356Œ1368.
Wei, Q.; Feng, L.; Sun, H.; Wang, R.; Guo, C.; and Yin,
Y. 2023. Fine-grained classication with noisy labels. In
CVPR
, 11651Œ11660.
Wei, T.; Shi, J.-X.; Tu, W.-W.; and Li, Y.-F. 2021. Ro-
bust long-tailed learning under label noise.
arXiv preprint
arXiv:2108.11569
.
Whang, S. E.; Roh, Y.; Song, H.; and Lee, J.-G. 2021.
Data Collection and Quality Challenges in Deep Learn-
ing: A Data-Centric AI Perspective.
arXiv preprint
arXiv:2112.06409
.
Wu, T.; Huang, Q.; Liu, Z.; Wang, Y.; and Lin, D. 2020.
Distribution-balanced loss for multi-label classication in
long-tailed datasets. In
ECCV
, 162Œ178.
Xia, X.; Liu, T.; Han, B.; Gong, M.; Yu, J.; Niu, G.; and
Sugiyama, M. 2022. Sample selection with uncertainty of
losses for learning with noisy labels. In
ICLR
.
Yuan, J.; Zhang, Y.; Shi, Z.; Geng, X.; Fan, J.; and Rui, Y.
2023. Balanced masking strategy for multi-label image clas-
sication.
Neurocomputing
, 522: 64Œ72.
Zhang, H.; Cisse, M.; Dauphin, Y. N.; and Lopez-Paz, D.
2018. Mixup: Beyond empirical risk minimization. In
ICLR
.
Zhang, M.; Hu, L.; Shi, C.; and Wang, X. 2020. Adversarial
label-ipping attack and defense for graph neural networks.
In
ICDM
, 791Œ800.
Zhang, S.; Li, Z.; Yan, S.; He, X.; and Sun, J. 2021. Distri-
bution alignment: A unied framework for long-tail visual
recognition. In
CVPR
, 2361Œ2370.
Zhao, W.; and Gomes, C. 2021. Evaluating multi-label clas-
siers with noisy labels.
arXiv preprint arXiv:2102.08427
.
References
Arpit, D.; Jastrzebski, S.; Ballas, N.; Krueger, D.; Bengio,
E.; Kanwal, M. S.; Maharaj, T.; Fischer, A.; Courville, A.;
Bengio, Y.; et al. 2017. A closer look at memorization in
deep networks. In
ICML
, 233Œ242.
Bai, Y.; Yang, E.; Han, B.; Yang, Y.; Li, J.; Mao, Y.; Niu,
G.; and Liu, T. 2021. Understanding and improving early
stopping for learning with noisy labels. In
NeurIPS
, 24392Œ
24403.
Bello, M.; N
´
apoles, G.; Vanhoof, K.; and Bello, R. 2021.
Data quality measures based on granular computing for
multi-label classication.
Information Sciences
, 560: 51Œ67.
Ben-Baruch, E.; Ridnik, T.; Zamir, N.; Noy, A.; Friedman,
I.; Protter, M.; and Zelnik-Manor, L. 2021. Asymmetric loss
for multi-label classication. In
ICCV
, 82Œ91.
Chen, P.; Ye, J.; Chen, G.; Zhao, J.; and Heng, P.-A. 2021.
Beyond class-conditional assumption: A primary attempt to
combat instance-dependent label noise. In
AAAI
, 11442Œ
11450.
Cole, E.; Mac Aodha, O.; Lorieul, T.; Perona, P.; Morris, D.;
and Jojic, N. 2021 . Multi-label learning from single positive
labels. In
CVPR
, 933Œ942.
Cubuk, E. D.; Zoph, B.; Shlens, J.; and Le, Q. V. 2020. Ran-
daugment: Practical automated data augmentation with a re-
duced search space. In
CVPRW
, 702Œ703.
Ding, Y.; Zhou, T.; Zhang, C.; Luo, Y.; Tang, J.; and Gong,
C. 2022. Multi-class Label Noise Learning via Loss Decom-
position and Centroid Estimation. In
SDM
, 253Œ261.
Du, Y.; Shen, J.; Zhen, X.; and Snoek, C. G. 2023. Su-
perDisco: Super-Class Discovery Improves Visual Recog-
nition for the Long-Tail. In
CVPR
, 19944Œ19954.
Durand, T.; Mehrasa, N.; and Mori, G. 2019. Learning a
deep convnet for multi-label classication with partial la-
bels. In
CVPR
, 647Œ657.
Everingham, M.; Van Gool, L.; Williams, C. K.; Winn, J.;
and Zisserman, A. 2010. The PASCAL visual object classes
(VOC) challenge.
International Journal of Computer Vision
,
88(2): 303Œ338.
Ferreira, B. Q.; Costeira, J. P.; and Gomes, J. P. 2021. Ex-
plainable Noisy Label Flipping for Multi-Label Fashion Im-
age Classication. In
CVPRW
, 3916Œ3920.
Galdran, A.; Carneiro, G.; and Gonz
´
alez Ballester, M. A.
2021. Balanced-mixup for highly imbalanced medical im-
age classication. In
MICCAI
, 323Œ333.
Guo, H.; and Wang, S. 2021. Long-tailed multi-label vi-
sual recognition by collaborative training on uniform and
re-balanced samplings. In
CVPR
, 15089Œ15098.

Han, B.; Yao, Q.; Yu, X.; Niu, G.; Xu, M.; Hu, W.; Tsang,
I.; and Sugiyama, M. 2018. Co-teaching: Robust training
of deep neural networks with extremely noisy labels. In
NeurIPS
, 8536Œ8546.
Hu, M.; Han, H.; Shan, S.; and Chen, X. 2018. Multi-label
learning from noisy labels with non-linear feature transfor-
mation. In
ACCV
, 404Œ419.
Huang, J.; Qu, L.; Jia, R.; and Zhao, B. 2019. O2u-net: A
simple noisy label detection approach for deep neural net-
works. In
ICCV
, 3326Œ3334.
Huynh, D.; and Elhamifar, E. 2020. Interactive multi-label
cnn learning with partial labels. In
CVPR
, 9423Œ9432.
Kim, Y.; Kim, J. M.; Akata, Z.; and Lee, J. 2022. Large Loss
Matters in Weakly Supervised Multi-Label Classication. In
CVPR
, 14156Œ14165.
Kim, Y.; Kim, J. M.; Jeong, J.; Schmid, C.; Akata, Z.; and
Lee, J. 2023. Bridging the Gap between Model Explanations
in Partially Annotated Multi-label Classication. In
CVPR
,
3408Œ3417.
Lanchantin, J.; Wang, T.; Ordonez, V.; and Qi, Y. 2021. Gen-
eral multi-label image classication with transformers. In
CVPR
, 16478Œ16488.
Li, J.; Socher, R.; and Hoi, S. C. 2020. DivideMix: Learning
with noisy labels as semi-supervised learning. In
ICLR
.
Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Doll
´
ar, P.
2017. Focal loss for dense object detection. In
CVPR
, 2980Œ
2988.
Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ra-
manan, D.; Doll
´
ar, P.; and Zitnick, C. L. 2014. Microsoft
COCO: Common objects in context. In
ECCV
, 740Œ755.
Liu, S.; Zhang, L.; Yang, X.; Su, H.; and Zhu, J. 2021.
Query2label: A simple transformer way to multi-label clas-
sication.
arXiv preprint arXiv:2107.10834
.
Liu, Z.; Luo, P.; Qiu, S.; Wang, X.; and Tang, X. 2016.
DeepFashion: Powering Robust Clothes Recognition and
Retrieval with Rich Annotations. In
CVPR
, 1096Œ1104.
Loyola-Gonz
´
alez, O.; Mart
´
nez-Trinidad, J. F.; Carrasco-
Ochoa, J. A.; and Garc
´
a-Borroto, M. 2016. Study of the im-
pact of resampling methods for contrast pattern based clas-
siers in imbalanced databases.
Neurocomputing
, 175: 935Œ
947.
Park, S.; Hong, Y.; Heo, B.; Yun, S.; and Choi, J. Y. 2022.
The Majority Can Help The Minority: Context-rich Minor-
ity Oversampling for Long-tailed Classication. In
CVPR
,
6887Œ6896.
Ridnik, T.; Sharir, G.; Ben-Cohen, A.; Ben-Baruch, E.; and
Noy, A. 2023. ML-Decoder: Scalable and versatile classi-
cation head. In
WACV
, 32Œ41.
Shikun, L.; Xiaobo, X.; Hansong, Z.; Yibing, Z.; Shim-
ing, G.; and Tongliang, L. 2022. Estimating Noise Tran-
sition Matrix with Label Correlations for Noisy Multi-Label
Learning. In
NeurIPS
.
Song, H.; Kim, M.; and Lee, J.-G. 2019. SELFIE: Refur-
bishing unclean samples for robust deep learning. In
ICML
,
5907Œ5915.
Song, H.; Kim, M.; Park, D.; Shin, Y.; and Lee, J.-G. 2022.
Learning from noisy labels with deep neural networks: A
survey.
IEEE TNNLS
.
Tarekegn, A. N.; Giacobini, M.; and Michalak, K. 2021. A
review of methods for imbalanced multi-label classication.
Pattern Recognition
, 118: 107965.
Trivedi, S. K.; Dey, S.; Kumar, A.; and Panda, T. K. 2017.
Handbook of research on advanced data mining techniques
and applications for business intelligence
. IGI Global.
Wang, Q.; Shen, B.; Wang, S.; Li, L.; and Si, L. 2014. Binary
codes embedding for fast image tagging with incomplete la-
bels. In
ECCV
, 425Œ439.
Wang, S.; Minku, L. L.; and Yao, X. 2014. Resampling-
based ensemble methods for online class imbalance learn-
ing.
IEEE Transactions on Knowledge and Data Engineer-
ing
, 27(5): 1356Œ1368.
Wei, Q.; Feng, L.; Sun, H.; Wang, R.; Guo, C.; and Yin,
Y. 2023. Fine-grained classication with noisy labels. In
CVPR
, 11651Œ11660.
Wei, T.; Shi, J.-X.; Tu, W.-W.; and Li, Y.-F. 2021. Ro-
bust long-tailed learning under label noise.
arXiv preprint
arXiv:2108.11569
.
Whang, S. E.; Roh, Y.; Song, H.; and Lee, J.-G. 2021.
Data Collection and Quality Challenges in Deep Learn-
ing: A Data-Centric AI Perspective.
arXiv preprint
arXiv:2112.06409
.
Wu, T.; Huang, Q.; Liu, Z.; Wang, Y.; and Lin, D. 2020.
Distribution-balanced loss for multi-label classication in
long-tailed datasets. In
ECCV
, 162Œ178.
Xia, X.; Liu, T.; Han, B.; Gong, M.; Yu, J.; Niu, G.; and
Sugiyama, M. 2022. Sample selection with uncertainty of
losses for learning with noisy labels. In
ICLR
.
Yuan, J.; Zhang, Y.; Shi, Z.; Geng, X.; Fan, J.; and Rui, Y.
2023. Balanced masking strategy for multi-label image clas-
sication.
Neurocomputing
, 522: 64Œ72.
Zhang, H.; Cisse, M.; Dauphin, Y. N.; and Lopez-Paz, D.
2018. Mixup: Beyond empirical risk minimization. In
ICLR
.
Zhang, M.; Hu, L.; Shi, C.; and Wang, X. 2020. Adversarial
label-ipping attack and defense for graph neural networks.
In
ICDM
, 791Œ800.
Zhang, S.; Li, Z.; Yan, S.; He, X.; and Sun, J. 2021. Distri-
bution alignment: A unied framework for long-tail visual
recognition. In
CVPR
, 2361Œ2370.
Zhao, W.; and Gomes, C. 2021. Evaluating multi-label clas-
siers with noisy labels.
arXiv preprint arXiv:2102.08427
.

A. Pseudocode
The overall procedure of BalanceMix is described in Algo-
rithm 1, which is simple and self-explanatory. During the
warm-up phase, it updates the model with a standard ap-
proach using the BCE loss on the minority-augmented mini-
batch. After the warm-up phase, ne-grained label-wise
management is performed before generating the minority-
augmented mini-batch; in detail, all labels are processed
and categorized into clean, re-labeled, and ambiguous ones.
Next, the two mini-batches are mixed by Eq. (5) with the
rened labels. Then, the model is updated by the proposed
loss function in Eq. (10).
Algorithm 1: BalanceMix
I
N P U T
:
~
D
: noisy data,
b
: batch size,
epoch
: training epochs
w ar m
: warm-up epochs,

: mixup coefcient,

: re-
labeling threshold
O
U T P U T
:

t
: DNN parameters
1:

 
Initialize DNN parameters;
2:
for
i
= 1
to
epoch
do
3:
for
j
= 1
to
j
~
D j
=b
do
4:
/* Sampling from two samplers */
5:
Draw a mini-batch
B
R
by the random sampler;
6:
Draw a mini-batch
B
M
by the minority sampler;
7:
if
i

w ar m
then
8:
/* Update with given labels */
9:
Generate a mini-batch
B
mix
by Eq. (5);
10:
Update the model by Eq. (1);
11:
else
12:
/* Update with label management */
13:
Perform the label-wise management;
14:
Generate a mini-batch
B
mix
by Eq. (5);
15:
Update the model by Eq. (10);
16:
/* Update the minority sampler and GMMs */
17:
Update the sampling probability by Eq. (4);
18:
Fitting the GMMs to the loss of entire data;
19:
return

B. Analysis of Minority Sampling
We analyze the correlation between average precision and
prediction condence in the presence of noisy labels. Fig-
ure 3 was obtained without label noise. Figure 5 is obtained
with
label noise. The Perason correlation coefcient is still
very high, though the absolute values of the condence and
precision are decreased owing to label noise. The coefcient
was calculated between ten class groups.
In addition, we show how the sampling probability
changes with our condence-based minority oversampling
method in Figure 6. Minority instances are initially oversam-
pled with a high probability, but the degree of oversampling
gradually decreases as the imbalance problem gets resolved.
C. Imbalance in Benchmark Datasets
We investigate the imbalance of positive labels across
classes in three benchmark datasets, Pascal-VOC
3
, MS-
3
http://host.robots.ox.ac.uk/pascal/VOC/
Figure 5: Prediction condence (left) and average precision
(right) in COCO with mislabeling of
40%
at the 40
%
of
training epochs.
Figure 6: Sampling probability over the training period.
COCO
4
, and DeepFashion
5
. We use a ne-grained subset of
DeepFashion with 16,000 training and 4,000 validation in-
stances as well as multi-labels of
26
attribute classes, which
are provided by the authors. Fig. 7 shows the distribution
of the numbers of positive labels across classes, where the
dashed lines split the classes into many-shot
[10000
;
1
)
,
medium-shot
[1000
;
10000)
, and few-shot
[0
;
1000)
classes;
Pascal-VOC does not have the many-shot classes.
A few majority classes occupy most of the positive labels
in the data. Hence, we dene the
class imbalance ratio
fol-
lowing the literature (Zhang et al. 2021; Park et al. 2022),
CLS
Imb
:
= max
1

i

k
N
i
=
min
1

i

k
N
i
;
(12)
which is the maximum ratio of the number of positive labels
in the majority class to that in the minority class. In addition,
an image contains few positive labels but many negative la-
bels. Hence, we dene the
positive-negative ratio
by
PN
Imb
:
=
X
1

i

k
N
0
i
=
X
1

i

k
N
i
;
(13)
where
N
0
i
is the number of negative labels for the
i
-th class.
As for these two imbalance ratios, Pascal-VOC, MS-COCO,
and DeepFashion have class imbalance ratios of
14
,
339
, and
239
, and positive-negative imbalance ratios of
13
,
27
, and
3
,
respectively.
D. Detailed Experiment Conguration
All the algorithms are implemented using Pytorch 21.11 and
run using two NVIDIA V100 GPUs utilizing distributed
data parallelism. We ne-tune ResNet-50 pre-trained on
ImageNet-1K for 20, 50, and 40 epochs for Pascal-VOC (a
batch size of
32
), MS-COCO (a batch size of
64
), and Deep-
Fashion (a batch size of
64
) using an SGD optimizer with
a momentum of 0.9 and a weight decay of
10

4
. All the
4
https://cocodataset.org/
5
https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

Figure 7: Imbalanced labels of the three benchmark datasets.

Mis.
20%
Mis.
40%

Missing Label
1.000
82.5 77.8
0.550
77.4
0.975
84.3
81.4
0.600
77.3
0.950
84.0
81.5
0.700
76.8
0.900
83.7 81.3
0.900
76.3
Table 7: Parameter search for

when xing

= 4
:
0
.

All
Many Medium Few
1.0
80.3
84.8 81.2 63.2
2.0
81.1
85.6
82.3 60.3
4.0
81.6
85.5
82.6 63.1
8.0
81.3
85.1 82.5 60.1
Table 8: Parameter search for

when xing

= 0
:
975
.
Class Group
MS-COCO
Pascal-VOC
Category
Method
Mis. 20% Mis. 40% Rand. 20% Rand. 40% Single
Mis. 20% Mis. 40% Rand. 20% Rand. 40% Single
Default
BCE Loss
73.1

0.4 63.8

0.8 59.8

0.5 43.5

0.8 69.7

1.8
82.9

0.1 75.4

0.3 76.3

0.5 72.0

5.5 85.7

0.1
Noisy Labels
Co-teaching
82.5

0.1 78.6

0.4 65.5

0.0 43.6

0.1 68.1

1.4
92.5

0.3 90.9

0.1 82.6

0.4 70.0

0.4 81.9

1.6
ASL
82.8

0.1 80.3

0.2 75.0

0.1 66.2

0.1 73.3

0.2
91.2

0.0 86.4

0.1 90.1

0.1 74.8

0.3 86.8

0.1
Missing Labels
LL-R
80.7

0.1 75.3

0.2 74.0

0.2 69.3

0.2 74.2

0.2
87.5

0.5 83.1

1.2 85.5

0.3 78.6

2.2 89.1

0.4
LL-Ct
81.3

0.0 77.4

0.1 73.2

0.3 70.1

0.1 76.9

0.1
88.8

0.5 84.8

1.4 87.1

0.2 78.8

0.3 89.3

0.1
Proposed
BalanceMix
84.3

0.1 81.6

0.3 76.5

0.1 74.5

0.1 77.4

0.1
92.9

0.1 92.0

0.0 91.2

0.1 84.4

0.1 92.6

0.1
Table 9: Last mAPs on MS-COCO and Pascal-VOC with standard errors.
images are resized with
448

448
resolution. The initial
learning rate is set to be
0
:
01
and decayed with a cosine
annealing without restart. The number of warm-up epochs
is set to be 5, 10, and 5 for the three datasets, respec-
tively. We adopt a state-of-the-art Transformer-based de-
coder (Lanchantin et al. 2021; Liu et al. 2021; Ridnik et al.
2023) for the classication head. These experiment setups
are exactly the same for all compared methods.
The hyperparameters for the compared methods are con-
gured favorably, as suggested in the original papers.
Ł
 Co-teaching (Han et al. 2018): We extend the vanilla ver-
sion to support multi-label classication. Two models are
maintained for co-training. Instead of using the known
noise ratio, we t a bi-modal univariate GMM to the
losses of all instances, i.e., instance-level modeling. Then,
the instances whose probability of being clean is greater
than
0
:
5
are selected as clean instances.
Ł
 ASL (Ben-Baruch et al. 2021): Three hyperparameters Œ

+
which is a down-weighting coefcient for positive la-
bels,


which is a down-weighting coefcient for nega-
tive labels, and
m
which is a probability margin Œ are set
to be
0
:
0
,
4
:
0
, and
0
:
05
, respectively.
Ł
 LL-R & LL-Ct (Kim et al. 2022): The only hyperparame-
ter is

r el
, which determines the speed of increasing the
rejection (or correction) rate. The default value used in
the original paper was
0
:
2
for 10 epochs. Hence, we mod-
ify the value according to our training epochs, such that
the rejection (or correction) ratios at the nal epoch are
the same. Specically, it is set to be
0
:
1
for
20
epochs
(Pascal-VOC),
0
:
04
for
50
epochs (MS-COCO), and
0
:
05
for
40
epochs (DeepFashion), respectively.
Regarding the state-of-the-art comparison with ResNet-
101 and TResNet-L, we follow exactly the same settings
in the backbone, image resolution, and data augmentation
(Ridnik et al. 2023). See Table 5 for details.
E. Hyperparameters
BalanceMix introduces two hyperparameters:

, a con-
dence threshold for re-labeling and

, the parameter of the
beta distribution for Mixup. We search for a suitable pair of
these two hyperparameters based on MS-COCO.
First, we x

= 4
:
0
and conduct a grid search to nd the
best

, as summarized in Table 7. Intuitively, a high thresh-
old value achieves high precision in re-labeling, while a low
threshold value achieves high recall. For mislabeling, high
precision is more benecial than high recall; thus, the in-
terval of
0
:
950
Œ
0
:
975
exhibits the best mAP. However, in
the missing (single positive) label setup, high recall precedes
high precision because increasing the amount of positive la-
bels is more benecial; thus, the interval of
0
:
550
Œ
0
:
600
ex-
hibits the best mAP. Overall, we use

= 0
:
550
for the miss-
ing label setup, while

= 0
:
975
for other setups.
Second, we x

= 0
:
975
and repeat a grid search for
the best

. Table 8 summarizes the mAPs on MS-COCO

Class Group
All
Medium-shot
Few-shot
Category
Method
0% 20% 40%
0% 20% 40%
0% 20% 40%
Default
BCE Loss
87.7 82.9 75.4
94.1 85.3 77.1
84.9 81.8 74.8
Noisy Labels
Co-teaching
90.8 92.5
90.9
93.9
94.0
92.8
89.4 91.8
90.0
ASL
91.4
91.2 86.4
92.9 92.4 80.5
90.8
90.7 88.9
T-estimator
91.0 89.5 89.0
92.4 91.9 91.6
90.2 88.5 87.9
Missing Labels
LL-R
81.8 87.5 83.1
92.5 89.6 85.7
77.2 86.6 81.9
LL-Ct
84.0 88.8 84.8
90.3 90.7 87.9
81.3 87.9 83.5
Proposed
BalanceMix
93.3 92.9 92.0
95.1 94.4 93.6
92.5 92.2 91.3
Table 10: Last mAPs on Pascal-VOC with
mislabeling
of
0
Œ
40%
. The 1st and 2nd best values are in bold and underlined.
Class Group
All
Medium-shot
Few-shot
Category
Method
0% 20% 40%
0% 20% 40%
0% 20% 40%
Default
BCE Loss
87.7 76.3 72.0
94.1 80.0 74.4
84.9 74.8 71.0
Noisy Labels
Co-teaching
90.8 82.6 70.3
93.9
87.1 82.8
89.4 80.6 64.9
ASL
91.4
90.1
74.8
92.9 92.3
87.6
90.8
89.2
69.3
T-estimator
91.0 85.9 70.1
92.4 89.3 80.3
90.2 84.4 65.6
Missing Labels
LL-R
81.8 85.5 78.6
92.5 88.8 83.0
77.2 84.0 76.8
LL-Ct
84.0 87.1 78.8
90.3 89.3 82.9
81.3 86.1 77.1
Proposed
BalanceMix
93.3 91.2 84.4
95.1 93.8 91.7
92.5 90.0 81.3
Table 11: Last mAPs on Pascal-VOC with
random ipping
of
0
Œ
40%
. The 1st and 2nd best values are in bold and underlined.
Mixing
coef.

0.0 1.0 2.0 4.0 8.0
Many-shot
85.4 85.6 85.1 85.5 84.8
Few-shot
57.4 60.3 62.4 63.1 63.2
Table 12: Varying

on COCO with mislabeling of
40%
.
with a mislabeling ratio of
40%
. The best mAP for many-
shot classes is observed when

= 2
:
0
. However, the overall
mAP of BalanceMix is the best when

= 4
:
0
owing to the
highest mAP on medium-shot and few-shot classes. There-
fore, we use

= 4
:
0
in general.
These hyperparameter values found may not be optimal
as we validate them only in a few experiment settings, but
BalanceMix shows satisfactory performance with them in
all the experiments presented in the paper. We believe that
the performance of BalanceMix could be further improved
via a more sophisticated parameter search.
F. Additional Main Results
Results with Standard Errors
Table 9 summarizes the last mAPs on MS-COCO and
Pascal-VOC. We repeat the experiments thrice and report
the averaged mAPs as well as their standard errors. These
standard errors are, in general, very small.
Results on Pascal-VOC
Tables 10 and 11 summarize the mAPs on Pascal-VOC with
mislabeing and random ipping. The performance trends
are similar to those on MS-COCO except that Co-teaching
exhibits higher mAPs than ASL in the mislabeling noise.
In Pascal-VOC unlike MS-COCO, the number of positive
labels per instance is only two on average. Therefore, the
Method
Clean Mis. 40% Rand. 40% Missing
Co-teaching
82.8 78.6 43.6 68.1
T-estimator
84.3 80.5 69.9 16.8
LL-R
82.5 75.3 69.3 74.2
LL-Ct
79.4 77.4 70.1 76.9
BalanceMix (wo Min.)
85.0 80.9 73.0 77.0
Table 13: Analysis of BalanceMix w.o using the minority
sampler on MS-COCO.
instance-level selection of Co-teaching can perform better
than ASL. However, in the random ipping noise where
even negative labels are ipped by a given noise ratio, Co-
teaching is much worse than ASL. BalanceMix consistently
exhibits the best mAPs for all class categories. Regarding
T-estimator, it performs much better than BCE Loss and
exhibits comparable performance to Co-teaching and ASL,
even if 10% of training data is not used for training since it
is required for the noisy validation set.
G. Analysis of Label-wise Management
G.1. Mixing with Different Diversity
The diversity is added by mixing the instances from the ran-
dom sampler with the instances from the minority sampler
via Mixu p. Thus, when the Mixup coefcient

is
0
, mixing
is not performed at all, and the diversity is the lowest. On
the other hand, as

becomes larger, minority samples are
more strongly mixed with random samples, and the diversity
gets higher. As shown in Table 12, increasing the value of

enhances few-shot class performance, but exc
```
