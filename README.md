# Cloud-Project
Implementing Neural network on amazon sagemaker
Introduction:
The Gaming Industry’s growth is
tremendous over the past few years,
with over half a billion new gamers
in three years. Since people are
restricted to stay at home, the
number keeps growing as the social
aspects of gamers are also
improving over time. However, the
gaming experience business is still
in progress. Many companies came

and built excellent gaming
platforms and products. However, we
have experienced that there is
still room for improvement in the
gaming experience. Some level of
interactive player control in which
VR games are making progress.
Currently, there is still a lack of
VR technologies regarding how
realistic the VR game feels. VR
technologies can completely change
how our society interacts, and
businesses interact with customers.
Much improvement is needed, and the
first step should be making a
realistic-looking VR. This
generation of more realistic-
looking VR and gaming, in general,
can be achieved using the
Generative Adversarial Network
algorithm, also known as GAN.
Generation of more realistic images
is the first step for this process.
Rather than building a model for
whole VR technologies or games. In
GAN, the generator network takes a
sample and generates a sample of
data. After this, the discriminator
network decides whether the data is
generated or taken from the actual
sample using a binary
classification problem with a
sigmoid function. The generative
model analysis the distribution of

the data in such a way that after
the training phase, the probability
of the discriminator making a
mistake maximizes, and the
discriminator is based on a model
that will estimate the probability
that the sample is coming from the
real data or not the generator.
This way, the generator will slowly
develop to generate realistic data,
which can be used to develop or
generate real-looking images for
various purposes. My aim is to use
this GAN to help build a realistic-
looking VR output but first we will
be observing on game scenes. 1
Generative Adversarial Networks, or
GANs for short, are an approach to
generative modelling using deep
learning methods, such as
convolutional neural networks.
Generative modelling is an
unsupervised learning task in
machine learning that involves
automatically discovering and
learning the regularities or
patterns in input data in such a
way that the model can be used to
generate or output new examples
that plausibly could have been
drawn from the original dataset.
GANs are a clever way of training a
generative model by framing the
problem as a supervised learning

problem with two sub-models: the
generator model that we train to
generate new examples, and the
discriminator model that tries to
classify examples as either real
(from the domain) or fake
(generated). The two models are
trained together in a zero-sum
game, adversarial, until the
discriminator model is fooled about
half the time, meaning the
generator model is generating
plausible examples. GANs are an
exciting and rapidly changing
field, delivering on the promise of
generative models in their ability
to generate realistic examples
across a range of problem domains,
most notably in image-to-image
translation tasks such as
translating photos of summer to
winter or day to night, and in
generating photorealistic photos of
objects, scenes, and people that
even humans cannot tell are fake.
The generator of the DCGAN
architecture takes 100 uniform
generated values using normal
distribution as an input. First, it
changes the dimension to 4x4x1024
and performed a fractionally
strided convolution in 4 times with
stride of 1/2 (this means every
time when applied, it doubles the

image dimension while reducing the
number of output channels). The
generated output has dimensions of
(64, 64, 3). There are some
architectural changes proposed in
generator such as removal of all
fully connected layer, use of Batch
Normalization which helps in
stabilizing training. In this
paper, the authors use ReLU
activation function in all layers
of generator, except for the output
layers. We will be implementing
generator with similar guidelines
but not completely same
architecture. The role of the
discriminator here is to determine
that the image comes from either
real dataset or generator. The
discriminator can be simply
designed similar to a convolution
neural network that performs a
image classification task. However,
the authors of this paper suggested
some changes in the discriminator
architecture. Instead of fully
connected layers, they used only
strided-convolutions with Leaky
ReLU as activation function, the
input of the generator is a single
image from dataset or generated
image and the output is a score
that determines the image is real
or generated.

Needs and motivation:
The Gaming Industry’s growth is
tremendous over the past few years,
with over half a billion new gamers
in three years. Since people are
restricted to stay at home, the
number keeps growing as the social
aspects of gamers are also
improving over time. However, the
gaming experience business is still
in progress. Many companies came
and built excellent gaming
platforms and products. However, we
have experienced that there is
still room for improvement in the
gaming experience. Some level of
interactive player control in which
VR games are making progress.
Currently, there is still a lack of
VR technologies regarding how
realistic the VR game feels. VR
technologies can completely change
how our society interacts, and
businesses interact with customers.
Much improvement is needed, and the
first step should be making a
realistic-looking VR. This
generation of more realistic-
looking VR and gaming, in general,
can be achieved using the
Generative Adversarial Network
algorithm, also known as GAN.

Generation of more realistic images
is the first step for this process.
Rather than building a model for
whole VR technologies or games. In
GAN, the generator network takes a
sample and generates a sample of
data. After this, the discriminator
network decides whether the data is
generated or taken from the actual
sample using a binary
classification problem with a
sigmoid function. The generative
model analyses the distribution of
the data in such a way that after
the training phase, the probability
of the discriminator making a
mistake maximizes, and the
discriminator is based on a model
that will estimate the probability
that the sample is coming from the
real data or not the generator.
This way, the generator will slowly
develop to generate realistic data,
which can be used to develop or
generate real-looking images for
various purposes. My aim is to use
this GAN to help build a realistic-
looking VR output but first we will
be observing on game scenes.
Objectives :
GAN Working to get something from
the generator, we must input

something i.e., ‘z’. This ‘z’ is a
noise (no information) and after
passing this ‘z’ to model ‘G’, it
will produce G(z). Let Pdata(x)
represents probability distribution
of original data, Pz(z) represents
the distribution of the noise, and
Pg(x) is the distribution function
for the output of Generator. Next
we will be passing the
reconstructed data and the original
data to the discriminator model
which will 3 give the probability
of input belonging to the original
data. This way generator will try
to fool the discriminator. Without
making things complicated, we will
discuss about an algorithm that
will be used to improve the image.
Rather than directly working with
VR games which can bring up lot of
issue, we are going to work on
making the images look more
realistic and better. Image
enhancement using GAN to give a
better-looking photo than original.
SRGAN is called super resolution
generative adversarial network
which can enhance the images with
low resolution, but unlike other
solution, SRGAN also helps to
recover the texture detail. Other
methods usually check for similar
pixel in the image whose result is

not too satisfactory output.
Therefore, SRGAN uses perceptual
loss function and residual blocks.
We formulate the perceptual loss as
the weighted sum of a content loss
and an adversarial loss component.
Content loss is pixel wise MSE
loss, along with content loss,
adversarial loss component is also
added. The generative loss is
defined based on the probabilities
of the discriminator over all
training samples. Few of the pre-
processing that has to be done is
images cropping. All the input
images can have a same dimension as
same width and height is a good
way. After this, we need to prepare
the low-resolution images of data.
Last but not the least,
normalization of images. Content
loss is computed once the generator
will give output. It compares the
output of first convolution of VGG.
In SRGAN they have used both the
content loss and Adversarial loss.
How to implement Generative
Adversarial Network using Amazon
Sage Maker:
We know that Generative Adversarial
Network deals with extreme CPU
specifications for running neural

network model solely based on
generators and the discriminators
but, due to budget restrictions for
a lot of student population this
could be a big challenge hence to
tackle this kind of problem we used
a virtualised environment such as
Amazon sage maker which creates a
virtualised CPU board for
implementation. Amazon Sage maker
is an amazing tool which helps in
designing a virtualised environment
which helps you to choose from
various options such as a CPU to a
virtualised RAM and V-RAM and even
helps in making a GPU which helps
to run the tensor-flow aspects of
the model with extreme smoothness
now the main idea it to implement a
butter-flow running model of a
simple GAN which will have a single
discriminator and single generator.
Why Amazon Sage Maker?
It is a fully managed machine
learning service which helps data-
scientists and developers to
quickly train machine-learning
models and then directly deploy
them into a production ready
environment, this is done using a
Jupyter authorised notebooks for
easy access to various data-sources

for exploration and analysis so
don’t need to manage a lot of
servers in one go. It also gives a
native-support to bring your own
algorithms and frameworks by
flexible distribution for training
options that adjust to your
specific workflow. This helps data-
scientists in creating an easy
approach to a varied level of
specifications with ease and helps
making their lives easy.
Amazon Sage Maker features:
Studio:
An integrated machine learning
environment this helps in training,
deployment, building and analysis
of your machine learning models all
done in the same application with
ease.
Model registry:
Versioning, artifact and lineage
tracking, approval workflow and
cross counter deployment of your
machine learning models with ease.
Projects:

This helps in creating end to end
machine learning solutions with
CI/CD by using Sage Maker projects
for easy implementations.
ML lineage tracking:
One of the most important features
when it comes to workflows in the
models, this helps in tracking
lineage of the total workflows in
machine learning environment.
Data Wrangler:
This helps in importing, analysing,
preparing, creating and featuring
data in sage maker studio. It also
helps in integrating data wrangler
into your machine learning
workflows to simplify and
streamline data for pre-processing
the feature engineering using
almost little to no amount of
coding. Here, you can also add your
own python scripts and
transformations to customise your
data prep-workflow.
Feature Store:
A centralized store for features
and associated metadata so features
can be easily discovered and

reused. You can create two types of
stores, an Online or Offline store.
The Online Store can be used for
low latency, real-time inference
use cases and the Offline Store can
be used for training and batch
inference.
Jump start:
Learn about Sage Maker features and
capabilities through curated 1-
click solutions, example notebooks,
and pretrained models that you can
deploy. You can also fine-tune the
models and deploy them.
Clarify: Improve your machine
learning models by detecting
potential bias and help explain the
predictions that models make.
Edge manager: Optimize custom
models for edge devices, create and
manage fleets and run models with
an efficient runtime.
Ground Truth:
High-quality training datasets by
using workers along with machine
learning to create labelled
datasets.

Augmented AI:
Build the workflows required for
human review of ML predictions.
Amazon A2I brings human review to
all developers, removing the
undifferentiated heavy lifting
associated with building human
review systems or managing large
numbers of human reviewers.

Studio Notebooks:
The next generation of Sage Maker
notebooks that include AWS Single
Sign-On (AWS SSO) integration, fast
start-up times, and single-click
sharing.
Experiments:
Experiment management and tracking.
You can use the tracked data to
reconstruct an experiment,
incrementally build on experiments
conducted by peers, and trace model
lineage for compliance and audit
verifications.
Debugger:
Inspect training parameters and
data throughout the training
process. Automatically detect and
alert users to commonly occurring

errors such as parameter values
getting too large or small.

Autopilot:
Users without machine learning
knowledge can quickly build
classification and regression
models.
Model Monitor:
Monitor and analyse models in
production (endpoints) to detect
data drift and deviations in model
quality.
Neo:
Train machine learning models once,
then run anywhere in the cloud and
at the edge.
Elastic Inference:
Speed up the throughput and
decrease the latency of getting
real-time inferences.
Reinforcement learning:
Maximize the long-term reward that
an agent receives as a result of
its actions.
Pre-processing:

Analyse and pre-process data,
tackle feature engineering, and
evaluate models.
Batch Transform:
Pre-process datasets, run inference
when you don&#39;t need a persistent
endpoint, and associate input
records with inferences to assist
the interpretation of results.
