# Sure Start VAIL 2021 Program
A repository for the SureStart VAIL 2021 Program that I am completing as an AI and Machine Learning trainee. 
Program includes techincal skill training on:
- Machine learning
- Natural language processing
- Computer vision
- Data analytics & visualization
- Ethical AI development

## Responses

### Day 5: (02/12/21): What are Neural Networks (NN)?

Materials for [day 5](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%205%20-%20What%20are%20Neural%20Networks) can be found here.

Today I reviewed the [guide](https://serokell.io/blog/deep-learning-and-neural-network-guide) from yesterday on the common components of Neural Networks, and how they work with different ML functions and algorithms, (i.e. Neuron, Weights, Bias, Functions, Sigmoid, Softmax, Input vs Output, etc.)

Using this [database](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection), I chose one a creators code to try to follow, [Saai Sudarsanan](https://www.kaggle.com/saaisudarsanand/sarcasm-detection-using-tf-keras). 

I created a Kaggle Notebook that had a 92% accuracy rate and I tried to output his code using TF Keras. The model includes examples of the sigmoid and relu activation functions. Lines 57 and 60 contain errors that I was unable to fix, that evaluate the accuracy of the model, however, the rest functioned well.

### Day 4: (02/11/21): What is Deep Learning?

Today, I had to think about a real-world problem and see if I could find a dataset that has the characteristics of the data of that problem. Then, I had to think about the deep learning algorithm that I would likely use to develop a solution to it. 

There have been a lot of developments made in the technology and algorithms used to to track cells. Specifically, this is relevant in neurodegenerative diseases, such as Alzheimer's Disease. A deep learning machine that could track and or categorize abnormal cells and/or proteins could help to diagnose the disease earlier on and would lead to faster treatment. 

Data set: OASIS-1: [Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults](https://www.oasis-brains.org)

This data set consists of a cross-sectional collection of T1-weighted MRI scans that include subjects with no AD, very mild AD and moderate AD. A classification approach using convuolutional neural networks could be used to classify these images quickly into levels of AD onset. I would choose this approach because CNNs have the ability to process images in an efficient manner.

[Difference between AD brain and normal brain](https://core4.bmctoday.net/storage/images/1559306926-0619_CF3_Fig2.png) and the [Source](https://practicalneurology.com/articles/2019-june/brain-imaging-in-differential-diagnosis-of-dementia)

Classification would depend on sizes and shapes of neuronal loss in the brain. Shades of the pixels would also need to be considered where darker shades would represent the neuronal loss as there is no activity in those parts of the brain. It would then classify the images as no AD, mild AD and severe AD. Then an NN algorithm would be implemented, I have also read that an [ELM](https://ieeexplore.ieee.org/document/6583856) (Extreme Learning Machine) would be effective in doing this faster.

### Day 3: (02/10/21): Introduction to ML and TensorFlow

Materials for day 3 can be found [here]

1. What are “Tensors” and what are they used for in Machine Learning? 
TensorFlow is an open source library created by Google for numerical computation and machine learning. It uses ML and deep learning, such as neural networking models and algorithms. It is inspired by the way that the brain functions.

2. What did you notice about the computations that you ran in the TensorFlow
programs (i.e. interactive models) in the tutorial?
Tensorflow uses components and basis vectors, where basis vectors transform one way between reference frames and the components transform in just such a way as to keep the combination between components and basis vectors the same.


### Day 2: (02/09/21): Introduction to Machine Learning (ML) and Scikit-Learn

Materials for day 2 can be found [here](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%202%20-%20INTRODUCTION%20TO%20MACHINE%20LEARNING%20(ML)%20AND%20SCIKIT-LEARN)

1. What is the difference between supervised and unsupervised learning? 
Supervised Learning refers to a program that has been trained on a pre defined set of training examples. The main goal is to come up with a predictor function ```h(x)``` that uses mathematical algorithms to optimize this function so that the given data input ```x``` predicts the ```h(x)``` value.
Unsupervised Learning refers to a program that is given a bunch of data and must find patterns and relations in itself.

2. Describe why the following statement is FALSE: ```scikit-Learn``` has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries.
This statement is false because ```scikit-Learn``` is a library in python that provides unsupervised and supervised learning algorithms. It does not have the power to visualize data, that is why we need to import other libraries to visualize it.

### Day 1: (02/08/21): Introduction to SureStart
I am an undergraduate student studying Neuroscience at Boston University and I am excited to be introduced into the world of AI and Machine Learning! I specifically want to learn techniques that will complement computational neuroscience, such as creating neural networks that can mimic brain pathways. I am also interested in the application that this field has in medicinal and health sciences. I think this a great opportunity that will prepare me not only for future elective classes that I will take, but also I hope to gain skills that will make me a competitive candidate in the marketplace.
