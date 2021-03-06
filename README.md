# Sure Start VAIL 2021 Program
A repository for the SureStart VAIL 2021 Program that I am completing as an AI and Machine Learning trainee. 
Program includes techincal skill training on:
- Machine learning
- Natural language processing
- Computer vision
- Data analytics & visualization
- Ethical AI development

## Responses

## Week 4 - Applications of Machine Learning

## Day 24: Affective Computing

Materials for [Day 24](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2024%20-%20Affective%20Computing).

Today I implemented a Speech Emotion Analyzer model on the [RAVDESS Dataset](https://smartlaboratory.org/ravdess/), which consists of female and male actors voices. I used a Speech Emotion Recognition tutorial by [shivamburnwal](https://www.kaggle.com/shivamburnwal/speech-emotion-recognition). 
I also followed a tutorial in order to record my own voice with pyaudio and wave.

## Day 23: Computer Vision (CV)

Materials for [Day 23](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2023%20-%20Computer%20Vision%20(CV))

Today I implemented an emotion detection project using OpenCV.

## Day 22: Natural Language Processing (NLP)

It is important to consider the ethical implications of big NLP models, such as that of the GPT-2 model. This model is a large-scale unsupervised language model which generates paragraphs of text, performs on language modeling benchmarks and can do reading comprehension, machine translation, question answering and summarization. It's main goal is to predict the next word given some previous text.
However, there are ethical implication of big NLP models such as the GPT-2, which is one of the reasons that it was not released. Some of the malicious applications could include generating misleading news articles, impersonating people online, automating the production of faked content on social medial and automating the production of spam. I think that it is a shame that due to malicious practices, models such as these cannot be used by the public. I think that public data and public models can advance research significantly by allowing cross-colaboration.

## Week 3 - Functions and Predictions

## Day 19: Upsampling and Autoencoders

Materials for [Day 19](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2019%20-%20Upsampling%20and%20Autoencoders)

Today, I completed an upsampling tutorial about adding upscaling layers to a deep learning model and followed an autoencoder tutorial used on the MNIST dataset.

## Day 18: Data Overfitting and Regularization

Materials for [Day 18](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2018%20-%20Data%20Overfitting%20and%20Regularization)

Today, I used a hands on tutorial about differnt approaches to handle overfitting in deep learning models for a twitter dataset [(Twitter US Airline Sentiment data set)](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). We compared three models for three methods of overfitting:
1. Reduce the network’s capacity by removing layers or reducing the number of elements in the hidden layers.
2. Apply regularization, which comes down to adding a cost to the loss function for large weights.
3. Use Dropout layers, which will randomly remove certain features by setting them to zero.

Overall, the accuracy fo the data was able to increase, although still relatively low (78.01%), all three options reduced overfitting, however, the dropout layer performed the best on the test data.

For the Neural Network for Housing Prices that I completed yesterday (and added a model that used the mean squared error loss function), by looking at the model loss plot you could see that this loss function had the smallest loss for increasing epochs compared to the other models that used the binary crossentropy loss function.

## Day 17: Loss Functions

Materials for [Day 17](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2017%20-%20Loss%20Functions)

Today, we learned about loss functions and using housing price data from Zillow, I trained the data set using a Neural Network and then implemented a model that used the mean squared error loss function to compare how it does compared to other loss functions.

## Day 16: Activation Functions

Material for [Day 16](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2016%20-%20Activation%20Functions)

The Rectified Linear Activation Function (ReLU) is a piecewise linear function that will output the input directly if it is positive and if it is negative, it will ouput zero. It is easier to train that other activation functions, such as the sigmoid and TanH function and therefore achieves better performmance. The ReLU activation function has four main advantages: Computational Simplicity, Representation Sparsity, Linear Behavior and Training Deep Networks.

1. Computational Simplicity: With the use of the max() function, ReLU becomes trivial to impliment and much cheaper than using exponential functions (sigmoid and tanH)
2. Representational Sparsity: It outputs the true value zero, instead of approximating it like the tanh and sigmoid functions. Therefore, hidden layers are able to contain more than one true zero value.
    - Application: Autoencoders, where network learns a compact representation of the code layer (images or series).
3. Linear Behaviour: It acts like a linear activation function and is therefore easier to optimize. This mostly gets rid of the vanishing gradients proplem because they stay proportional to the node actications
4. Training Deep Networks: Exploits improvements in hardware and successfully is able to train deep multi-layered networks with backpropagation.
    - Application: Boltzmann machines are no longer needed (stochastic recurrent neural network).

[Source](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

## Day 15: Ethics Driven ML Practice

Materials for [Day 15](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2015%20-%20Ethics%20Driven%20ML%20Practice)

Today I built a gender classification model using a dataset of facial images. Unfortunately, the model was taking several hours to train, so I was unable to use the model to predict a gender. However, I got to fold 3 of 10 and the overall accuracy at that point was 94.67 %.

## Week 2 - CNN, Data and Machine Learning

### Day 12: Image Classification and ML

Materials for [Day 12](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2012%20-%20Image%20Classification%20and%20ML)

Today, by using this Keras Kaggle Notebook, I learned how to build an Image Classification model, however, I changed the epoch size to 10 from 50 and preprocessed with option 2 (apply it to the dataset) instead of option 1 (make it part of the model). I noticed that the time it took for the model to train was much longer and obbviously the accuracy decreased with a lower epoch number.

### Day 11: Neural Network Practice with MNIST Digits

Materials for [Day 11](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%2011%20-%20Neural%20Network%20Practice%20with%20MNIST%20Digits)

Following the guide to develop a [CNN for classifying MNIST datasets](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/), I was able to achieve a 99.250 % accuracy and the model was able to predict the correct number (7) from a sample image that I introduced it to.

### Day 10: Neural Networks Layers

A Convolutional Neural Network is a type of neural network composed of a convolution/pooling mechanism that breaks up the image into features and analyzes them and a fully connected layer that takes the output of convolution/pooling and predicts the best classifier for the image.
- Much more efficient
- Only connected to a few local neurons in previous layer with the same set of weights
- Applications: Image processing and Computer Vision

Whilst, a fully connected neural network is the network architecture, in which all neurons connect to all neurons in the next layer
- In a fully connected layer, each neuron is connected to every neuon in previous layer (carrying its own weight)
- Not very good for feature extraction
- Applications: Not very efficient for image recognition and classification, as they are prone to overfitting. Used for solving buisness problems, such as sales and forecasting, customer research, data validation and risk managment.

[Article](https://missinglink.ai/guides/convolutional-neural-networks/fully-connected-layers-convolutional-neural-networks-complete-guide/#section3)

### Day 9: Algorithmic Bias and Data sets

Today, I played [Survival of the Best Fit](https://www.survivalofthebestfit.com) to learn more about how AI might impact human resources and hiring processes in different fields.

1.How do you think Machine Learning or AI concepts were utilized in the design of this game?
In this game, classification was used to decide whether to hire or reject an applicant. The training data that is used came from the manual hiring process that I completed at the start of the game. In this case, I was hiring based on ambition and tended to hire more orange people than blue. Therefore, once the machine algorithm had trained with this data set, it began to mimick the past hiring process behaviour and therefore rejected more blue than orange, even though there where blue candidates that where highly qualified. 

2.Can you give a real-world example of a biased machine learning model, and share your ideas on how you make this model more fair, inclusive, and equitable? Please reflect on why you selected this specific biased model.

The COMPAS system is a regression model used ot predict whether or not a perpetrator was likely to recidivate. However, the issue of the model was that it was predicting double the amount of false positives for African American ethnicities than for Caucasian ethnicites. See [here](https://aibusiness.com/document.asp?doc_id=761095). The problem was that it was traied on a dataset that had a very small number of features and that the model did not consider domain, questioning and the answers depending on racial and sexual components. Therefore, in order to imporve this model, it would be imperative to have a very large dataset that would account for components such as race and sex so that the model would not have human biases creeping into the model.

### Day 8: Introduction to Convolutional Neural Networks (CNN)

Materials for [Day 8](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%208%20-%20Introduction%20to%20Convolutional%20Neural%20Networks%20(CNN))

Today, I evaluated the model performance by looking at the confusion matrices for the MNIST data set and compared it to the performance reported in the original [Kaggle Tutorial Notebook](https://www.kaggle.com/kanncaa1/convolutional-neural-network-cnn-tutorial) that used a smaller subset of the data. From the confusion matrix I obtained, you can see that I got a higher accuracy than the original Tutorial Notebook. This is most likely due to the fact that the model had more data to train with initially.
The recall was much higher, where my confusion matrix was getting up to 679 correct, whilst the tutorial matrix had a maximum of 477 predictions correct.


## Week 1 - The Basics

### Day 5: (02/12/21): What are Neural Networks (NN)?

Materials for [Day 5](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%205%20-%20What%20are%20Neural%20Networks)

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

Materials for [Day 3](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%203%20-%20Introduction%20to%20ML%20and%20Tensor%20Flow)

1. What are “Tensors” and what are they used for in Machine Learning? 
TensorFlow is an open source library created by Google for numerical computation and machine learning. It uses ML and deep learning, such as neural networking models and algorithms. It is inspired by the way that the brain functions.

2. What did you notice about the computations that you ran in the TensorFlow
programs (i.e. interactive models) in the tutorial?
Tensorflow uses components and basis vectors, where basis vectors transform one way between reference frames and the components transform in just such a way as to keep the combination between components and basis vectors the same.


### Day 2: (02/09/21): Introduction to Machine Learning (ML) and Scikit-Learn

Materials for [Day 2](https://github.com/myriaambp/SureStart-VAIL/tree/main/Day%202%20-%20INTRODUCTION%20TO%20MACHINE%20LEARNING%20(ML)%20AND%20SCIKIT-LEARN)

1. What is the difference between supervised and unsupervised learning? 
Supervised Learning refers to a program that has been trained on a pre defined set of training examples. The main goal is to come up with a predictor function ```h(x)``` that uses mathematical algorithms to optimize this function so that the given data input ```x``` predicts the ```h(x)``` value.
Unsupervised Learning refers to a program that is given a bunch of data and must find patterns and relations in itself.

2. Describe why the following statement is FALSE: ```scikit-Learn``` has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries.
This statement is false because ```scikit-Learn``` is a library in python that provides unsupervised and supervised learning algorithms. It does not have the power to visualize data, that is why we need to import other libraries to visualize it.

### Day 1: (02/08/21): Introduction to SureStart

I am an undergraduate student studying Neuroscience at Boston University and I am excited to be introduced into the world of AI and Machine Learning! I specifically want to learn techniques that will complement computational neuroscience, such as creating neural networks that can mimic brain pathways. I am also interested in the application that this field has in medicinal and health sciences. I think this a great opportunity that will prepare me not only for future elective classes that I will take, but also I hope to gain skills that will make me a competitive candidate in the marketplace.
