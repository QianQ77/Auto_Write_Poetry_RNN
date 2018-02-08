# Auto_Write_Poetry_RNN

In this project, we give an RNN a huge chunk of text and ask it to model the probability distribution, essentially predicting the next character in the sequence;

Using tensorflow, scikit-learn, we convert the texts into one-hot-encoding format, where the vector length of the encoding is equal to the number of possible characters and only one item in the vector is set to 1, with the rest being 0;

Then, implement a multi-layer RNN that can take these encodings as inputs using Tensorflow and train the model on the poetry obtained;

Feed a character (can start with random character) into the RNN and get a distribution over what characters are likely to come next. Sample from this distribution, and feed it right back in to get the next letter. 

## 1

Used poems of John Keats downloaded from following link: 
https://archive.org/stream/poemsbyjohnkeats00keat/poemsbyjohnkeats00keat_djvu.txt
 
## 2.

load the text file and convert it into integers in order to make it easier to use as input in the network

## 3.

Created model using RNN technique, following are key steps in the model creation:

a.	One-hot encode the input tokens

b.	Build the input place holder and the LTSM cell

c.	Run each sequence step through RNN and collect the output

d.	Get softmax predictions and logits

e.	Ensure the gradients never grow overly large using gradient clipping

f.	Use Adam Optimizer for the learning step

## 4.

Train the model on the poetry obtained:

a.	Create batch of inputs and train the model

b.	For every 200 outputs iterations, the model is saved

c.	The model is saved at every checkpoint inside savepoints directory

## 5.

The first character is fed randomly using random ascii generator and the poetry is generated. The length of the poetry is 600 characters.


# Sample Output text produced by the RNN

Forest him. 

With the sweet primoness of all the shorns. 
The who should be some honour to the speak. 

The silent was they shade, the silver bright 

Of the shade borrow from their sight and deed. 

They should be thy bliss'd their breasts 

Of some who stars the mountains of the stream. 

And the shadow of mortals, though she loved 

To she he they still morning to the streams 

That they should see at once the sharp of shade, and flee 

A chear and temple of the mountain plain. 

I see the head the sound of melody. 

The streams of him, a shepherd starring trees 
Their face of half-deep then


# How To Run
First, run Train.py to build and train a RNN model;
Second, run GeneratePoetry.py to generate a poem and save it as Output.txt
