from train_module import train_cnn, train_nn

dataset = 'BD'
dirname  = 'results/CNN/{}'.format(dataset) 

train_cnn(dataset, dirname)
dataset = 'RT'
dirname = 'results/CNN/{}'.format(dataset)

train_nn (dataset , dirname)

# Train and evaluate the NN model on BD data set
dataset  = 'BD'
dirname  = 'results/NN/{}'.format(dataset) # this can be changed as you wish
train_nn(dataset, dirname)

# Train and evaluate the NN model on RT data set
dataset  = 'RT'
dirname  = 'results/NN/{}'.format(dataset) # This can be changed as you wish
train_nn(dataset, dirname)