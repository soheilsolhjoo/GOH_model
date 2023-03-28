from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

# Define custom loss function
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# Define training data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Define the model
model = Sequential()
model.add(Dense(units=2, input_dim=2, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with custom loss function
model.compile(loss=custom_loss, optimizer='adam')

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss = model.evaluate(X, y, verbose=0)

# Print the loss
print('Loss: ', loss)
