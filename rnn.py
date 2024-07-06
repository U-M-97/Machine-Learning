import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.wxh = None
        self.whh = None
        self.why = None
        self.bh = None
        self.by = None

    def initialize_parameters(self):
        self.wxh = np.random.normal(loc=0.0, scale=np.sqrt(2 / (self.hidden_size + self.input_size)), size=(self.hidden_size, self.input_size))
        self.whh = np.random.normal(loc=0.0, scale=np.sqrt(2 / (self.hidden_size + self.hidden_size)), size=(self.hidden_size, self.hidden_size))
        self.why = np.random.normal(loc=0.0, scale=np.sqrt(2 / (self.output_size + self.hidden_size)), size=(self.output_size, self.hidden_size))
        self.bh = np.random.normal(loc=0.0, scale=np.sqrt(2 / self.hidden_size), size=self.hidden_size)
        self.by = np.random.normal(loc=0.0, scale=np.sqrt(2 / self.output_size), size=self.output_size)

    def mse(self, t, targets):
        return np.sum((targets[t] - self.y[t]) ** 2) / len(self.y[t])

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.hidden_states = []
        self.y = []
        
        for x in inputs:
            h = np.tanh((np.dot(self.whh, h) + np.dot(self.wxh, x) + self.bh))
            o = np.dot(self.why, h) + self.by.reshape(-1, 1)
            self.hidden_states.append(h)
            self.y.append(o)

        return self.y

    def backward(self, inputs, targets):
        dl_dwxh = np.zeros((self.wxh.shape))
        dl_dwhh = np.zeros((self.whh.shape))
        dl_dwhy = np.zeros((self.why.shape))
        dl_dbh = np.zeros((self.bh.shape))
        dl_dby = np.zeros((self.by.shape))

        dh_next = np.zeros((self.hidden_size, 1))

        for i in range(len(inputs) - 1, 0, -1):
            dl_dy = self.y[i] - targets[i].reshape(-1, 1)
            dl_dwhy += np.dot(dl_dy, self.hidden_states[i].T)
            print(dl_dy.shape)
            print(dl_dby.shape)
            dl_dby += dl_dy

            dl_dh = np.dot(self.why[i].T, dl_dy) + dh_next
            dh_do = (1 - self.hidden_states[i] ** 2) * dl_dh
            dl_dwhh = np.dot(dh_do, self.hidden_states[i - 1].T)
            dl_dbh += dh_do
            dl_dwxh = np.dot(dh_do, inputs[i])

            dh_next = np.dot(self.whh.T, dh_do)

        return dl_dwxh, dl_dwhh, dl_dwhy, dl_dbh, dl_dby 
    
    def update_parameters(self, dl_dwxh, dl_dwhh, dl_dwhy, dl_dbh, dl_dby, lr):
        self.wxh -= lr * dl_dwxh 
        self.whh -= lr * dl_dwhh 
        self.why -= lr * dl_dwhy 
        self.bh -= lr * dl_dbh
        self.by -= lr * dl_dby

def main():
    input_size = 3
    hidden_size = 5
    output_size = 2
    sequence_length = 4
    epochs = 1000
    lr = 0.01

    inputs = [np.random.randn(input_size) for _ in range(sequence_length)]
    targets = [np.random.randn(output_size) for _ in range(sequence_length)]

    rnn = RNN(input_size, hidden_size, output_size, lr)
    rnn.initialize_parameters()

    for epoch in range(epochs):
        rnn.forward(inputs)
        dl_dwxh, dl_dwhh, dl_dwhy, dl_dbh, dl_dby = rnn.backward(inputs, targets)
        rnn.update_parameters(dl_dwxh, dl_dwhh, dl_dwhy, dl_dbh, dl_dby)

        if epoch % 100 == 0:
            loss = 0
            for t in range(len(sequence_length)):
                loss += rnn.mse(t, targets)
            print("Loss = ", loss)
            
main()