# MULITPLE INPUT TO 1 OUTPUT
toes = [8.5 , 9.5, 9.9, 9.0]
winloss = [0.65, 0.8, 0.8, 0.9]
fans = [1.2 , 1.3, 0.5, 1.0]

# multiple inputs with only 1 output
input = [toes[0], winloss[0], fans[0]]
weights = [0.1, 0.2, -.1]
alpha = 0.01

win_or_lose_binary = [1, 1, 0, 1]
# prediction tries to get close to true value
true = win_or_lose_binary[0]

def neural_network(inp, wei):
    pred = 0
    for i in range(len(inp)):
        pred += (inp[i] * wei[i])
    return pred

def ele_mul(number, vector):
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

for iter in range(3):
    # multiply weights * input
    prediction = neural_network(input, weights)
    # calc error
    error = (prediction - true) ** 2
    # calc raw error
    delta = prediction - true
    # calc weight delta to correct for scaling, negative reversal, and stopping
    # basically raw error * input
    weight_deltas = ele_mul(delta, input)
    
    # print out variables
    print("input: ", input)
    print("weights: ", weights)
    print("prediction: ", prediction)
    print("true value: ", true)
    print("error: ", error)
    print("delta raw error", delta)
    print("weight deltas: " , weight_deltas)

    # update new weights
    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]
    print("new weights:", weights)
    print("\n")