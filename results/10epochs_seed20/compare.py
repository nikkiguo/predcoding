import matplotlib.pyplot as plt

# Define the constant for the x-axis range
N_EPOCHS = 10  

# Load y-axis data from files
def load_data(filename):
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file]

tanh = load_data('data_tanh.txt')
logsig = load_data('data_logsig.txt')
linear = load_data('data_linear.txt')

x = list(range(0, N_EPOCHS))

# Plot each line
plt.plot(x, tanh, label="tanh", color="blue", marker="o")
plt.plot(x, logsig, label="logsig", color="green", marker="o")
plt.plot(x, linear, label="linear", color="red", marker="o")

# Customize the plot
plt.ylim(0, 1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of PCN Using Different Activation Functions")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
