import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# read the log file
with open("C:\\Users\\dhc40\\manyDG\\log_new\\sleep\\sleep_dev_100_1678259913.7813637.log", "r") as f:
    log_lines = f.readlines()

# extract the test and validation accuracies from the log file
test_accs = []
val_accs = []
for line in log_lines:
    if "test accuracy" in line:
        test_acc = float(line.split(":")[1].split(",")[0].strip())
        test_accs.append(test_acc)
    elif "val accuracy" in line:
        val_acc = float(line.split(":")[1].split(",")[0].strip())
        val_accs.append(val_acc)

# plot the test and validation accuracies for each epoch
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(len(test_accs)), test_accs, label='Test Accuracy')
ax.plot(range(len(val_accs)), val_accs, label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Test and Validation Accuracies for Each Epoch')
ax.legend()

# set y-axis ticks format to display with a precision of 4 decimal places
fmt = ticker.FormatStrFormatter('%.5f')
ax.yaxis.set_major_formatter(fmt)
print(test_accs)
# plt.show()