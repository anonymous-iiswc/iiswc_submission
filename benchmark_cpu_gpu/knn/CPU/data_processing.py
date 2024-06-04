import random
import csv

# Load the data from the CSV file
with open('./loan_data.csv', 'r') as file:
    lines = file.readlines()

# Parse the CSV data
header = lines[0].strip().split(',')
data = [line.strip().split(',') for line in lines[1:]]

# Debug print to check the header and the first few lines of the data
#print("Header:", header)
#print("First 3 rows of data:", data[:3])

# Find the index of the "purpose" column
if 'purpose' in header:
    purpose_index = header.index('purpose')
else:
    raise ValueError("'purpose' column not found in the header")

# Remove the "purpose" column from the data and display the top 3 samples
header.pop(purpose_index)
data = [row[:purpose_index] + row[purpose_index+1:] for row in data]
#print("Updated Header:", header)
#print("First 3 rows of data after removing 'purpose' column:", data[:3])

# Convert data to 32-bit integers where possible
def convert_to_int32(row):
    new_row = []
    for value in row:
        new_row.append(int(float(value)))
    return new_row

data = [convert_to_int32(row) for row in data]

# Train and Test Split
random.seed(255)

# the index of not_fully_paid is the last index in the dataset
target_index = 13

# Create a list of indices and shuffle them
indices = list(range(len(data)))
random.shuffle(indices)

# Split the indices into training and testing sets (75:25 ratio)
split_index = int(len(indices) * 0.75)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Create training and testing datasets
train_data = [data[i] for i in train_indices]
test_data = [data[i] for i in test_indices]

print(train_data[:3])


# Save the train_data to a CSV file
with open('train_knn_benchmark_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(header)
    writer.writerows(train_data)


# Save the test_data to a CSV file
with open('test_knn_benchmark_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(header)
    writer.writerows(test_data)