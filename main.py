import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check if the number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

numbers = list(range(1, 101))
labels = ['prime' if is_prime(num) else 'non-prime' for num in numbers]

# Splitting data into training, validation, and testing
train_data, temp_data, train_labels, temp_labels = train_test_split(numbers, labels, test_size=0.4, stratify=labels, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# Visualization
categories = ['Training', 'Validation', 'Testing']
prime_counts = [train_labels.count('prime'), val_labels.count('prime'), test_labels.count('prime')]
non_prime_counts = [train_labels.count('non-prime'), val_labels.count('non-prime'), test_labels.count('non-prime')]

x = range(len(categories))

plt.bar(x, prime_counts, width=0.4, label='Prime', color='green', align='center')
plt.bar(x, non_prime_counts, width=0.4, label='Non-Prime', color='red', bottom=prime_counts, align='center')

plt.xlabel('Dataset Split')
plt.ylabel('Number of Samples')
plt.title('Dataset Split for Prime and Non-Prime Numbers')
plt.xticks(x, categories)
plt.legend()
plt.show()