import psutil

total_memory = psutil.virtual_memory().total
total_memory_gb = total_memory / (1024 ** 3)

print("Total memory available:", total_memory_gb, "GB")
