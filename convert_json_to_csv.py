import json
import csv

# Path to your JSON annotation file
json_file_path = 'annotations.json'

# Path to save the CSV file
csv_file_path = 'annotations.csv'

# Load JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write CSV header - customize based on your JSON keys
    # Example header: image filename, blur, bright, dark, framing, obstruction, rotation, other, unrecognizable
    header = ['image']
    # Assuming labels keys can vary, get keys from first item's labels dictionary
    first_item = data[0]
    label_keys = list(first_item['labels'].keys())
    header.extend(label_keys)
    header.append('unrecognizable')  # if present in your JSON
    
    writer.writerow(header)
    
    # Write rows - one row per image
    for item in data:
        row = [item['image']]
        for key in label_keys:
            row.append(item['labels'].get(key, 0))
        # Append unrecognizable count if present, else 0
        row.append(item.get('unrecognizable', 0))
        writer.writerow(row)

print(f"CSV file saved to {csv_file_path}")
