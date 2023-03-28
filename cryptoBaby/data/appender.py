import csv

# Open file
with open('D:\\GitHub\\ai\\AIconomy\\cryptoBaby\\data\\BTC.csv', 'r') as csv_file_in:
    # Read file
    csv_reader = csv.reader(csv_file_in, delimiter=',')

    # Open output file
    with open('BTCout.csv', 'w', newline='') as csv_file_out:
        # Write to file
        csv_writer = csv.writer(csv_file_out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # Iterate over rows
        for i, row in enumerate(csv_reader):
            # Write row
            csv_writer.writerow([i] + row)