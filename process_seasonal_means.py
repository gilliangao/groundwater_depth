import csv
from datetime import datetime
from collections import defaultdict

# Read the CSV
data = []
with open('/Users/gillian/Documents/FF-RGroundwater-TL33_4-GWL.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        data.append(row)

# Parse dates and filter 2000-2080
filtered_data = []
for row in data:
    date_str = row[2]
    date = datetime.strptime(date_str, '%d/%m/%Y')
    year = date.year
    month = date.month
    if 2000 <= year <= 2080:
        filtered_data.append([year, month] + [float(x) for x in row[3:]])

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'

# Group by year and season
seasonal_data = defaultdict(lambda: defaultdict(list))
for row in filtered_data:
    year, month = row[0], row[1]
    season = get_season(month)
    values = row[2:]
    for i, val in enumerate(values):
        seasonal_data[(year, season)][i].append(val)

# Calculate means
ensemble_cols = headers[3:]
result = []
for (year, season), values_dict in seasonal_data.items():
    row = [year, season]
    for i in range(len(ensemble_cols)):
        vals = values_dict[i]
        mean_val = sum(vals) / len(vals) if vals else 0
        row.append(mean_val)
    result.append(row)

# Write to CSV
with open('/Users/gillian/Documents/seasonal_means_2000_2080.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Year', 'Season'] + ensemble_cols)
    writer.writerows(result)

print("Seasonal means calculated and saved to seasonal_means_2000_2080.csv")