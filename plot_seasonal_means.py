import csv
import matplotlib.pyplot as plt

# Read the CSV
data = {}
seasons = ['DJF', 'MAM', 'JJA', 'SON']
with open('/Users/gillian/Documents/seasonal_means_2000_2080.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    ensemble_cols = headers[2:]  # Skip Year and Season
    for row in reader:
        year = int(row[0])
        season = row[1]
        values = [float(x) for x in row[2:]]
        if season not in data:
            data[season] = {'years': [], 'values': {col: [] for col in ensemble_cols}}
        data[season]['years'].append(year)
        for i, col in enumerate(ensemble_cols):
            data[season]['values'][col].append(values[i])

# Plot for each season
for season in seasons:
    plt.figure(figsize=(12, 8))
    for col in ensemble_cols:
        plt.plot(data[season]['years'], data[season]['values'][col], label=col, linewidth=1)
    plt.xlabel('Year')
    plt.ylabel('Groundwater Level Mean')
    plt.title(f'Seasonal Means for {season} (2000-2080)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/Users/gillian/Documents/seasonal_plot_{season}.png', dpi=300)
    plt.close()

print("Plots saved as seasonal_plot_DJF.png, seasonal_plot_MAM.png, etc.")