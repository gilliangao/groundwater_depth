import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def remove_outliers(data):
    if len(data) < 4:
        return data  # Not enough data for IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]

# Read the CSV
data = []
with open('/Users/gillian/Documents/seasonal_means_2000_2080.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    ensemble_cols = headers[2:]
    for row in reader:
        year = int(row[0])
        season = row[1]
        values = [float(x) for x in row[2:]]
        data.append({'year': year, 'season': season, 'values': values})

# For plots: seasonal means with outliers removed, for each season
seasons = ['DJF', 'MAM', 'JJA', 'SON']
seasonal_data = {season: {'years': [], 'means': []} for season in seasons}

for row in data:
    season = row['season']
    year = row['year']
    values = row['values']
    # Remove outliers from the 11 ensemble values for this year-season
    filtered = remove_outliers(values)
    if filtered:
        seasonal_data[season]['years'].append(year)
        seasonal_data[season]['means'].append(np.mean(filtered))

# Sort the data by year
for season in seasons:
    years_means = list(zip(seasonal_data[season]['years'], seasonal_data[season]['means']))
    years_means.sort()
    seasonal_data[season]['years'], seasonal_data[season]['means'] = zip(*years_means)

# Calculate trends and save to CSV
trend_results = []
regression_models = {}
print("Seasonal Trend Analysis (Linear Regression)")
print("Season\tSlope\t\tIntercept\tR-squared\tP-value")
for i, season in enumerate(seasons):
    years = np.array(seasonal_data[season]['years'])
    means = np.array(seasonal_data[season]['means'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, means)
    print(f"{season}\t{slope:.4f}\t\t{intercept:.2f}\t{r_value**2:.4f}\t\t{p_value:.4f}")
    trend_results.append({
        'Season': season,
        'Slope': slope,
        'Intercept': intercept,
        'R_squared': r_value**2,
        'P_value': p_value,
        'Std_Error': std_err
    })
    regression_models[season] = {'slope': slope, 'intercept': intercept}

# Save trend results to CSV
with open('/Users/gillian/Documents/seasonal_trends_table.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Season', 'Slope', 'Intercept', 'R_squared', 'P_value', 'Std_Error'])
    writer.writeheader()
    writer.writerows(trend_results)

# Calculate trends and plot combined
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'orange']
season_names = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)']

for i, season in enumerate(seasons):
    years = np.array(seasonal_data[season]['years'])
    means = np.array(seasonal_data[season]['means'])
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, means)
    
    # Trend line
    trend_line = slope * years + intercept
    
    # Plot data points
    plt.scatter(years, means, color=colors[i], alpha=0.6, s=10, label=f'{season_names[i]} data')
    
    # Plot trend line
    plt.plot(years, trend_line, color=colors[i], linewidth=2, 
             label=f'{season_names[i]} trend (slope: {slope:.4f})')

plt.xlabel('Year')
plt.ylabel('Groundwater Level Mean (Outliers Removed)')
plt.title('Seasonal Trends in Groundwater Levels (2000-2080)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/gillian/Documents/seasonal_trends_combined.png', dpi=300)
plt.close()

print("\nTrend table saved as seasonal_trends_table.csv")
print("Combined trend plot saved as seasonal_trends_combined.png")

# For table: predicted seasonal means for 2025, 2030, 2050 using regression
target_years = [2025, 2030, 2050]
predicted_table = []

for year in target_years:
    row = {'Year': year}
    for season in seasons:
        model = regression_models[season]
        predicted = model['slope'] * year + model['intercept']
        row[season] = predicted
    predicted_table.append(row)

# Save predicted seasonal means table to CSV
with open('/Users/gillian/Documents/predicted_seasonal_means_table.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Year', 'DJF', 'MAM', 'JJA', 'SON'])
    writer.writeheader()
    writer.writerows(predicted_table)

# Print predicted table
print("\nPredicted Seasonal Groundwater Means for Selected Years (From Linear Regression)")
print("Year\tDJF\t\tMAM\t\tJJA\t\tSON")
for row in predicted_table:
    print(f"{row['Year']}\t{row['DJF']:.2f}\t\t{row['MAM']:.2f}\t\t{row['JJA']:.2f}\t\t{row['SON']:.2f}")

print("\nPredicted seasonal means table saved as predicted_seasonal_means_table.csv")