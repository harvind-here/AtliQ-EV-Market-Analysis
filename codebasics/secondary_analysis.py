import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
sales_by_state = pd.read_csv('RPC12_Input_For_Participants/datasets/electric_vehicle_sales_by_state.csv')
sales_by_makers = pd.read_csv('RPC12_Input_For_Participants/datasets/electric_vehicle_sales_by_makers.csv')
dim_date = pd.read_csv('RPC12_Input_For_Participants/datasets/dim_date.csv')

# Convert date columns to datetime
sales_by_state['date'] = pd.to_datetime(sales_by_state['date'], format='%d-%b-%y')
sales_by_makers['date'] = pd.to_datetime(sales_by_makers['date'], format='%d-%b-%y')
dim_date['date'] = pd.to_datetime(dim_date['date'], format='%d-%b-%y')

# Merge sales_by_state with dim_date
sales_by_state = pd.merge(sales_by_state, dim_date, on='date')

# Calculate penetration rate
sales_by_state['penetration_rate'] = (sales_by_state['electric_vehicles_sold'] / sales_by_state['total_vehicles_sold']) * 100

# Analysis for 4-wheeler EVs in 2023 and 2024
four_wheeler_sales = sales_by_state[(sales_by_state['vehicle_category'] == '4-Wheelers') & 
                                    (sales_by_state['fiscal_year'].isin([2023, 2024]))]

# Top 5 states by 4-wheeler EV sales
top_5_states = four_wheeler_sales.groupby('state')['electric_vehicles_sold'].sum().nlargest(5).index

# Visualize 4-wheeler EV sales trend for top 5 states
plt.figure(figsize=(12, 6))
for state in top_5_states:
    state_data = four_wheeler_sales[four_wheeler_sales['state'] == state]
    plt.plot(state_data['date'], state_data['electric_vehicles_sold'], label=state)

plt.title('4-Wheeler EV Sales Trend (2023-2024) - Top 5 States')
plt.xlabel('Date')
plt.ylabel('Electric Vehicles Sold')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze penetration rates
penetration_rates = four_wheeler_sales.groupby('state')['penetration_rate'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
penetration_rates.plot(kind='bar')
plt.title('Average 4-Wheeler EV Penetration Rate by State (2023-2024)')
plt.xlabel('State')
plt.ylabel('Penetration Rate (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyze top manufacturers
top_manufacturers = sales_by_makers[(sales_by_makers['vehicle_category'] == '4-Wheelers') & 
                                    (sales_by_makers['date'].dt.year.isin([2023, 2024]))].groupby('maker')['electric_vehicles_sold'].sum().nlargest(5)

plt.figure(figsize=(10, 6))
top_manufacturers.plot(kind='bar')
plt.title('Top 5 4-Wheeler EV Manufacturers (2023-2024)')
plt.xlabel('Manufacturer')
plt.ylabel('Electric Vehicles Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Top 5 states by 4-wheeler EV sales (2023-2024):")
print(four_wheeler_sales.groupby('state')['electric_vehicles_sold'].sum().nlargest(5))

print("\nTop 5 states by 4-wheeler EV penetration rate (2023-2024):")
print(penetration_rates.head())

print("\nTop 5 4-wheeler EV manufacturers (2023-2024):")
print(top_manufacturers)

# Additional analysis for government incentives and subsidies
# (This information is not provided in the given datasets, so additional data sources are needed)

# Additional analysis for charging station infrastructure
# (Additional data required)

# Research potential brand ambassadors based on the EV market in India
# (This requires additional research and analysis)

# Recommendations for AtliQ Motors
# 1. Target the top 5 states with high 4-wheeler EV sales and penetration rates
# 2. Partner with top manufacturers to offer competitive products
# 3. Invest in marketing campaigns to raise awareness about the benefits of EVs
