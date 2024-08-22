import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression


ev_sales_by_state = pd.read_csv('RPC12_Input_For_Participants/datasets/electric_vehicle_sales_by_state.csv')
ev_sales_by_makers = pd.read_csv('RPC12_Input_For_Participants/datasets/electric_vehicle_sales_by_makers.csv')
dim_date = pd.read_csv('RPC12_Input_For_Participants/datasets/dim_date.csv')

ev_sales_by_state['date'] = pd.to_datetime(ev_sales_by_state['date'], format='%d-%b-%y')
ev_sales_by_makers['date'] = pd.to_datetime(ev_sales_by_makers['date'], format='%d-%b-%y')
dim_date['date'] = pd.to_datetime(dim_date['date'], format='%d-%b-%y')


ev_sales_by_state = pd.merge(ev_sales_by_state, dim_date, on='date')
ev_sales_by_makers = pd.merge(ev_sales_by_makers, dim_date, on='date')


ev_sales_by_state.sort_values('date', inplace=True)
ev_sales_by_makers.sort_values('date', inplace=True)

def calculate_cagr(start_value, end_value, num_periods):
    return (end_value / start_value) ** (1 / num_periods) - 1

def top_bottom_makers(df, fiscal_years):
    result = {}
    for fy in fiscal_years:
        fy_data = df[(df['fiscal_year'] == fy) & (df['vehicle_category'] == '2-Wheelers')]
        sales_by_maker = fy_data.groupby('maker')['electric_vehicles_sold'].sum().sort_values(ascending=False)
        result[fy] = {
            'top_3': sales_by_maker.head(3),
            'bottom_3': sales_by_maker.tail(3)
        }
    return result


def top_5_states_penetration(df, fiscal_year):
    fy_data = df[df['fiscal_year'] == fiscal_year].copy()
    fy_data['penetration_rate'] = fy_data['electric_vehicles_sold'] / fy_data['total_vehicles_sold'] * 100
    return fy_data.groupby('state')['penetration_rate'].mean().nlargest(5)


def states_with_negative_penetration(df):
    df['penetration_rate'] = df['electric_vehicles_sold'] / df['total_vehicles_sold'] * 100
    penetration_by_state_year = df.groupby(['state', 'fiscal_year'])['penetration_rate'].mean().unstack()
    negative_growth = penetration_by_state_year[penetration_by_state_year[2024] < penetration_by_state_year[2022]]
    return negative_growth.index.tolist()


def quarterly_trends_top_5_makers(df):
    top_5_makers = df[df['vehicle_category'] == '4-Wheelers'].groupby('maker')['electric_vehicles_sold'].sum().nlargest(5).index
    quarterly_sales = df[(df['vehicle_category'] == '4-Wheelers') & (df['maker'].isin(top_5_makers))].groupby(['maker', 'fiscal_year', 'quarter'])['electric_vehicles_sold'].sum().unstack(level=[1, 2])
    return quarterly_sales

def compare_delhi_karnataka(df, fiscal_year):
    fy_data = df[df['fiscal_year'] == fiscal_year]
    fy_data['penetration_rate'] = fy_data['electric_vehicles_sold'] / fy_data['total_vehicles_sold'] * 100
    comparison = fy_data[fy_data['state'].isin(['Delhi', 'Karnataka'])].groupby('state').agg({
        'electric_vehicles_sold': 'sum',
        'penetration_rate': 'mean'
    })
    return comparison

def cagr_top_5_makers(df):
    top_5_makers = df[df['vehicle_category'] == '4-Wheelers'].groupby('maker')['electric_vehicles_sold'].sum().nlargest(5).index
    sales_by_year = df[(df['vehicle_category'] == '4-Wheelers') & (df['maker'].isin(top_5_makers))].groupby(['maker', 'fiscal_year'])['electric_vehicles_sold'].sum().unstack()
    cagr = sales_by_year.apply(lambda x: calculate_cagr(x[2022], x[2024], 2), axis=1)
    return cagr


def top_10_states_cagr(df):
    total_sales_by_state_year = df.groupby(['state', 'fiscal_year'])['total_vehicles_sold'].sum().unstack()
    cagr = total_sales_by_state_year.apply(lambda x: calculate_cagr(x[2022], x[2024], 2), axis=1)
    return cagr.nlargest(10)


def peak_low_seasons(df):
    monthly_sales = df.groupby(df['date'].dt.month)['electric_vehicles_sold'].sum()
    peak_month = monthly_sales.idxmax()
    low_month = monthly_sales.idxmin()
    return {'peak_month': peak_month, 'low_month': low_month}


def project_ev_sales_2030(df):
    def top_10_states_penetration(df, fiscal_year):
        fy_data = df[df['fiscal_year'] == fiscal_year].copy()
        fy_data['penetration_rate'] = fy_data['electric_vehicles_sold'] / fy_data['total_vehicles_sold'] * 100
        return fy_data.groupby('state')['penetration_rate'].mean().nlargest(10)
    top_10_states = top_10_states_penetration(df, 2024).index[:10]
    projections = {}
    
    for state in top_10_states:
        try:
            state_data = df[df['state'] == state].copy()
            state_data['ds'] = pd.to_datetime(state_data['date'])
            state_data['y'] = state_data['electric_vehicles_sold']
            state_data = state_data.sort_values('ds')
            
            
            if len(state_data) < 24: 
                print(f"Insufficient data for {state}. Using linear regression.")
                X = np.array(range(len(state_data))).reshape(-1, 1)
                y = state_data['y'].values
                model = LinearRegression().fit(X, y)
                future_X = np.array([len(state_data) + 71]).reshape(-1, 1) 
                projection = model.predict(future_X)[0]
            else:
                
                model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                model.fit(state_data[['ds', 'y']])
                future_dates = model.make_future_dataframe(periods=72, freq='ME') 
                forecast = model.predict(future_dates)
                projection = forecast.iloc[-1]['yhat']
            
            projections[state] = max(0, projection) 
        
        except Exception as e:
            print(f"Error processing {state}: {e}")
            projections[state] = None
    
    return pd.Series(projections)


def revenue_growth_rate(df):
    df['revenue'] = df.apply(lambda row: row['electric_vehicles_sold'] * (85000 if row['vehicle_category'] == '2-Wheelers' else 1500000), axis=1)
    revenue_by_year = df.groupby(['fiscal_year', 'vehicle_category'])['revenue'].sum().unstack()
    growth_rate_22_24 = (revenue_by_year.loc[2024] / revenue_by_year.loc[2022] - 1) * 100
    growth_rate_23_24 = (revenue_by_year.loc[2024] / revenue_by_year.loc[2023] - 1) * 100
    return pd.DataFrame({'2022 vs 2024': growth_rate_22_24, '2023 vs 2024': growth_rate_23_24})

print("1. Top 3 and bottom 3 makers for 2-wheelers in FY 2023 and 2024:")
print(top_bottom_makers(ev_sales_by_makers, [2023, 2024]))

print("\n2. Top 5 states with highest penetration rate in FY 2024:")
top_5_states = top_5_states_penetration(ev_sales_by_state, 2024)
print(top_5_states)

print("\n3. States with negative penetration from 2022 to 2024:")
print(states_with_negative_penetration(ev_sales_by_state))

print("\n4. Quarterly trends for top 5 EV makers (4-wheelers):")
print(quarterly_trends_top_5_makers(ev_sales_by_makers))

print("\n5. Compare EV sales and penetration rates in Delhi vs Karnataka for 2024:")
print(compare_delhi_karnataka(ev_sales_by_state, 2024))

print("\n6. CAGR for top 5 4-wheeler makers from 2022 to 2024:")
print(cagr_top_5_makers(ev_sales_by_makers))

print("\n7. Top 10 states with highest CAGR in total vehicles sold from 2022 to 2024:")
print(top_10_states_cagr(ev_sales_by_state))

print("\n8. Peak and low season months for EV sales:")
print(peak_low_seasons(ev_sales_by_state))

print("\n9. Projected EV sales for top 10 states by penetration rate in 2030:")
print(project_ev_sales_2030(ev_sales_by_state))

print("\n10. Revenue growth rate for 2-wheelers and 4-wheelers:")
print(revenue_growth_rate(ev_sales_by_state))


plt.figure(figsize=(12, 6))
sns.lineplot(data=ev_sales_by_state, x='date', y='electric_vehicles_sold', hue='vehicle_category')
plt.title('EV Sales Trend by Vehicle Category')
plt.xlabel('Date')
plt.ylabel('Number of EVs Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=ev_sales_by_makers[ev_sales_by_makers['vehicle_category'] == '4-Wheelers'], 
            x='maker', y='electric_vehicles_sold', hue='fiscal_year')
plt.title('4-Wheeler EV Sales by Maker and Fiscal Year')
plt.xlabel('Maker')
plt.ylabel('Number of EVs Sold')
plt.xticks(rotation=45)
plt.legend(title='Fiscal Year')
plt.tight_layout()
plt.show()


results = pd.DataFrame({
    'Metric': ['Total EVs Sold', 'Avg Penetration Rate'],
    'Value': [ev_sales_by_state['electric_vehicles_sold'].sum(), ev_sales_by_state['electric_vehicles_sold'].sum() / ev_sales_by_state['total_vehicles_sold'].sum() * 100]
})
results.to_csv('ev_analysis_results.csv', index=False)

detailed_results = pd.DataFrame({
    'Metric': ['Total EVs Sold', 'Avg Penetration Rate', 'Top EV Maker 2024', 'Top State by Penetration 2024'],
    'Value': [
        ev_sales_by_state['electric_vehicles_sold'].sum(),
        ev_sales_by_state['electric_vehicles_sold'].sum() / ev_sales_by_state['total_vehicles_sold'].sum() * 100,
        top_bottom_makers(ev_sales_by_makers, [2024])[2024]['top_3'].index[0],
        top_5_states_penetration(ev_sales_by_state, 2024).index[0]
    ]
})
detailed_results.to_csv('ev_analysis_detailed_results.csv', index=False)