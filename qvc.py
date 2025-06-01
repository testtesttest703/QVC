from datetime import datetime, timedelta
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Import other necessary libraries
from google.colab import drive
import pickle
from tqdm import tqdm  # For progress bars
import shutil

# Current date and user
TIMESTAMP = "2025-05-20 17:38:47"  # UTC time - updated with current timestamp
USERNAME = "testtesttest703"

# Define constants for the model
MAX_HISTORY_YEARS = 9  # Maximum years of historical data to use (back to 2016)

# Set up APA-compliant plotting style
plt.style.use('default')  # Reset to default first to avoid any conflicts
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # Standard fonts only
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'png'

# Mount Google Drive
drive.mount('/content/drive')

print(f"Analysis session started at {TIMESTAMP}")
print(f"User: {USERNAME}")

# Valid tickers
valid_tickers = ["NVDA", "GOOG", "MSFT", "AAPL", "TSLA"]

def load_treasury_rates():
    """
    Load daily Treasury rates from CSV file in Google Drive

    Returns:
    pd.DataFrame: DataFrame with daily Treasury rates
    """
    try:
        filepath = f"/content/drive/MyDrive/daily-treasury-rates.csv"

        if os.path.exists(filepath):
            # Load the data using pandas
            df = pd.read_csv(filepath)

            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            print(f"Loaded Treasury rates data spanning from {df['Date'].min()} to {df['Date'].max()}")
            return df
        else:
            print("Treasury rates file not found.")
            return None

    except Exception as e:
        print(f"Error loading Treasury rates: {str(e)}")
        return None

def get_risk_free_rate(treasury_rates, trading_day, days_to_expiration):
    """
    Get risk-free rate from Treasury rates data for the specific trading day and expiration horizon

    Parameters:
    treasury_rates (pd.DataFrame): Treasury rates data
    trading_day (str): Trading day in MM/DD/YYYY format
    days_to_expiration (int): Days to expiration

    Returns:
    float: Annual risk-free rate for the horizon
    """
    if treasury_rates is None:
        return 0.02  # Default if no data available

    try:
        # Convert trading day to datetime
        trading_date = pd.to_datetime(trading_day, format='%m/%d/%Y')

        # Find closest date in treasury data
        closest_date_idx = treasury_rates['Date'].sub(trading_date).abs().idxmin()
        closest_date = treasury_rates.loc[closest_date_idx, 'Date']

        # If date difference is too large, use default
        if abs((closest_date - trading_date).days) > 5:
            print(f"No close Treasury data for {trading_day}, using default rate")
            return 0.02

        # Determine which rate column to use based on days to expiration
        if days_to_expiration <= 45:
            col = '1 Mo'
        elif days_to_expiration <= 75:
            col = '2 Mo'
        elif days_to_expiration <= 135:
            col = '4 Mo'
        elif days_to_expiration <= 225:
            col = '6 Mo'
        elif days_to_expiration <= 548:
            col = '1 Yr'
        else:
            col = '2 Yr'

        # Get the rate (as percentage) and convert to decimal
        rate = treasury_rates.loc[closest_date_idx, col] / 100.0

        print(f"Using {col} Treasury rate of {rate:.4f} ({rate*100:.2f}%) for {trading_day} with {days_to_expiration} days to expiration")
        return rate

    except Exception as e:
        print(f"Error getting risk-free rate: {str(e)}")
        return 0.02  # Default if error

def load_minute_data(ticker):
    """
    Load minute-level price data for a given ticker from CSV file in Google Drive

    Parameters:
    ticker (str): Ticker symbol

    Returns:
    pd.DataFrame: DataFrame with minute-level price data
    """
    # Use the path to Google Drive where your data is stored
    filepath = f"/content/drive/MyDrive/{ticker}_minute_data.csv"

    try:
        # Load the data using pandas
        df = pd.read_csv(filepath)
        df['timestamp_et'] = pd.to_datetime(df['timestamp_et'])

        print(f"Loaded {len(df)} minute records for {ticker} spanning from {df['timestamp_et'].min()} to {df['timestamp_et'].max()}")
        return df

    except Exception as e:
        print(f"Error loading minute data for {ticker}: {str(e)}")
        print("Trying alternate path...")

        try:
            # Try an alternate path
            filepath = f"{ticker}_minute_data.csv"
            df = pd.read_csv(filepath)
            df['timestamp_et'] = pd.to_datetime(df['timestamp_et'])
            print(f"Loaded {len(df)} minute records for {ticker} using alternate path")
            return df
        except Exception as e2:
            print(f"Error with alternate path: {str(e2)}")
            return pd.DataFrame()  # Return empty DataFrame on error

def calculate_n_day_returns(minute_data, n_days=7, start_year=2016):
    """
    Calculate historical n-day returns from minute data

    Parameters:
    minute_data (pd.DataFrame): Minute data with timestamp and close price
    n_days (int): Number of days for returns
    start_year (int): Start year for historical data

    Returns:
    np.array: Array of historical n-day returns
    """
    # Filter data from start_year
    minute_data_filtered = minute_data[minute_data['timestamp_et'].dt.year >= start_year].copy()

    # Get daily close prices (using all minutes, not just market hours)
    minute_data_filtered['date'] = minute_data_filtered['timestamp_et'].dt.date
    daily_closes = minute_data_filtered.groupby('date')['close'].last()

    # Calculate n-day returns
    n_day_returns = []
    dates = sorted(list(daily_closes.index))

    for i in range(0, len(dates) - n_days, n_days):
        if i+n_days < len(dates):  # Safety check
            start_price = daily_closes[dates[i]]
            end_price = daily_closes[dates[i + n_days]]
            return_val = end_price / start_price - 1
            n_day_returns.append(return_val)

    return np.array(n_day_returns)

def calculate_distribution_moments(distribution, distribution_type, min_bound=-0.5, max_bound=0.5):
    """
    Calculate moments of a distribution by only considering values between bounds.

    Args:
        distribution: Distribution object (from scipy.stats)
        distribution_type: Type of distribution ('risk_neutral' or 'historical')
        min_bound: Lower bound for moment calculation (-0.5)
        max_bound: Upper bound for moment calculation (0.5)

    Returns:
        Dictionary with distribution moments
    """
    import numpy as np
    from scipy import integrate

    # Create a truncated version of the PDF that's 0 outside our bounds
    def truncated_pdf(x):
        if min_bound <= x <= max_bound:
            return distribution.pdf(x)
        return 0

    # Calculate the normalizing constant for the truncated distribution
    norm_constant, _ = integrate.quad(truncated_pdf, min_bound, max_bound)

    # Function to calculate nth raw moment
    def calculate_moment(n):
        integrand = lambda x: x**n * truncated_pdf(x)
        moment, _ = integrate.quad(integrand, min_bound, max_bound)
        return moment / norm_constant if norm_constant > 0 else 0

    # Calculate raw moments
    mean = calculate_moment(1)
    m2 = calculate_moment(2)
    m3 = calculate_moment(3)
    m4 = calculate_moment(4)

    # Convert to central moments
    variance_raw = m2 - mean**2
    std_dev_raw = np.sqrt(variance_raw) if variance_raw > 0 else 0

    # Calculate skewness and kurtosis from central moments
    if std_dev_raw > 0:
        skewness_raw = (m3 - 3*mean*m2 + 2*mean**3) / (std_dev_raw**3)
        kurtosis_raw = (m4 - 4*mean*m3 + 6*mean**2*m2 - 3*mean**4) / (std_dev_raw**4)
    else:
        skewness_raw = 0
        kurtosis_raw = 3

    # Annualize variance and std dev using consistent formula for both distributions
    days_to_expiry = 7  # Default value for annualization scaling
    # This will be replaced with the correct value when processing actual data

    variance_annualized = variance_raw * (252.0 / days_to_expiry)  # Same formula for both distributions
    std_dev_annualized = np.sqrt(variance_annualized)  # Same formula for both distributions

    # Create prefix based on distribution type
    prefix = "rnd_" if distribution_type == "risk_neutral" else "hist_"

    result = {
        f"{prefix}mean_return": mean,
        f"{prefix}variance_raw": variance_raw,
        f"{prefix}variance_annualized": variance_annualized,
        f"{prefix}std_dev_raw": std_dev_raw,
        f"{prefix}std_dev_annualized": std_dev_annualized,
        f"{prefix}skewness_raw": skewness_raw,
        f"{prefix}kurtosis_raw": kurtosis_raw,
        f"{prefix}distribution_bounds": [min_bound, max_bound]
    }

    return result
def debug_raw_data(volatility_data):
    all_dates = list(volatility_data.keys())

    # Sort dates properly (MM/DD/YYYY format)
    def date_sort_key(date_str):
        try:
            return pd.to_datetime(date_str, format='%m/%d/%Y')
        except:
            return pd.to_datetime('1/1/1900', format='%m/%d/%Y')  # fallback for bad dates

    all_dates.sort(key=date_sort_key)

    print(f"Raw data spans from {all_dates[0]} to {all_dates[-1]}")
    print(f"Total dates in raw data: {len(all_dates)}")

    # Check January-February dates for ANY year
    jan_feb_dates = []
    for d in all_dates:
        try:
            parsed_date = pd.to_datetime(d, format='%m/%d/%Y')
            if parsed_date.month in [1, 2]:  # January or February
                jan_feb_dates.append(d)
        except:
            continue

    print(f"January-February dates: {len(jan_feb_dates)}")

    if jan_feb_dates:
        print("First 10 Jan-Feb dates:")
        for date in jan_feb_dates[:10]:
            print(f"  {date}: ", end="")
            for exp in volatility_data[date]:
                days = int(volatility_data[date][exp]['time_to_expiration'] * 365)
                print(f"{days}d ", end="")
            print()

    # Debug the target expiration filtering
    print(f"\nChecking for 6-10 day expirations in Jan-Feb:")
    target_days = [6, 7, 8, 9, 10]
    valid_jan_feb = []

    for date in jan_feb_dates:
        for exp in volatility_data[date]:
            days = int(volatility_data[date][exp]['time_to_expiration'] * 365)
            if days in target_days:
                valid_jan_feb.append((date, days))
                break

    print(f"Jan-Feb dates with 6-10 day options: {len(valid_jan_feb)}")
    if valid_jan_feb:
        print("First 10 valid Jan-Feb dates:")
        for date, days in valid_jan_feb[:10]:
            print(f"  {date}: {days}d expiration")

    # Check what get_valid_trading_days would return
    print(f"\nTesting get_valid_trading_days function:")
    valid_days = get_valid_trading_days(volatility_data, target_days, 0)
    valid_jan_feb_from_function = [d for d in valid_days if pd.to_datetime(d, format='%m/%d/%Y').month in [1, 2]]
    print(f"get_valid_trading_days returns {len(valid_jan_feb_from_function)} Jan-Feb dates")

    if len(valid_jan_feb) != len(valid_jan_feb_from_function):
        print(f"ERROR: Raw data has {len(valid_jan_feb)} valid Jan-Feb dates, but function returns {len(valid_jan_feb_from_function)}")
        print("This is likely the source of your March 3rd filtering bug!")
def fit_smoother_historical_distribution(returns, min_bound=-0.5, max_bound=0.5):
    """
    Fit a smoother distribution to historical returns that prioritizes
    natural curvature over perfect statistical fit.

    Parameters:
    returns (np.array): Array of returns
    min_bound (float): Minimum bound for moment calculation (-0.5)
    max_bound (float): Maximum bound for moment calculation (0.5)

    Returns:
    dict: Distribution parameters and moments
    """
    import numpy as np
    from scipy import stats

    # First try fitting a t-distribution with reasonable degrees of freedom
    # This generally produces smoother curves than trying to optimize AIC
    try:
        # Start with reasonable degrees of freedom (5-15 typically gives smoother curves)
        # rather than allowing extremely low values that can create sharp peaks
        t_params = list(stats.t.fit(returns, f0=10))

        # Ensure degrees of freedom parameter isn't too low (which causes sharp peaks)
        if t_params[0] < 4:
            t_params[0] = 4  # Force minimum df=4 for smoother curves

        # Create the distribution with adjusted parameters
        t_dist = stats.t(*t_params)

        # Calculate moments within bounds
        t_moments = calculate_distribution_moments(t_dist, "historical", min_bound, max_bound)

        result = {
            'distribution': 't',
            'params': t_params,
            'returns': returns,
            'smoothing_applied': True
        }

        # Add the bounded moments
        result.update(t_moments)

        return result

    except Exception as e:
        print(f"Error fitting smooth t-distribution: {e}")

        # Fall back to normal distribution which is naturally smooth
        try:
            norm_params = stats.norm.fit(returns)
            norm_dist = stats.norm(*norm_params)

            # Calculate moments
            norm_moments = calculate_distribution_moments(norm_dist, "historical", min_bound, max_bound)

            result = {
                'distribution': 'norm',
                'params': norm_params,
                'returns': returns,
                'smoothing_applied': True
            }

            # Add the moments
            result.update(norm_moments)

            return result

        except Exception as e2:
            print(f"Error fitting smooth normal distribution: {e2}")
            # Last resort - try the original method but with bounded moments
            return fit_historical_distribution(returns, min_bound, max_bound)

def fit_historical_distribution(returns, min_bound=-0.5, max_bound=0.5):
    """
    Fit best distribution to historical returns and calculate moments
    within the specified bounds.

    Parameters:
    returns (np.array): Array of returns
    min_bound (float): Minimum bound for moment calculation (-0.5)
    max_bound (float): Maximum bound for moment calculation (0.5)

    Returns:
    dict: Best fit distribution parameters and moments
    """
    # Define distributions to try
    distributions = ['norm', 'skewnorm', 't', 'laplace']
    best_aic = np.inf
    best_dist = None
    best_params = None

    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            # Fit distribution
            params = dist.fit(returns)

            # Calculate log-likelihood
            loglik = np.sum(dist.logpdf(returns, *params))

            # Calculate AIC
            k = len(params)
            aic = 2 * k - 2 * loglik

            if aic < best_aic:
                best_aic = aic
                best_dist = dist_name
                best_params = params
        except Exception as e:
            print(f"Error fitting {dist_name} distribution: {e}")
            continue

    # If no distribution fits, default to normal
    if best_dist is None:
        best_dist = 'norm'
        best_params = stats.norm.fit(returns)

    # Create the distribution object
    dist = getattr(stats, best_dist)(*best_params)

    # Calculate moments within the specified bounds
    dist_moments = calculate_distribution_moments(dist, "historical", min_bound, max_bound)

    # Add additional info to the result
    result = {
        'distribution': best_dist,
        'params': best_params,
        'returns': returns,
        'aic': best_aic
    }

    # Add all moments calculated with bounds
    result.update(dist_moments)

    return result

def plot_historical_distribution_fit(historical_returns, hist_distribution, ticker, trading_day, expiration_days, save_dir=None):
    """
    Plot histogram of historical returns with the fitted distribution curve.

    Parameters:
    historical_returns (np.array): Raw historical returns
    hist_distribution (dict): Dictionary with fitted distribution parameters
    ticker (str): Ticker symbol
    trading_day (str): Trading day in MM/DD/YYYY format
    expiration_days (int): Days to expiration
    save_dir (str): Directory to save the plot

    Returns:
    plt.Figure: Figure object for the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    # Get distribution info
    dist_name = hist_distribution['distribution']
    params = hist_distribution['params']
    dist = getattr(stats, dist_name)

    # Create histogram of actual returns
    n_bins = max(30, min(100, int(len(historical_returns)/10)))
    _, bins, _ = ax.hist(historical_returns, bins=n_bins, alpha=0.6, density=True,
                        color='lightblue', label=f'Historical Returns (n={len(historical_returns)})')

    # Create x values for plotting the fitted distribution
    x = np.linspace(min(historical_returns), max(historical_returns), 1000)

    # Plot the fitted distribution curve
    pdf = dist.pdf(x, *params)
    ax.plot(x, pdf, 'r-', linewidth=2,
            label=f'Fitted {dist_name.capitalize()} Distribution')

    # Add distribution parameters to the plot
    param_text = f"Distribution: {dist_name.capitalize()}\n"
    param_names = {
        't': ['df (degrees of freedom)', 'loc', 'scale'],
        'norm': ['loc (mean)', 'scale (std)'],
        'skewnorm': ['a (skew)', 'loc', 'scale'],
        'laplace': ['loc', 'scale']
    }

    if dist_name in param_names:
        for i, name in enumerate(param_names[dist_name]):
            if i < len(params):
                param_text += f"{name}: {params[i]:.4f}\n"

    # Add distribution statistics
    stats_text = (
        f"Statistics within bounds:\n"
        f"Mean: {hist_distribution['hist_mean_return']:.4f}\n"
        f"Std Dev: {hist_distribution['hist_std_dev_raw']:.4f}\n"
        f"Skewness: {hist_distribution['hist_skewness_raw']:.4f}\n"
        f"Kurtosis: {hist_distribution['hist_kurtosis_raw']:.4f}\n"
        f"Bounds: {hist_distribution['hist_distribution_bounds']}"
    )

    # Place text boxes
    ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Set labels and title
    ax.set_xlabel('Return (S_t/S_0 - 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')

    title = f'{ticker}: Fitted Historical Distribution\n'
    title += f'Trading Day: {trading_day}, Expiration: {expiration_days} days'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(loc='upper left')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add footnote
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.01, f"Generated: {timestamp} | User: {USERNAME}",
             fontsize=8, color='black', ha='center')

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = f"{save_dir}/{ticker}_{trading_day.replace('/', '_')}_{expiration_days}d_hist_fit_{timestamp_str}.png"
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Saved historical distribution fit plot to {fig_path}")

    return fig

def calculate_realized_moments_comprehensive(minute_data, trading_day_dt, expiration_day_dt, include_overnight=True):
    """
    Calculate realized moments using all available minute data, properly handling overnight returns.

    Parameters:
    minute_data: DataFrame with minute price data
    trading_day_dt: Trading day datetime
    expiration_day_dt: Expiration day datetime
    include_overnight: Whether to include overnight returns in the calculation

    Returns:
    dict: Dictionary with realized moments
    """
    # Filter data between trading day and expiration
    period_data = minute_data[
        (minute_data['timestamp_et'] >= trading_day_dt) &
        (minute_data['timestamp_et'] <= expiration_day_dt)
    ]

    # Check if we have any data for this period
    if len(period_data) == 0:
        print(f"No minute data found between {trading_day_dt.date()} and {expiration_day_dt.date()}")
        return None

    # Sort by timestamp
    period_data = period_data.sort_values('timestamp_et')

    # Add date column for day identification
    period_data['date'] = period_data['timestamp_et'].dt.date

    # Calculate overall period return
    start_price = period_data['close'].iloc[0]
    end_price = period_data['close'].iloc[-1]
    period_return = end_price / start_price - 1

    # List to store all returns (intraday and overnight)
    all_returns = []

    # Dictionary to track last price of each day
    last_prices = {}
    first_prices = {}

    # Get unique dates
    unique_dates = sorted(period_data['date'].unique())

    # Calculate intraday returns for each day AND track close prices
    for date in unique_dates:
        day_data = period_data[period_data['date'] == date]

        # Store first and last price for each day
        first_prices[date] = day_data['close'].iloc[0]
        last_prices[date] = day_data['close'].iloc[-1]

        # Calculate intraday returns (between consecutive minutes)
        day_returns = day_data['close'].pct_change().dropna().values
        all_returns.extend(day_returns)

    # Add overnight returns if requested
    overnight_returns = []
    if include_overnight:
        for i in range(len(unique_dates) - 1):
            current_date = unique_dates[i]
            next_date = unique_dates[i + 1]

            # Calculate overnight return
            overnight_return = (first_prices[next_date] / last_prices[current_date]) - 1
            overnight_returns.append(overnight_return)
            all_returns.append(overnight_return)

    # Convert to numpy array
    returns_array = np.array(all_returns)

    # Calculate variance as sum of squared returns (no mean centering)
    realized_variance = np.sum(returns_array**2)
    realized_std = np.sqrt(realized_variance)

    # Count trading days observed
    days_observed = len(unique_dates)

    # Calculate higher moments
    if len(returns_array) > 0 and realized_std > 0:
        # Calculate mean return
        mean_return = np.mean(returns_array)

        # Calculate skewness
        m3 = np.mean((returns_array - mean_return)**3)
        realized_skewness = m3 / (realized_std**3)

        # Calculate kurtosis
        m4 = np.mean((returns_array - mean_return)**4)
        realized_kurtosis = m4 / (realized_std**4)
    else:
        realized_skewness = 0
        realized_kurtosis = 3  # Normal distribution kurtosis

    # Print diagnostics
    print(f"Realized moment calculation diagnostics:")
    print(f"  - Total minutes with returns: {len(returns_array)}")
    print(f"  - Intraday returns: {len(returns_array) - len(overnight_returns)}")
    print(f"  - Overnight returns: {len(overnight_returns)}")
    print(f"  - Sum of squared returns: {realized_variance:.6f}")
    print(f"  - Realized volatility: {realized_std:.6f}")
    print(f"  - Days observed: {days_observed}")

    # Get days between trading and expiration for consistent annualization
    days_to_expiry = (expiration_day_dt.date() - trading_day_dt.date()).days

    # Annualize variance using consistent method for both implied and realized
    # Use the same formula as for implied variance: variance * (252/days_to_expiry)
    annualized_variance = realized_variance * (252.0 / days_to_expiry)
    annualized_vol = np.sqrt(annualized_variance)

    # Also calculate traditional realized variance annualization for reference
    traditional_annualized_variance = realized_variance * (252 * 390) / len(returns_array)
    traditional_annualized_vol = np.sqrt(traditional_annualized_variance)

    return {
        'period_return': period_return,
        'realized_variance': realized_variance,
        'realized_std': realized_std,
        'realized_skewness': realized_skewness,
        'realized_kurtosis': realized_kurtosis,
        'annualized_variance': annualized_variance,
        'annualized_vol': annualized_vol,
        'traditional_annualized_variance': traditional_annualized_variance,
        'traditional_annualized_vol': traditional_annualized_vol,
        'n_days': days_observed,
        'days_to_expiry': days_to_expiry,
        'n_minutes': len(returns_array) - len(overnight_returns),
        'n_overnight': len(overnight_returns),
        'n_total_returns': len(returns_array)
    }

def resample_to_five_min(minute_data):
    """
    Resample minute-level data to 5-minute bars and compute log returns.
    """
    # Create a copy to avoid modifying the original dataframe
    data = minute_data.copy()
    data['timestamp_et'] = pd.to_datetime(data['timestamp_et'])
    data = data.set_index('timestamp_et')

    # Resample to 5-minute intervals (last price in each bin)
    data_5m = data['close'].resample('5T').last().dropna().to_frame()

    # Calculate 5-minute log returns
    data_5m['log_return'] = np.log(data_5m['close'] / data_5m['close'].shift(1))
    data_5m = data_5m.dropna().reset_index()

    return data_5m

def calculate_realized_higher_moments_amaya(minute_data, trading_day_dt, expiration_day_dt, row_dict=None):
    """
    Calculate realized skewness and kurtosis using Amaya et al. (2015),
    using only intraday 5-min data (9:30-16:00) across the window.

    Parameters:
    minute_data (pd.DataFrame): Minute-level price data
    trading_day_dt (datetime): Trading day datetime
    expiration_day_dt (datetime): Expiration day datetime
    row_dict (dict): Optional dictionary with original moments

    Returns:
    dict: Dictionary with realized variance, skewness, and kurtosis
    """
    # First resample to 5-minute bars
    five_min_data = resample_to_five_min(minute_data)

    # Filter for data within the date range
    mask = (five_min_data['timestamp_et'] >= trading_day_dt) & (five_min_data['timestamp_et'] <= expiration_day_dt)
    period_data = five_min_data[mask].copy()

    if len(period_data) == 0:
        print(f"No 5-minute data found between {trading_day_dt.date()} and {expiration_day_dt.date()}")
        return None

    # Restrict to intraday times (9:30 AM to 4:00 PM)
    period_data['time'] = period_data['timestamp_et'].dt.time
    period_data = period_data[
        (period_data['time'] >= pd.to_datetime('09:30').time()) &
        (period_data['time'] <= pd.to_datetime('16:00').time())
    ]

    if len(period_data) == 0:
        print(f"No intraday (9:30–16:00) data from {trading_day_dt.date()} to {expiration_day_dt.date()}")
        return None

    # Get log returns
    log_returns = period_data['log_return'].values

    # Count the number of observations
    N = len(log_returns)

    if N < 10:
        print(f"Insufficient intraday 5-min returns ({N}) for calculating moments")
        return None

    # Calculate realized variance (sum of squared returns)
    realized_variance = np.sum(log_returns**2)

    if realized_variance > 0:
        # Apply proper scaling factors as specified in Amaya et al. (2015)
        realized_skewness = np.sqrt(N) * np.sum(log_returns**3) / (realized_variance**(3/2))
        realized_kurtosis = N * np.sum(log_returns**4) / (realized_variance**2)
    else:
        realized_skewness = 0
        realized_kurtosis = 3  # Normal distribution kurtosis

    # Get original values if available (for comparison)
    original_skew = 0
    original_kurt = 3
    if row_dict:
        original_skew = row_dict.get('original_realized_skewness', row_dict.get('realized_skewness', 0))
        original_kurt = row_dict.get('original_realized_kurtosis', row_dict.get('realized_kurtosis', 3))

    # Calculate days until expiration
    days_until_expiration = (expiration_day_dt.date() - trading_day_dt.date()).days

    print(f"Amaya realized moments (intraday 5-min only) from {trading_day_dt.date()} to {expiration_day_dt.date()}:")
    print(f"  Intraday 5-min obs: {N}")
    print(f"  Realized variance: {realized_variance:.6f}")
    print(f"  Realized skewness: {realized_skewness:.4f} (with sqrt(N) scaling)")
    print(f"  Realized kurtosis: {realized_kurtosis:.4f} (with N scaling)")

    return {
        'days_observed': days_until_expiration,
        'days_until_expiration': days_until_expiration,
        'total_observations': N,
        'realized_variance': realized_variance,
        'realized_skewness_raw': realized_skewness,
        'realized_kurtosis_raw': realized_kurtosis,
        'original_skewness': original_skew,
        'original_kurtosis': original_kurt,
        'realized_skewness': realized_skewness,  # Using Amaya's formula with sqrt(N)
        'realized_kurtosis': realized_kurtosis,  # Using Amaya's formula with N
        'amaya_sqrt_N': np.sqrt(N),
        'amaya_N': N,
        'data_source': '5-minute intraday'
    }

def calculate_realized_higher_moments_amaya_wrapper():
    """
    Wrapper function to calculate realized skewness and kurtosis using
    the Amaya et al. (2015) formulas and compare with implied moments.
    Uses 5-minute intraday returns for improved accuracy.
    """
    # Path to where ticker results are stored
    base_dir = '/content/drive/MyDrive'

    # Get list of tickers with saved data
    ticker_options = []
    for ticker in valid_tickers:
        results_dir = f'{base_dir}/{ticker}_analysis_results'

        # Check if directory exists
        if os.path.exists(results_dir):
            # Check for standard moments file
            if os.path.exists(f"{results_dir}/{ticker}_moments_summary.csv"):
                ticker_options.append((ticker, f"{results_dir}/{ticker}_moments_summary.csv", "all expirations"))

            # Check for expiration-specific files
            exp_dir = f"{results_dir}/expiration_specific"
            if os.path.exists(exp_dir):
                # Look for specific day files
                for day_length in [6, 7, 8, 9, 10]:
                    exp_file = f"{exp_dir}/{ticker}_{day_length}day_options_summary.csv"
                    if os.path.exists(exp_file):
                        ticker_options.append((ticker, exp_file, f"{day_length}-day"))

    if not ticker_options:
        print("No ticker data found to analyze with Amaya formulas.")
        return

    # Display available tickers for analysis
    print("\nAvailable tickers for Amaya moments calculation:")
    for i, (ticker, file_path, expiry_info) in enumerate(ticker_options):
        print(f"{i+1}. {ticker} ({expiry_info})")

    # Get user selection
    selection = input("\nSelect a ticker to analyze (number) or 'all' for all tickers: ")

    tickers_to_process = []
    if selection.lower() == 'all':
        tickers_to_process = ticker_options
    else:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(ticker_options):
                tickers_to_process = [ticker_options[idx]]
            else:
                print("Invalid selection. Please try again.")
                return
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")
            return

    # Process each selected ticker
    for ticker, csv_file, expiry_info in tickers_to_process:
        print(ticker, expiry_info)
        print(f"\nProcessing {ticker} ({expiry_info})...")

        # Determine time series directory
        if "all expirations" in expiry_info:
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/time_series"
            day_length = None
        else:
            # Extract days from expiry_info
            day_length = int(expiry_info.split("-")[0])
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific/{day_length}day_time_series"

        # Create time series directory if it doesn't exist
        os.makedirs(time_series_dir, exist_ok=True)

        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: Moments CSV file not found: {csv_file}")
            continue

        # Load minute data for the ticker
        try:
            minute_data = load_minute_data(ticker)
            if len(minute_data) == 0:
                print(f"Error: No minute data available for {ticker}. Skipping analysis.")
                continue

            print(f"Successfully loaded {len(minute_data)} minute records")
        except Exception as e:
            print(f"Error loading minute data for {ticker}: {str(e)}")
            continue

        try:
            # Load the moments data
            moments_df = pd.read_csv(csv_file)
            print(f"Loaded {len(moments_df)} rows from {csv_file}")

            # Make a backup of the original data
            backup_csv = csv_file.replace(".csv", "_original.csv")
            if not os.path.exists(backup_csv):
                moments_df.to_csv(backup_csv, index=False)
                print(f"Created backup at {backup_csv}")

            # Convert trading_day to datetime
            moments_df['trading_day_dt'] = pd.to_datetime(moments_df['trading_day'], format='%m/%d/%Y')

            # Get expiration days - either from the day_length or from the data
            if day_length:
                moments_df['expiration_days'] = day_length
            else:
                # Find expiration days column
                if 'expiration_days' not in moments_df.columns and 'time_to_expiration_years' in moments_df.columns:
                    moments_df['expiration_days'] = moments_df['time_to_expiration_years'] * 365
                elif 'days_to_expiry' in moments_df.columns:
                    moments_df['expiration_days'] = moments_df['days_to_expiry']

            # Calculate expiration dates
            moments_df['expiration_day_dt'] = moments_df['trading_day_dt'] + pd.to_timedelta(moments_df['expiration_days'], unit='D')

            # Recalculate realized moments using Amaya formulas with 5-minute data
            print("\nCalculating realized moments using Amaya et al. (2015) formulas with 5-minute intraday returns...")

            updated_moments = []
            for i, row in tqdm(moments_df.iterrows(), total=len(moments_df), desc="Processing days"):
                # Calculate the Amaya higher moments with 5-min data
                amaya_moments = calculate_realized_higher_moments_amaya(
                    minute_data,
                    row['trading_day_dt'],
                    row['expiration_day_dt'],
                    row.to_dict()
                )

                if amaya_moments:
                    # Update with new moments
                    row_dict = row.to_dict()

                    # Store original values for comparison
                    row_dict['original_realized_skewness'] = row_dict.get('realized_skewness', 0)
                    row_dict['original_realized_kurtosis'] = row_dict.get('realized_kurtosis', 3)

                    # Update with Amaya moments
                    row_dict['realized_skewness'] = amaya_moments['realized_skewness']
                    row_dict['realized_kurtosis'] = amaya_moments['realized_kurtosis']
                    row_dict['amaya_days_observed'] = amaya_moments['days_observed']
                    row_dict['amaya_observations'] = amaya_moments['total_observations']
                    row_dict['amaya_method'] = '5-min intraday'

                    updated_moments.append(row_dict)
                else:
                    # Keep original row if can't calculate new moments
                    updated_moments.append(row.to_dict())

            # Convert back to DataFrame
            updated_df = pd.DataFrame(updated_moments)

            # Sort by trading day for time series
            updated_df = updated_df.sort_values('trading_day_dt')

            # Save the updated data
            updated_csv = csv_file.replace(".csv", "_amaya_5min.csv")
            updated_df.to_csv(updated_csv, index=False)
            print(f"Saved updated moments with Amaya 5-min calculations to {updated_csv}")

            # Create plots comparing original and new moments, but DO NOT remove outliers
            print("\nCreating comparison plots (with all data points, no outlier removal)...")

            # Plot skewness comparison
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # Plot original realized skewness
            if 'original_realized_skewness' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['original_realized_skewness'], 'b--',
                        alpha=0.5, linewidth=1.5, label='Original Realized Skewness')

            # Plot new Amaya realized skewness - NO outlier removal as requested
            ax.plot(updated_df['trading_day_dt'], updated_df['realized_skewness'], 'b-',
                    linewidth=2, label='Amaya 5-Min Realized Skewness')

            # Plot implied skewness for comparison - NO outlier removal as requested
            if 'rnd_skewness' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['rnd_skewness'], 'r-',
                        linewidth=2, label='Implied (RND) Skewness')

            # Add reference line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Set title and labels
            title = f'{ticker}: Skewness Comparison with 5-Min Amaya Method'
            if day_length:
                title += f' (Exactly {day_length}-day Expiration)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Skewness', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates
            fig.autofmt_xdate()

            # Add metadata
            days_note = f"Exactly {day_length}-day expiration" if day_length else "Multiple expirations"
            fig.text(0.5, 0.01,
                    f"Generated: {TIMESTAMP} | User: {USERNAME} | {days_note} | Using Amaya et al. (2015) 5-min intraday formulas",
                    fontsize=9, ha='center')

            # Save figure
            skew_fig_path = f"{time_series_dir}/{ticker}_skewness_amaya_5min_comparison.png"
            plt.savefig(skew_fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved skewness comparison to {skew_fig_path}")
            plt.close(fig)

            # Plot kurtosis comparison
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # Plot original realized kurtosis
            if 'original_realized_kurtosis' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['original_realized_kurtosis'], 'b--',
                        alpha=0.5, linewidth=1.5, label='Original Realized Kurtosis')

            # Plot new Amaya realized kurtosis - NO outlier removal as requested
            ax.plot(updated_df['trading_day_dt'], updated_df['realized_kurtosis'], 'b-',
                    linewidth=2, label='Amaya 5-Min Realized Kurtosis')

            # Plot implied kurtosis for comparison - NO outlier removal as requested
            if 'rnd_kurtosis' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['rnd_kurtosis'], 'r-',
                        linewidth=2, label='Implied (RND) Kurtosis')

            # Add reference line at normal kurtosis (3)
            ax.axhline(y=3, color='black', linestyle='-', alpha=0.3, label='Normal Kurtosis (3)')

            # Set title and labels
            title = f'{ticker}: Kurtosis Comparison with 5-Min Amaya Method'
            if day_length:
                title += f' (Exactly {day_length}-day Expiration)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Kurtosis', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates
            fig.autofmt_xdate()

            # Add metadata
            fig.text(0.5, 0.01,
                    f"Generated: {TIMESTAMP} | User: {USERNAME} | {days_note} | Using Amaya et al. (2015) 5-min intraday formulas",
                    fontsize=9, ha='center')

            # Save figure
            kurt_fig_path = f"{time_series_dir}/{ticker}_kurtosis_amaya_5min_comparison.png"
            plt.savefig(kurt_fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved kurtosis comparison to {kurt_fig_path}")
            plt.close(fig)

            # Ask if user wants to update the original file
            update_choice = input(f"\nReplace original moments data with new Amaya 5-min calculations for {ticker}? (y/n): ")
            if update_choice.lower() == 'y':
                # Save to original CSV
                updated_df.to_csv(csv_file, index=False)
                print(f"Updated {csv_file} with Amaya 5-min realized moments")

                # Regenerate all time series plots
                if day_length:
                    print("\nRegenerating time series plots with updated moments...")
                    plot_skewness_kurtosis_time_series(
                        updated_df.to_dict('records'),
                        f"{ticker} ({day_length}-day)",
                        save_dir=time_series_dir
                    )

                    # Also create plot specifically comparing original vs Amaya moments
                    plot_amaya_comparison(
                        updated_df.to_dict('records'),
                        ticker,
                        day_length,
                        save_dir=time_series_dir
                    )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAmaya 5-minute intraday moments calculation completed.")
    print("You can use the generated plots to analyze differences between original and Amaya-based higher moments.")

def calculate_rnd_statistics(return_range, normalized_rnd, time_to_expiration_years,
                           min_bound=-0.5, max_bound=0.5, density_threshold=0.0001):
    """
    Calculate statistics for RND, using dynamic bounds determined by density threshold.

    Parameters:
    return_range: Array of return values
    normalized_rnd: Array of normalized RND values corresponding to return_range
    time_to_expiration_years: Time to expiration in years
    min_bound: Minimum possible return bound (fallback/safety value)
    max_bound: Maximum possible return bound (fallback/safety value)
    density_threshold: Minimum density threshold to determine bounds dynamically

    Returns:
    dict: Dictionary with RND statistics
    """
    try:
        # Ensure RND is normalized
        rnd_integral = np.trapz(normalized_rnd, return_range)
        if abs(rnd_integral - 1.0) > 0.05:  # If not normalized within 5%
            if rnd_integral > 0:
                normalized_rnd = normalized_rnd / rnd_integral

        # Find index closest to x=0 (at-the-money)
        atm_index = np.argmin(np.abs(return_range))

        # Dynamic bounds determination
        # Start from ATM and move right until density falls below threshold
        right_bound_index = atm_index
        while (right_bound_index < len(return_range) - 1 and
               normalized_rnd[right_bound_index] > density_threshold):
            right_bound_index += 1

        # Start from ATM and move left until density falls below threshold
        left_bound_index = atm_index
        while (left_bound_index > 0 and
               normalized_rnd[left_bound_index] > density_threshold):
            left_bound_index -= 1

        # Get the corresponding return values for these bounds
        dynamic_min_bound = return_range[left_bound_index]
        dynamic_max_bound = return_range[right_bound_index]

        # Apply safety check - don't exceed the original fallback bounds if specified
        dynamic_min_bound = max(dynamic_min_bound, min_bound)
        dynamic_max_bound = min(dynamic_max_bound, max_bound)

        # Print information about the dynamic bounds found
        print(f"Dynamic RND bounds: [{dynamic_min_bound:.4f}, {dynamic_max_bound:.4f}] with threshold {density_threshold}")

        # Apply bounds to focus on meaningful part of distribution
        bounds_mask = (return_range >= dynamic_min_bound) & (return_range <= dynamic_max_bound)
        bounded_returns = return_range[bounds_mask]  # Returns within dynamic bounds
        bounded_rnd = normalized_rnd[bounds_mask]    # RND values within dynamic bounds

        # Normalize the bounded RND to integrate to 1.0
        bounded_integral = np.trapz(bounded_rnd, bounded_returns)
        if bounded_integral > 0:
            bounded_rnd = bounded_rnd / bounded_integral

        # Check if we have enough data points
        if len(bounded_returns) >= 3:
            # Calculate statistics weighted by probability
            rnd_mean = np.average(bounded_returns, weights=bounded_rnd)
            rnd_var = np.average((bounded_returns - rnd_mean)**2, weights=bounded_rnd)
            rnd_std = np.sqrt(rnd_var)

            # Calculate skewness and kurtosis
            m3 = np.average((bounded_returns - rnd_mean)**3, weights=bounded_rnd)
            m4 = np.average((bounded_returns - rnd_mean)**4, weights=bounded_rnd)

            # Calculate skewness and kurtosis from moments
            rnd_skewness = m3 / (rnd_std**3) if rnd_std > 0 else 0
            rnd_kurt = (m4 / (rnd_std**4)) if rnd_std > 0 else 0

            # Get days to expiration for consistent annualization
            days_to_expiry = time_to_expiration_years * 365

            # Annualized RND volatility using consistent method with realized volatility
            # Use the same formula: variance * (252/days_to_expiry)
            rnd_annualized_var = rnd_var * (252.0 / days_to_expiry)
            rnd_annualized_vol = np.sqrt(rnd_annualized_var)

            # Also calculate traditional annualization for reference
            traditional_annualized_vol = rnd_std / np.sqrt(time_to_expiration_years)

            return {
                'mean': rnd_mean,
                'mean_pct': rnd_mean * 100,
                'variance': rnd_var,
                'std_dev': rnd_std,
                'annualized_variance': rnd_annualized_var,
                'annualized_vol': rnd_annualized_vol,
                'traditional_annualized_vol': traditional_annualized_vol,
                'days_to_expiry': days_to_expiry,
                'skewness': rnd_skewness,
                'kurtosis': rnd_kurt,
                'bounds_used': [dynamic_min_bound, dynamic_max_bound],
                'density_threshold': density_threshold
            }
        else:
            return {
                'mean': np.nan,
                'mean_pct': np.nan,
                'variance': np.nan,
                'std_dev': np.nan,
                'annualized_variance': np.nan,
                'annualized_vol': np.nan,
                'traditional_annualized_vol': np.nan,
                'days_to_expiry': time_to_expiration_years * 365,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'bounds_used': [dynamic_min_bound, dynamic_max_bound],
                'density_threshold': density_threshold,
                'error': 'Insufficient data points within bounds'
            }
    except Exception as e:
        print(f"Error calculating RND statistics: {str(e)}")
        return {
            'mean': np.nan,
            'mean_pct': np.nan,
            'variance': np.nan,
            'std_dev': np.nan,
            'annualized_variance': np.nan,
            'annualized_vol': np.nan,
            'traditional_annualized_vol': np.nan,
            'days_to_expiry': time_to_expiration_years * 365,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'bounds_used': [min_bound, max_bound],
            'density_threshold': density_threshold,
            'error': str(e)
        }

def get_actual_period_return(ticker, minute_data, trading_day, days_to_expiration):
    """
    Calculate the actual return realized over the forecasting period

    Parameters:
    ticker (str): Stock ticker symbol
    minute_data (pd.DataFrame): Minute data with timestamp and close prices
    trading_day (str): Trading day in MM/DD/YYYY format
    days_to_expiration (int): Days to expiration

    Returns:
    float: Realized return over the period
    """
    try:
        # Convert trading day string to datetime
        trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
        expiration_day_dt = trading_day_dt + timedelta(days=days_to_expiration)

        # Filter minute data to get the trading day's close price
        # Get last price on trading day
        trading_day_data = minute_data[minute_data['timestamp_et'].dt.date == trading_day_dt.date()]
        if len(trading_day_data) == 0:
            print(f"No data for trading day {trading_day}")
            return 0.0

        start_price = trading_day_data['close'].iloc[-1]

        # Get the last price on or before expiration day
        # Filter for days up to expiration
        expiration_data = minute_data[
            (minute_data['timestamp_et'].dt.date <= expiration_day_dt.date()) &
            (minute_data['timestamp_et'].dt.date > trading_day_dt.date())
        ]

        if len(expiration_data) == 0:
            print(f"No data between {trading_day} and expiration")
            return 0.0

        # Get the last available price
        end_price = expiration_data['close'].iloc[-1]

        # Calculate return
        realized_return = end_price / start_price - 1.0

        print(f"Realized return from {trading_day} to {expiration_day_dt.strftime('%m/%d/%Y')}: {realized_return:.4f} ({realized_return*100:.2f}%)")
        return realized_return

    except Exception as e:
        print(f"Error calculating actual period return: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0  # Default to zero return on error

def compare_rnd_to_realized_comprehensive(volatility_data, trading_day, ticker, minute_data,
                                         treasury_rates=None, x_range_limits=(-0.5, 0.5),
                                         include_overnight=True, show_plot=True, save_results=True):
    """
    Compare risk-neutral density to realized minute returns moments with all improvements:
    - Uses all minute data with proper overnight handling
    - Accurately calculates realized moments
    - Shows full distribution heights
    - Applies bounds to moment calculations
    - Supports time series generation for specific expiration periods
    - Uses consistent annualization between implied and realized
    - Creates separate plots for historical distribution fitting

    Parameters:
    volatility_data: Volatility data dictionary
    trading_day: Trading day string (MM/DD/YYYY)
    ticker: Stock ticker
    minute_data: DataFrame with minute price data
    treasury_rates: DataFrame with treasury rates
    x_range_limits: Limits for return range
    include_overnight: Whether to include overnight returns
    show_plot: Whether to show plot
    save_results: Whether to save results

    Returns:
    tuple: (fig, ax, stats_dict)
    """
    try:
        # Get the first expiration for this trading day
        first_expiration = next(iter(volatility_data[trading_day]))

        # Get time to expiration in years and days
        time_to_expiration_years = volatility_data[trading_day][first_expiration]['time_to_expiration']
        time_to_expiration_days = int(time_to_expiration_years * 365)

        # Get risk-free rate
        risk_free_rate = 0.02  # Default
        if treasury_rates is not None:
            risk_free_rate = get_risk_free_rate(treasury_rates, trading_day, time_to_expiration_days)

        print(f"Trading Day: {trading_day}")
        print(f"Time to Expiration: {time_to_expiration_years:.4f} years ({time_to_expiration_days} days)")

        # Convert dates
        trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
        expiration_day_dt = trading_day_dt + timedelta(days=time_to_expiration_days)

        # Calculate realized moments using the comprehensive method
        realized_moments = calculate_realized_moments_comprehensive(
            minute_data, trading_day_dt, expiration_day_dt, include_overnight)

        if realized_moments is None:
            print(f"Insufficient minute data between {trading_day} and expiration")
            return None

        # Calculate historical returns for the same horizon
        historical_minute_data = minute_data[minute_data['timestamp_et'] < trading_day_dt].copy()
        historical_returns = calculate_n_day_returns(
            historical_minute_data,
            n_days=time_to_expiration_days,
            start_year=2016
        )

        # Fit historical distribution with smoothness prioritized
        if len(historical_returns) > 10:
            hist_distribution = fit_smoother_historical_distribution(
                historical_returns,
                min_bound=x_range_limits[0],
                max_bound=x_range_limits[1]
            )
            print(f"Fitted smoother {hist_distribution['distribution']} distribution to {len(historical_returns)} historical returns")
            print(f"Historical moments calculated within bounds: [{x_range_limits[0]}, {x_range_limits[1]}]")

            # Create and save a plot of the fitted distribution
            if save_results:
                hist_plots_dir = f'/content/drive/MyDrive/{ticker}_analysis_results/historical_fits'
                os.makedirs(hist_plots_dir, exist_ok=True)

                # Plot the historical distribution fit
                plot_historical_distribution_fit(
                    historical_returns,
                    hist_distribution,
                    ticker,
                    trading_day,
                    time_to_expiration_days,
                    save_dir=hist_plots_dir
                )
        else:
            hist_distribution = None
            print(f"Insufficient historical returns to fit distribution ({len(historical_returns)} samples)")

        # Get RND data
        strike_range = np.array(volatility_data[trading_day][first_expiration]['strike_range'])
        underlying_price = volatility_data[trading_day][first_expiration]['underlyingPrice']

        # Use quartic RND as specified
        density_key = 'risk_neutral_density_quartic'
        if density_key in volatility_data[trading_day][first_expiration]:
            price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
        else:
            available_keys = [k for k in volatility_data[trading_day][first_expiration].keys()
                            if 'density' in k and 'quartic' in k]
            if available_keys:
                density_key = available_keys[0]
                price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                print(f"Using alternative quartic RND: {density_key}")
            else:
                print("No quartic RND found, using standard RND")
                density_key = 'risk_neutral_density'
                if density_key in volatility_data[trading_day][first_expiration]:
                    price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                else:
                    print("No standard RND found either, please check data")
                    return None

        # Create proper strike range for the RND
        rnd_strikes = np.linspace(min(strike_range), max(strike_range), len(price_rnd))

        # Convert strike range to returns (S_t/S_0 - 1)
        return_range = rnd_strikes / underlying_price - 1.0

        # Transform the RND from price space to return space using Jacobian
        rnd = price_rnd * underlying_price

        # Normalize RND to area of 1
        total_rnd_area = np.trapz(rnd, return_range)
        if total_rnd_area > 0:
            normalized_rnd = rnd / total_rnd_area
        else:
            normalized_rnd = rnd

        # Calculate RND statistics in return space with dynamic bounds
        rnd_stats = calculate_rnd_statistics(return_range, normalized_rnd, time_to_expiration_years,
                                           x_range_limits[0], x_range_limits[1], density_threshold=0.0001)

        # Create figure for comparison with full height distributions (APA-compliant)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, dpi=300)

        # Plot RND on ax1 - preserve full height
        ax1.plot(return_range, normalized_rnd, 'red', linewidth=2,
                 label=f'Risk-Neutral Density ({density_key})')

        # Highlight the RND dynamic bounds
        if 'bounds_used' in rnd_stats:
            ax1.axvline(x=rnd_stats['bounds_used'][0], color='red', linestyle='--', alpha=0.5)
            ax1.axvline(x=rnd_stats['bounds_used'][1], color='red', linestyle='--', alpha=0.5)
            # Add shaded area for RND bounds
            ax1.axvspan(rnd_stats['bounds_used'][0], rnd_stats['bounds_used'][1],
                      alpha=0.1, color='red', label='RND Bounds')

        # Plot historical distribution if available - also full height
        if hist_distribution is not None:
            # Generate high-res x values for the historical PDF
            high_res_x = np.linspace(x_range_limits[0], x_range_limits[1], 2000)

            # Get distribution info
            dist_name = hist_distribution['distribution']
            params = hist_distribution['params']
            dist = getattr(stats, dist_name)

            # Calculate PDF
            hist_pdf = dist.pdf(high_res_x, *params)

            # Normalize to area = 1
            hist_area = np.trapz(hist_pdf, high_res_x)
            hist_pdf_normalized = hist_pdf / hist_area if hist_area > 0 else hist_pdf

            # Plot the normalized PDF - preserving full height
            ax1.plot(high_res_x, hist_pdf_normalized, 'green', linewidth=2,
                    label=f'Historical Returns ({dist_name.capitalize()})')

            # Add histogram of actual historical returns
            hist_returns = hist_distribution['returns']
            bounded_returns = [r for r in hist_returns if x_range_limits[0] <= r <= x_range_limits[1]]

            # Use a decent number of bins but don't normalize height
            num_bins = max(50, int(np.log2(len(bounded_returns)) + 1))

            # Use weights to scale the histogram for visibility
            weights = np.ones_like(bounded_returns) / len(bounded_returns)
            weights = weights * 0.3 * max(1, len(bounded_returns)) / num_bins

            ax1.hist(bounded_returns, bins=num_bins, weights=weights, alpha=0.3, color='green',
                    label=f'Historical Returns ({len(bounded_returns)} samples)')

        # Add vertical lines for important values
        period_rf = risk_free_rate * time_to_expiration_years
        ax1.axvline(0, color='black', linestyle='-', alpha=0.7, label='Current Price (0% Return)')
        ax1.axvline(period_rf, color='darkorange', linestyle='-.', alpha=0.7, label=f'Risk-Free Rate: {period_rf:.4f}')

        if realized_moments is not None:
            ax1.axvline(realized_moments['period_return'], color='purple', linestyle='--',
                      label=f'Realized Return: {realized_moments["period_return"]:.4f}')

        # Set plot limits and labels (APA-compliant)
        ax1.set_xlim(x_range_limits)
        ax1.set_xlabel('Return (S_t/S_0 - 1)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')

        # Add stats text boxes for realized moments
        if realized_moments is not None:
            realized_stats_text = (
                f"Realized Moments:\n"
                f"Return: {realized_moments['period_return']:.4f}\n"
                f"Variance (raw): {realized_moments['realized_variance']:.6f}\n"
                f"Std Dev (raw): {realized_moments['realized_std']:.4f}\n"
                f"Annualized Vol: {realized_moments['annualized_vol']:.4f}\n"
                f"Traditional Vol: {realized_moments['traditional_annualized_vol']:.4f}\n"
                f"Skewness: {realized_moments['realized_skewness']:.4f}\n"
                f"Kurtosis: {realized_moments['realized_kurtosis']:.4f}\n"
                f"Minutes: {realized_moments['n_minutes']}\n"
                f"Overnights: {realized_moments['n_overnight']}\n"
                f"Days: {realized_moments['n_days']}"
            )

            ax1.text(0.02, 0.97, realized_stats_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top', fontsize=10)

        # Add RND stats text box
        rnd_stats_text = (
            f"Risk-Neutral Moments:\n"
            f"Mean Return: {rnd_stats['mean']:.4f}\n"
            f"Variance: {rnd_stats['variance']:.6f}\n"
            f"Std Dev: {rnd_stats['std_dev']:.4f}\n"
            f"Annualized Vol: {rnd_stats['annualized_vol']:.4f}\n"
            f"Traditional Vol: {rnd_stats['traditional_annualized_vol']:.4f}\n"
            f"Skewness: {rnd_stats['skewness']:.4f}\n"
            f"Kurtosis: {rnd_stats['kurtosis']:.4f}\n"
            f"Bounds: [{rnd_stats['bounds_used'][0]:.3f}, {rnd_stats['bounds_used'][1]:.3f}]"
        )

        ax1.text(0.98, 0.97, rnd_stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                verticalalignment='top', horizontalalignment='right', fontsize=10)

        # Add historical stats if available
        if hist_distribution is not None:
            smoothing_note = "Smoothed curve applied" if hist_distribution.get('smoothing_applied', False) else ""

            hist_stats_text = (
                f"Historical Moments (Bounds: [{x_range_limits[0]}, {x_range_limits[1]}]):\n"
                f"Distribution: {hist_distribution['distribution'].capitalize()}\n"
                f"Mean: {hist_distribution['hist_mean_return']:.4f}\n"
                f"Variance: {hist_distribution['hist_variance_raw']:.6f}\n"
                f"Std Dev: {hist_distribution['hist_std_dev_raw']:.4f}\n"
                f"Skewness: {hist_distribution['hist_skewness_raw']:.4f}\n"
                f"Kurtosis: {hist_distribution['hist_kurtosis_raw']:.4f}\n"
                f"Sample Size: {len(hist_distribution['returns'])}\n"
                f"{smoothing_note}"
            )

            ax1.text(0.98, 0.6, hist_stats_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    verticalalignment='top', horizontalalignment='right', fontsize=10)

        # Add legend and title
        ax1.legend(loc='lower left')

        # Create title with information (APA-compliant)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = USERNAME

        title_text = f'{ticker}: Return Distributions (Full Height)\n'
        title_text += f'Trading Day: {trading_day}, Expiration: {time_to_expiration_days} days\n'
        title_text += f'Dynamic Bounds with Threshold: {rnd_stats.get("density_threshold", 0.0001)}\n'
        title_text += f'Overnight Returns: {"Included" if include_overnight else "Excluded"}'

        ax1.set_title(title_text, fontsize=14, fontweight='bold')

        # Create moment comparison bar chart (APA-compliant)
        moments = ['Mean', 'Variance', 'Skewness', 'Kurtosis']

        # Add data for realized moments
        realized_values = [realized_moments['period_return'],
                          realized_moments['realized_variance'],
                          realized_moments['realized_skewness'],
                          realized_moments['realized_kurtosis']]

        # Add RND values
        rnd_values = [rnd_stats['mean'],
                     rnd_stats['variance'],
                     rnd_stats['skewness'],
                     rnd_stats['kurtosis']]

        # Add historical values if available
        if hist_distribution is not None:
            hist_values = [
                hist_distribution['hist_mean_return'],
                hist_distribution['hist_variance_raw'],
                hist_distribution['hist_skewness_raw'],
                hist_distribution['hist_kurtosis_raw']
            ]

            # Plot bar chart with all three
            x = np.arange(len(moments))
            width = 0.25
            ax2.bar(x - width, realized_values, width, label='Realized', color='blue')
            ax2.bar(x, rnd_values, width, label='Risk-Neutral', color='red')
            ax2.bar(x + width, hist_values, width, label='Historical', color='green')
        else:
            # Plot just realized and RND
            x = np.arange(len(moments))
            width = 0.35
            ax2.bar(x - width/2, realized_values, width, label='Realized', color='blue')
            ax2.bar(x + width/2, rnd_values, width, label='Risk-Neutral', color='red')

        # Set bar chart labels (APA-compliant)
        ax2.set_xticks(x)
        ax2.set_xticklabels(moments, fontweight='bold')
        ax2.legend()
        ax2.set_title(f'Comparison of Distribution Moments', fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add footnote about consistent annualization
        bounds_str = f"[{rnd_stats['bounds_used'][0]:.3f}, {rnd_stats['bounds_used'][1]:.3f}]"
        fig.text(0.5, 0.01,
                 f"Generated: {timestamp} | User: {username} | Dynamic bounds: {bounds_str} (threshold: {rnd_stats.get('density_threshold', 0.0001)})\n" +
                 f"Consistent annualization: variance * (252/days_to_expiry) for both implied and realized",
                 fontsize=8, color='black', ha='center')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)

        if show_plot:
            plt.show()

        # Create dictionary with all stats for saving
        stats_dict = {
            'trading_day': trading_day,
            'expiration_days': time_to_expiration_days,
            'time_to_expiration_years': time_to_expiration_years,
            'ticker': ticker,
            'underlying_price': underlying_price,
            'include_overnight': include_overnight,

            # RND statistics
            'rnd_mean_return': rnd_stats['mean'],
            'rnd_variance': rnd_stats['variance'],
            'rnd_std_dev': rnd_stats['std_dev'],
            'rnd_annualized_variance': rnd_stats['annualized_variance'],
            'rnd_annualized_vol': rnd_stats['annualized_vol'],
            'rnd_traditional_annualized_vol': rnd_stats['traditional_annualized_vol'],
            'rnd_skewness': rnd_stats['skewness'],
            'rnd_kurtosis': rnd_stats['kurtosis'],
            'rnd_distribution_bounds': rnd_stats['bounds_used'],
            'rnd_density_threshold': rnd_stats.get('density_threshold', 0.0001),

            # Risk-free rate info
            'risk_free_rate_annualized': risk_free_rate,
            'risk_free_rate_period': period_rf,

            # Realized statistics
            'realized_return': realized_moments['period_return'],
            'realized_variance': realized_moments['realized_variance'],
            'realized_std': realized_moments['realized_std'],
            'realized_skewness': realized_moments['realized_skewness'],
            'realized_kurtosis': realized_moments['realized_kurtosis'],
            'realized_annualized_variance': realized_moments['annualized_variance'],
            'realized_annualized_vol': realized_moments['annualized_vol'],
            'realized_traditional_annualized_variance': realized_moments['traditional_annualized_variance'],
            'realized_traditional_annualized_vol': realized_moments['traditional_annualized_vol'],
            'realized_days_observed': realized_moments['n_days'],
            'realized_minutes_observed': realized_moments['n_minutes'],
            'realized_overnight_observed': realized_moments['n_overnight'],
            'realized_total_returns': realized_moments['n_total_returns'],

            # Timestamp and user info
            'timestamp': timestamp,
            'username': username
        }

        # Add historical stats if available
        if hist_distribution is not None:
            historical_stats = {
                'hist_distribution_type': hist_distribution['distribution'],
                'hist_distribution_params': list(hist_distribution['params']),
                'hist_mean_return': hist_distribution['hist_mean_return'],
                'hist_variance_raw': hist_distribution['hist_variance_raw'],
                'hist_variance_annualized': hist_distribution['hist_variance_annualized'],
                'hist_std_dev_raw': hist_distribution['hist_std_dev_raw'],
                'hist_std_dev_annualized': hist_distribution['hist_std_dev_annualized'],
                'hist_skewness_raw': hist_distribution['hist_skewness_raw'],
                'hist_kurtosis_raw': hist_distribution['hist_kurtosis_raw'],
                'hist_sample_size': len(hist_distribution['returns']),
                'hist_distribution_bounds': [x_range_limits[0], x_range_limits[1]],
                'hist_distribution_smoothing_applied': hist_distribution.get('smoothing_applied', False)
            }
            stats_dict.update(historical_stats)

        # Save results if requested
        if save_results:
            # Create results directory
            results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
            os.makedirs(f"{results_dir}/plots", exist_ok=True)
            os.makedirs(f"{results_dir}/data", exist_ok=True)

            # Create filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            overnight_str = "with_overnight" if include_overnight else "no_overnight"

            # Save figure
            fig_path = f"{results_dir}/plots/{ticker}_{trading_day.replace('/', '_')}_{time_to_expiration_days}d_{overnight_str}_{timestamp_str}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")

            # Save statistics
            stats_path = f"{results_dir}/data/{ticker}_{trading_day.replace('/', '_')}_{time_to_expiration_days}d_{overnight_str}_{timestamp_str}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats_dict, f, indent=2)
            print(f"Saved statistics to {stats_path}")

        return fig, ax1, stats_dict

    except Exception as e:
        print(f"Error in compare_rnd_to_realized_comprehensive: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_specific_expiration_days(volatility_data, target_days=[6, 7, 8, 9, 10], tolerance=0):
    """
    Get trading days with options expiring in EXACTLY the specified number of days
    (no tolerance allowed).

    Parameters:
    volatility_data: Dictionary with volatility data
    target_days: List of target days to expiration [6, 7, 8, 9, 10]
    tolerance: Must be 0 for exact matching only

    Returns:
    dict: Dictionary with target days as keys and lists of valid trading days as values
    """
    result = {days: [] for days in target_days}

    for trading_day in volatility_data:
        # Check each expiration for this trading day
        for expiration in volatility_data[trading_day]:
            # Get time to expiration in days
            expiry_years = volatility_data[trading_day][expiration]['time_to_expiration']
            expiry_days = int(expiry_years * 365)  # Convert to days

            # Check if this EXACTLY matches any of our target days (no tolerance)
            if expiry_days in target_days:
                result[expiry_days].append((trading_day, expiration, expiry_days))

    # Sort each list chronologically by trading day
    for target in result:
        if result[target]:
            result[target].sort(key=lambda x: x[0])
            print(f"Found {len(result[target])} trading days with EXACTLY {target}-day options")
        else:
            print(f"Warning: No trading days found with EXACTLY {target}-day options")

    return result

def create_expiration_specific_time_series(volatility_data, ticker, minute_data, treasury_rates,
                                         target_days=[6, 7, 8, 9, 10], save_dir=None):
    """
    Create separate time series for each specific option expiration length
    """
    # Get trading days with the specific expirations (EXACT matches only)
    expiration_days = get_specific_expiration_days(volatility_data, target_days, tolerance=0)

    results = {}

    # Process each expiration length separately
    for target_day in target_days:
        if not expiration_days[target_day]:
            print(f"No trading days with EXACTLY {target_day}-day expirations found for {ticker}")
            continue

        print(f"\nProcessing {len(expiration_days[target_day])} trading days with EXACTLY {target_day}-day expirations...")

        # Store data for this expiration length
        target_moments = []

        # Process each trading day
        for trading_day, expiration, exact_days in expiration_days[target_day]:
            try:
                # Verify the exact match again for safety
                if exact_days != target_day:
                    print(f"Skipping {trading_day} - expiration is {exact_days} days, not exactly {target_day} days")
                    continue

                # Calculate realized moments
                trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
                expiration_day_dt = trading_day_dt + timedelta(days=exact_days)

                realized_moments = calculate_realized_moments_comprehensive(
                    minute_data, trading_day_dt, expiration_day_dt, include_overnight=True)

                if realized_moments is None:
                    continue

                # Get risk-free rate
                risk_free_rate = get_risk_free_rate(treasury_rates, trading_day, exact_days)

                # Calculate historical returns
                historical_minute_data = minute_data[minute_data['timestamp_et'] < trading_day_dt].copy()
                historical_returns = calculate_n_day_returns(
                    historical_minute_data, n_days=exact_days, start_year=2016)   # Fit historical distribution with bounds
                if len(historical_returns) > 10:
                    hist_distribution = fit_smoother_historical_distribution(
                        historical_returns, min_bound=-0.5, max_bound=0.5)

                    # Plot the historical distribution fit if save_dir is provided
                    if save_dir:
                        hist_plots_dir = f"{save_dir}/{target_day}day_historical_fits"
                        os.makedirs(hist_plots_dir, exist_ok=True)

                        plot_historical_distribution_fit(
                            historical_returns,
                            hist_distribution,
                            ticker,
                            trading_day,
                            exact_days,
                            save_dir=hist_plots_dir
                        )
                else:
                    hist_distribution = None

                # Extract RND moments with bounds
                first_expiration = expiration
                time_to_expiration_years = volatility_data[trading_day][first_expiration]['time_to_expiration']

                # Get RND data
                strike_range = np.array(volatility_data[trading_day][first_expiration]['strike_range'])
                underlying_price = volatility_data[trading_day][first_expiration]['underlyingPrice']

                # Get RND
                density_key = 'risk_neutral_density_quartic'
                if density_key in volatility_data[trading_day][first_expiration]:
                    price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                else:
                    density_key = 'risk_neutral_density'
                    if density_key in volatility_data[trading_day][first_expiration]:
                        price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                    else:
                        continue  # Skip if no RND found

                # Transform to return space
                return_range = np.linspace(min(strike_range), max(strike_range), len(price_rnd)) / underlying_price - 1.0
                rnd = price_rnd * underlying_price  # Jacobian transformation

                # Normalize
                total_area = np.trapz(rnd, return_range)
                normalized_rnd = rnd / total_area if total_area > 0 else rnd

                # Calculate RND statistics with dynamic bounds
                rnd_stats = calculate_rnd_statistics(
                    return_range, normalized_rnd, time_to_expiration_years, -0.5, 0.5, density_threshold=0.0001)

                # Create moment data
                moment_data = {
                    'trading_day': trading_day,
                    'expiration_days': exact_days,
                    'time_to_expiration_years': time_to_expiration_years,
                    'underlying_price': underlying_price,
                    'rnd_mean_return': rnd_stats['mean'],
                    'rnd_variance': rnd_stats['variance'],
                    'rnd_std_dev': rnd_stats['std_dev'],
                    'rnd_annualized_variance': rnd_stats.get('annualized_variance', rnd_stats['variance'] * (252.0/exact_days)),
                    'rnd_annualized_vol': rnd_stats['annualized_vol'],
                    'rnd_traditional_annualized_vol': rnd_stats['traditional_annualized_vol'],
                    'rnd_skewness': rnd_stats['skewness'],
                    'rnd_kurtosis': rnd_stats['kurtosis'],
                    'rnd_bounds_used': rnd_stats['bounds_used'],
                    'rnd_density_threshold': rnd_stats.get('density_threshold', 0.0001),
                    'realized_return': realized_moments['period_return'],
                    'realized_variance': realized_moments['realized_variance'],
                    'realized_vol': realized_moments['realized_std'],
                    'realized_annualized_variance': realized_moments['annualized_variance'],
                    'realized_annualized_vol': realized_moments['annualized_vol'],
                    'realized_traditional_annualized_vol': realized_moments.get('traditional_annualized_vol', 0),
                    'realized_skewness': realized_moments['realized_skewness'],
                    'realized_kurtosis': realized_moments['realized_kurtosis'],
                    'n_days': realized_moments['n_days'],
                    'days_to_expiry': exact_days
                }

                # Add historical moments if available
                if hist_distribution is not None:
                    moment_data.update({
                        'hist_mean_return': hist_distribution['hist_mean_return'],
                        'hist_variance': hist_distribution['hist_variance_raw'],
                        'hist_std_dev': hist_distribution['hist_std_dev_raw'],
                        'hist_skewness': hist_distribution['hist_skewness_raw'],
                        'hist_kurtosis': hist_distribution['hist_kurtosis_raw'],
                        'hist_distribution_type': hist_distribution['distribution'],
                        'hist_sample_size': len(hist_distribution['returns'])
                    })

                target_moments.append(moment_data)

            except Exception as e:
                print(f"Error processing {trading_day} with {exact_days}-day expiration: {e}")
                continue
        # Save and plot time series for this expiration length
        if target_moments:
            # Save to CSV
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                df = pd.DataFrame(target_moments)
                csv_path = f"{save_dir}/{ticker}_{target_day}day_options_summary.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved {len(target_moments)} data points to {csv_path}")

            # Create time series plots for this specific expiration length
            time_series_dir = f"{save_dir}/{target_day}day_time_series"
            os.makedirs(time_series_dir, exist_ok=True)

            plot_skewness_kurtosis_time_series(
                target_moments,
                f"{ticker} ({target_day}-day)",
                save_dir=time_series_dir
            )

            results[target_day] = target_moments
        else:
            print(f"No valid results for {target_day}-day options")

    return results

def process_specific_expirations(volatility_data, ticker, minute_data, treasury_rates,
                                target_days=[6, 7, 8, 9, 10], save_dir=None, resume_from_day=None):
    """
    Process and analyze data for specific option expiration days
    """
    # Get mapping of expiration days to lists of trading days
    expiration_days = get_specific_expiration_days(volatility_data, target_days, tolerance=0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = {}

    for target_day in target_days:
        if not expiration_days[target_day]:
            print(f"No trading days found with EXACTLY {target_day}-day expirations for {ticker}")
            continue

        print(f"\nProcessing {len(expiration_days[target_day])} trading days for {ticker} with EXACTLY {target_day}-day expirations...")

        # Extract trading days for this expiration length
        trading_days_with_expirations = [td for td, _, _ in expiration_days[target_day]]
        trading_days_with_expirations.sort()

        # Handle resume_from_day if provided
        if resume_from_day and resume_from_day in trading_days_with_expirations:
            resume_index = trading_days_with_expirations.index(resume_from_day)
            trading_days_to_process = trading_days_with_expirations[resume_index+1:]
            print(f"Resuming from {resume_from_day}, {len(trading_days_to_process)} days remaining")
        else:
            trading_days_to_process = trading_days_with_expirations
            if resume_from_day:
                print(f"Resume day {resume_from_day} not found, processing all {len(trading_days_to_process)} days")

        target_moments = []

        # Load existing results if available
        csv_path = f"{save_dir}/{ticker}_{target_day}day_options_summary.csv"
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                target_moments = existing_df.to_dict('records')
                print(f"Loaded {len(target_moments)} existing records from {csv_path}")

                # Extract already processed days to avoid duplicates
                processed_days = set(existing_df['trading_day'].tolist())
                trading_days_to_process = [day for day in trading_days_to_process if day not in processed_days]
                print(f"After removing already processed days, {len(trading_days_to_process)} days remain")
            except Exception as e:
                print(f"Error loading existing results: {e}")

        # Process each trading day with this expiration length
        for trading_day_tuple in expiration_days[target_day]:
            trading_day, expiration_date, exact_days = trading_day_tuple

            # Skip if not in our processing list
            if trading_day not in trading_days_to_process:
                continue

            try:
                # Generate historical distribution if we have enough history
                hist_distribution = None

                # Calculate realized moments using minute data
                if minute_data is not None:
                    try:
                        trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
                    except (ValueError, TypeError):
                        # Handle Unix timestamp
                        trading_day_dt = pd.to_datetime(int(trading_day), unit='s')

                    try:
                        expiration_day_dt = pd.to_datetime(expiration_date, format='%m/%d/%Y')
                    except (ValueError, TypeError):
                        # Handle Unix timestamp
                        expiration_day_dt = pd.to_datetime(int(expiration_date), unit='s')

                    # Calculate realized moments using comprehensive method
                    result = compare_rnd_to_realized_comprehensive(
                        volatility_data,
                        trading_day,
                        ticker,
                        minute_data,
                        treasury_rates=treasury_rates,
                        x_range_limits=(-0.5, 0.5),
                        include_overnight=False,
                        show_plot=False,
                        save_results=True
                    )

                    if result is None:
                        continue

                    _, _, realized_moments = result

                # Extract RND moments with bounds
                first_expiration = expiration_date
                time_to_expiration_years = volatility_data[trading_day][first_expiration]['time_to_expiration']

                # Get RND data
                strike_range = np.array(volatility_data[trading_day][first_expiration]['strike_range'])
                underlying_price = volatility_data[trading_day][first_expiration]['underlyingPrice']

                # Get RND
                density_key = 'risk_neutral_density_quartic'
                if density_key in volatility_data[trading_day][first_expiration]:
                    price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                else:
                    density_key = 'risk_neutral_density'
                    if density_key in volatility_data[trading_day][first_expiration]:
                        price_rnd = np.array(volatility_data[trading_day][first_expiration][density_key])
                    else:
                        continue  # Skip if no RND found

                # Transform to return space
                return_range = np.linspace(min(strike_range), max(strike_range), len(price_rnd)) / underlying_price - 1.0
                rnd = price_rnd * underlying_price  # Jacobian transformation

                # Normalize
                total_area = np.trapz(rnd, return_range)
                normalized_rnd = rnd / total_area if total_area > 0 else rnd

                # Calculate RND statistics with dynamic bounds
                rnd_stats = calculate_rnd_statistics(
                    return_range, normalized_rnd, time_to_expiration_years, -0.5, 0.5, density_threshold=0.0001)

                # Create moment data
                moment_data = {
                    'trading_day': trading_day,
                    'expiration_days': exact_days,
                    'time_to_expiration_years': time_to_expiration_years,
                    'underlying_price': underlying_price,
                    'rnd_mean_return': rnd_stats['mean'],
                    'rnd_variance': rnd_stats['variance'],
                    'rnd_std_dev': rnd_stats['std_dev'],
                    'rnd_annualized_variance': rnd_stats.get('annualized_variance', rnd_stats['variance'] * (252.0/exact_days)),
                    'rnd_annualized_vol': rnd_stats['annualized_vol'],
                    'rnd_traditional_annualized_vol': rnd_stats['traditional_annualized_vol'],
                    'rnd_skewness': rnd_stats['skewness'],
                    'rnd_kurtosis': rnd_stats['kurtosis'],
                    'rnd_bounds_used': rnd_stats['bounds_used'],
                    'rnd_density_threshold': rnd_stats.get('density_threshold', 0.0001),
                    'rnd_smoothness_score': rnd_stats.get('smoothness', {}).get('smoothness_score', 0.0),
                    'rnd_is_smooth': rnd_stats.get('smoothness', {}).get('is_smooth', False),
                    'realized_return': realized_moments['period_return'],
                    'realized_variance': realized_moments['realized_variance'],
                    'realized_vol': realized_moments['realized_std'],
                    'realized_annualized_variance': realized_moments['annualized_variance'],
                    'realized_annualized_vol': realized_moments['annualized_vol'],
                    'realized_traditional_annualized_vol': realized_moments.get('traditional_annualized_vol', 0),
                    'realized_skewness': realized_moments['realized_skewness'],
                    'realized_kurtosis': realized_moments['realized_kurtosis'],
                    'n_days': realized_moments['n_days'],
                    'days_to_expiry': exact_days
                }

                # Add historical moments if available
                if hist_distribution is not None:
                    moment_data.update({
                        'hist_mean_return': hist_distribution['hist_mean_return'],
                        'hist_variance': hist_distribution['hist_variance_raw'],
                        'hist_std_dev': hist_distribution['hist_std_dev_raw'],
                        'hist_skewness': hist_distribution['hist_skewness_raw'],
                        'hist_kurtosis': hist_distribution['hist_kurtosis_raw'],
                        'hist_distribution_type': hist_distribution['distribution'],
                        'hist_sample_size': len(hist_distribution['returns'])
                    })

                # =================================================================
                # ADD AMAYA SCALING CALCULATION - BEGIN NEW CODE
                # =================================================================
                # Convert dates for Amaya calculation
                trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
                expiration_day_dt = pd.to_datetime(expiration_date, format='%m/%d/%Y')

                # Calculate realized moments using Amaya scaling
                amaya_moments = calculate_realized_higher_moments_amaya(
                    minute_data,
                    trading_day_dt,
                    expiration_day_dt,
                    moment_data
                )

                if amaya_moments:
                    # Store original values for comparison
                    moment_data['original_realized_skewness'] = moment_data['realized_skewness']
                    moment_data['original_realized_kurtosis'] = moment_data['realized_kurtosis']

                    # Update with Amaya-scaled values
                    moment_data['realized_skewness'] = amaya_moments['realized_skewness']
                    moment_data['realized_kurtosis'] = amaya_moments['realized_kurtosis']
                    moment_data['amaya_days_observed'] = amaya_moments.get('days_observed')
                    moment_data['amaya_observations'] = amaya_moments.get('total_observations')
                    moment_data['amaya_scaling_applied'] = True

                    print(f"  {trading_day}: Amaya scaling applied. Skew: {moment_data['realized_skewness']:.4f}, Kurt: {moment_data['realized_kurtosis']:.4f}")
                else:
                    print(f"  {trading_day}: Amaya scaling failed. Using original realized moments.")
                    moment_data['amaya_scaling_applied'] = False
                # =================================================================
                # ADD AMAYA SCALING CALCULATION - END NEW CODE
                # =================================================================

                target_moments.append(moment_data)

            except Exception as e:
                print(f"Error processing {trading_day} with {exact_days}-day expiration: {e}")
                continue

        # Save and plot time series for this expiration length
        if target_moments:
            # Save to CSV
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                df = pd.DataFrame(target_moments)
                csv_path = f"{save_dir}/{ticker}_{target_day}day_options_summary.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved {len(target_moments)} data points to {csv_path}")

            # Create time series plots for this specific expiration length
            time_series_dir = f"{save_dir}/{target_day}day_time_series"
            os.makedirs(time_series_dir, exist_ok=True)

            plot_skewness_kurtosis_time_series(
                target_moments,
                f"{ticker} ({target_day}-day)",
                save_dir=time_series_dir
            )

            # Add plots comparing original vs Amaya moments if we have them
            if any('original_realized_skewness' in moment for moment in target_moments):
                plot_amaya_comparison(
                    target_moments,
                    ticker,
                    target_day,
                    save_dir=time_series_dir
                )

            results[target_day] = target_moments
        else:
            print(f"No valid results for {target_day}-day options")

    return results

def plot_amaya_comparison(results_data, ticker, days, save_dir=None):
    """
    Create plots comparing original realized moments with Amaya scaled moments

    Parameters:
    results_data: List of dictionaries containing moment data
    ticker: Stock ticker symbol
    days: Number of days to expiration
    save_dir: Directory to save plots
    """
    if not results_data or len(results_data) < 2:
        print("Insufficient data for Amaya comparison plots")
        return

    # Check if Amaya data exists
    if 'original_realized_skewness' not in results_data[0]:
        print("No Amaya comparison data available")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
    df = df.sort_values('trading_day')

    # Create skewness comparison plot
    fig_skew, ax_skew = plt.subplots(figsize=(10, 6), dpi=300)
    ax_skew.plot(df['trading_day'], df['original_realized_skewness'], 'b--',
              alpha=0.5, linewidth=1.5, label='Original Realized Skewness')
    ax_skew.plot(df['trading_day'], df['realized_skewness'], 'b-',
              linewidth=2, label='Amaya Scaled Skewness')

    if 'rnd_skewness' in df.columns:
        ax_skew.plot(df['trading_day'], df['rnd_skewness'], 'r-',
                  linewidth=2, label='Implied (RND) Skewness')

    # Add zero reference line
    ax_skew.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Format plot
    ax_skew.set_title(f'{ticker} ({days}-day): Skewness with Amaya Scaling', fontsize=14, fontweight='bold')
    ax_skew.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_skew.set_ylabel('Skewness', fontsize=12, fontweight='bold')
    ax_skew.legend()
    ax_skew.grid(True, alpha=0.3)
    fig_skew.autofmt_xdate()

    # Add footnote
    fig_skew.text(0.5, 0.01,
               f"Generated: {TIMESTAMP} | User: {USERNAME} | Amaya et al. (2015) scaling applied",
               fontsize=8, ha='center')

    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_skew.savefig(f"{save_dir}/{ticker}_{days}day_skewness_amaya_comparison.png",
                       dpi=300, bbox_inches='tight')

    # Create kurtosis comparison plot
    fig_kurt, ax_kurt = plt.subplots(figsize=(10, 6), dpi=300)
    ax_kurt.plot(df['trading_day'], df['original_realized_kurtosis'], 'b--',
              alpha=0.5, linewidth=1.5, label='Original Realized Kurtosis')
    ax_kurt.plot(df['trading_day'], df['realized_kurtosis'], 'b-',
              linewidth=2, label='Amaya Scaled Kurtosis')

    if 'rnd_kurtosis' in df.columns:
        ax_kurt.plot(df['trading_day'], df['rnd_kurtosis'], 'r-',
                  linewidth=2, label='Implied (RND) Kurtosis')

    # Add normal kurtosis reference line
    ax_kurt.axhline(y=3, color='black', linestyle='-', alpha=0.3, label='Normal Kurtosis (3)')

    # Format plot
    ax_kurt.set_title(f'{ticker} ({days}-day): Kurtosis with Amaya Scaling', fontsize=14, fontweight='bold')
    ax_kurt.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_kurt.set_ylabel('Kurtosis', fontsize=12, fontweight='bold')
    ax_kurt.legend()
    ax_kurt.grid(True, alpha=0.3)
    fig_kurt.autofmt_xdate()

    # Add footnote
    fig_kurt.text(0.5, 0.01,
               f"Generated: {TIMESTAMP} | User: {USERNAME} | Amaya et al. (2015) scaling applied",
               fontsize=8, ha='center')

    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_kurt.savefig(f"{save_dir}/{ticker}_{days}day_kurtosis_amaya_comparison.png",
                       dpi=300, bbox_inches='tight')

    print(f"Created Amaya comparison plots for {ticker} ({days}-day)")

def calculate_realized_higher_moments_amaya_wrapper():
    """
    Wrapper function to calculate realized skewness and kurtosis using
    the Amaya et al. (2015) formulas and compare with implied moments.
    """
    # Path to where ticker results are stored
    base_dir = '/content/drive/MyDrive'

    # Get list of tickers with saved data
    ticker_options = []
    for ticker in valid_tickers:
        results_dir = f'{base_dir}/{ticker}_analysis_results'

        # Check if directory exists
        if os.path.exists(results_dir):
            # Check for standard moments file
            if os.path.exists(f"{results_dir}/{ticker}_moments_summary.csv"):
                ticker_options.append((ticker, f"{results_dir}/{ticker}_moments_summary.csv", "all expirations"))

            # Check for expiration-specific files
            exp_dir = f"{results_dir}/expiration_specific"
            if os.path.exists(exp_dir):
                # Look for specific day files
                for day_length in [6, 7, 8, 9, 10]:
                    exp_file = f"{exp_dir}/{ticker}_{day_length}day_options_summary.csv"
                    if os.path.exists(exp_file):
                        ticker_options.append((ticker, exp_file, f"{day_length}-day"))

    if not ticker_options:
        print("No ticker data found to analyze with Amaya formulas.")
        return

    # Display available tickers for analysis
    print("\nAvailable tickers for Amaya moments calculation:")
    for i, (ticker, file_path, expiry_info) in enumerate(ticker_options):
        print(f"{i+1}. {ticker} ({expiry_info})")

    # Get user selection
    selection = input("\nSelect a ticker to analyze (number) or 'all' for all tickers: ")

    tickers_to_process = []
    if selection.lower() == 'all':
        tickers_to_process = ticker_options
    else:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(ticker_options):
                tickers_to_process = [ticker_options[idx]]
            else:
                print("Invalid selection. Please try again.")
                return
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")
            return

    # Process each selected ticker
    for ticker, csv_file, expiry_info in tickers_to_process:
        print(f"\nProcessing {ticker} ({expiry_info})...")

        # Determine time series directory
        if "all expirations" in expiry_info:
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/time_series"
            day_length = None
        else:
            # Extract days from expiry_info
            day_length = int(expiry_info.split("-")[0])
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific/{day_length}day_time_series"

        # Create time series directory if it doesn't exist
        os.makedirs(time_series_dir, exist_ok=True)

        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: Moments CSV file not found: {csv_file}")
            continue

        # Load minute data for the ticker
        try:
            minute_data = load_minute_data(ticker)
            if len(minute_data) == 0:
                print(f"Error: No minute data available for {ticker}. Skipping analysis.")
                continue

            print(f"Successfully loaded {len(minute_data)} minute records")
        except Exception as e:
            print(f"Error loading minute data for {ticker}: {str(e)}")
            continue

        try:
            # Load the moments data
            moments_df = pd.read_csv(csv_file)
            print(f"Loaded {len(moments_df)} rows from {csv_file}")

            # Make a backup of the original data
            backup_csv = csv_file.replace(".csv", "_original.csv")
            if not os.path.exists(backup_csv):
                moments_df.to_csv(backup_csv, index=False)
                print(f"Created backup at {backup_csv}")

            # Convert trading_day to datetime
            moments_df['trading_day_dt'] = pd.to_datetime(moments_df['trading_day'], format='%m/%d/%Y')

            # Get expiration days - either from the day_length or from the data
            if day_length:
                moments_df['expiration_days'] = day_length
            else:
                # Find expiration days column
                if 'expiration_days' not in moments_df.columns and 'time_to_expiration_years' in moments_df.columns:
                    moments_df['expiration_days'] = moments_df['time_to_expiration_years'] * 365
                elif 'days_to_expiry' in moments_df.columns:
                    moments_df['expiration_days'] = moments_df['days_to_expiry']

            # Calculate expiration dates
            moments_df['expiration_day_dt'] = moments_df['trading_day_dt'] + pd.to_timedelta(moments_df['expiration_days'], unit='D')

            # Recalculate realized moments using Amaya formulas
            print("\nCalculating realized moments using Amaya et al. (2015) formulas...")

            updated_moments = []
            for i, row in tqdm(moments_df.iterrows(), total=len(moments_df), desc="Processing days"):
                # Calculate the Amaya higher moments
                amaya_moments = calculate_realized_higher_moments_amaya(
                    minute_data,
                    row['trading_day_dt'],
                    row['expiration_day_dt']
                )

                if amaya_moments:
                    # Update with new moments
                    row_dict = row.to_dict()

                    # Store original values for comparison
                    row_dict['original_realized_skewness'] = row_dict.get('realized_skewness', 0)
                    row_dict['original_realized_kurtosis'] = row_dict.get('realized_kurtosis', 3)

                    # Update with Amaya moments
                    row_dict['realized_skewness'] = amaya_moments['realized_skewness']
                    row_dict['realized_kurtosis'] = amaya_moments['realized_kurtosis']
                    row_dict['amaya_days_observed'] = amaya_moments['days_observed']
                    row_dict['amaya_observations'] = amaya_moments['total_observations']

                    updated_moments.append(row_dict)
                else:
                    # Keep original row if can't calculate new moments
                    updated_moments.append(row.to_dict())

            # Convert back to DataFrame
            updated_df = pd.DataFrame(updated_moments)

            # Sort by trading day for time series
            updated_df = updated_df.sort_values('trading_day_dt')

            # Save the updated data
            updated_csv = csv_file.replace(".csv", "_amaya.csv")
            updated_df.to_csv(updated_csv, index=False)
            print(f"Saved updated moments with Amaya calculations to {updated_csv}")

            # Create plots comparing original and new moments, and with implied moments
            print("\nCreating comparison plots...")

            # Plot skewness comparison
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # Plot original realized skewness
            if 'original_realized_skewness' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['original_realized_skewness'], 'b--',
                        alpha=0.5, linewidth=1.5, label='Original Realized Skewness')

            # Plot new Amaya realized skewness
            ax.plot(updated_df['trading_day_dt'], updated_df['realized_skewness'], 'b-',
                    linewidth=2, label='Amaya Realized Skewness')

            # Plot implied skewness for comparison
            if 'rnd_skewness' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['rnd_skewness'], 'r-',
                        linewidth=2, label='Implied (RND) Skewness')

            # Add reference line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Set title and labels
            title = f'{ticker}: Skewness Comparison'
            if day_length:
                title += f' (Exactly {day_length}-day Expiration)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Skewness', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates
            fig.autofmt_xdate()

            # Add metadata
            days_note = f"Exactly {day_length}-day expiration" if day_length else "Multiple expirations"
            fig.text(0.5, 0.01,
                    f"Generated: {TIMESTAMP} | User: {USERNAME} | {days_note} | Using Amaya et al. (2015) formulas",
                    fontsize=9, ha='center')

            # Save figure
            skew_fig_path = f"{time_series_dir}/{ticker}_skewness_amaya_comparison.png"
            plt.savefig(skew_fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved skewness comparison to {skew_fig_path}")
            plt.close(fig)

            # Plot kurtosis comparison
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # Plot original realized kurtosis
            if 'original_realized_kurtosis' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['original_realized_kurtosis'], 'b--',
                        alpha=0.5, linewidth=1.5, label='Original Realized Kurtosis')

            # Plot new Amaya realized kurtosis
            ax.plot(updated_df['trading_day_dt'], updated_df['realized_kurtosis'], 'b-',
                    linewidth=2, label='Amaya Realized Kurtosis')

            # Plot implied kurtosis for comparison
            if 'rnd_kurtosis' in updated_df.columns:
                ax.plot(updated_df['trading_day_dt'], updated_df['rnd_kurtosis'], 'r-',
                        linewidth=2, label='Implied (RND) Kurtosis')

            # Add reference line at normal kurtosis (3)
            ax.axhline(y=3, color='black', linestyle='-', alpha=0.3, label='Normal Kurtosis (3)')

            # Set title and labels
            title = f'{ticker}: Kurtosis Comparison'
            if day_length:
                title += f' (Exactly {day_length}-day Expiration)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Kurtosis', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format dates
            fig.autofmt_xdate()

            # Add metadata
            fig.text(0.5, 0.01,
                    f"Generated: {TIMESTAMP} | User: {USERNAME} | {days_note} | Using Amaya et al. (2015) formulas",
                    fontsize=9, ha='center')

            # Save figure
            kurt_fig_path = f"{time_series_dir}/{ticker}_kurtosis_amaya_comparison.png"
            plt.savefig(kurt_fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved kurtosis comparison to {kurt_fig_path}")
            plt.close(fig)

            # Ask if user wants to update the original file
            update_choice = 'y'
            if update_choice.lower() == 'y':
                # Save to original CSV
                updated_df.to_csv(csv_file, index=False)
                print(f"Updated {csv_file} with Amaya realized moments")

                # Regenerate all time series plots
                if day_length:
                    print("\nRegenerating time series plots with updated moments...")
                    plot_skewness_kurtosis_time_series(
                        updated_df.to_dict('records'),
                        f"{ticker} ({day_length}-day)",
                        save_dir=time_series_dir
                    )

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAmaya moments comparison completed.")
    print("You can use the generated plots to analyze differences between original and Amaya-based higher moments.")

def remove_outliers_from_moments(moments_data, column, n_std=30):
    """
    Remove outliers from a specific moment column

    Parameters:
    moments_data (pd.DataFrame): DataFrame with moments data
    column (str): Column name to check for outliers
    n_std (float): Number of standard deviations to use as threshold

    Returns:
    pd.DataFrame: Filtered DataFrame
    """
    if len(moments_data) == 0 or column not in moments_data.columns:
        return moments_data

    # Calculate mean and std
    mean = moments_data[column].mean()
    std = moments_data[column].std()

    if std == 0:  # Avoid division by zero
        return moments_data

    # Create mask for values within n standard deviations
    mask = (moments_data[column] > mean - n_std * std) & (moments_data[column] < mean + n_std * std)

    # Apply mask
    filtered_df = moments_data[mask]

    removed_count = len(moments_data) - len(filtered_df)
    print(f"Removed {removed_count} outliers from {column}")

    return filtered_df

def plot_moment_time_series(stats_data, ticker, moment_name, save_dir=None):
    """
    Plot time series of a specific moment with outlier removal and column name flexibility
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(stats_data)

        # Convert trading_day to datetime and sort
        df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
        df = df.sort_values('trading_day')

        # Define columns to use with fallback options
        if moment_name == 'return':
            realized_col = 'realized_return'
            # Check for different possible names for RND return
            if 'rnd_return' in df.columns:
                rnd_col = 'rnd_return'
            elif 'rnd_mean_return' in df.columns:
                rnd_col = 'rnd_mean_return'
            else:
                rnd_col = None
                print(f"Warning: No RND return column found. Will plot realized return only.")
            hist_col = 'hist_mean_return' if 'hist_mean_return' in df.columns else None
        elif moment_name == 'variance':
            realized_col = f'realized_{moment_name}'
            rnd_col = f'rnd_{moment_name}'
            hist_col = f'hist_{moment_name}_raw' if f'hist_{moment_name}_raw' in df.columns else None
        else:
            # For skewness, kurtosis
            realized_col = f'realized_{moment_name}'
            rnd_col = f'rnd_{moment_name}'
            hist_col = f'hist_{moment_name}_raw' if f'hist_{moment_name}_raw' in df.columns else None

        # Create APA-compliant plot
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        # Check if columns exist before plotting
        if realized_col in df.columns:
            # Print some stats to debug Amaya values
            print(f"Realized {moment_name} stats:")
            print(f"  Mean: {df[realized_col].mean()}")
            print(f"  Min: {df[realized_col].min()}")
            print(f"  Max: {df[realized_col].max()}")
            print(f"  Count: {df[realized_col].count()}")
            print(f"  Zeros: {(df[realized_col] == 0).sum()}")

            # Remove outliers
            df_realized = remove_outliers_from_moments(df, realized_col, n_std=3)
            ax.plot(df_realized['trading_day'], df_realized[realized_col], 'blue',
                    linewidth=2, label=f'Realized {moment_name.capitalize()}')

        if rnd_col and rnd_col in df.columns:
            # Print stats for RND values
            print(f"RND {moment_name} stats:")
            print(f"  Mean: {df[rnd_col].mean()}")
            print(f"  Min: {df[rnd_col].min()}")
            print(f"  Max: {df[rnd_col].max()}")

            df_rnd = remove_outliers_from_moments(df, rnd_col, n_std=3)
            ax.plot(df_rnd['trading_day'], df_rnd[rnd_col], 'red',
                    linewidth=2, label=f'Risk-Neutral {moment_name.capitalize()}')

        # Plot historical moments if available
        if hist_col and hist_col in df.columns:
            df_hist = remove_outliers_from_moments(df, hist_col, n_std=3)
            ax.plot(df_hist['trading_day'], df_hist[hist_col], 'green',
                    linewidth=2, label=f'Historical {moment_name.capitalize()}')

        # Add reference line at zero if skewness or kurtosis
        if moment_name in ['skewness', 'kurtosis']:
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # For kurtosis, add reference at 3 (normal distribution)
            if moment_name == 'kurtosis':
                ax.axhline(y=3, color='black', linestyle='--', alpha=0.3,
                           label='Normal Kurtosis (3)')

        # Add linear trend lines
        if realized_col in df.columns and len(df_realized) > 5:
            # Add trend for realized
            z = np.polyfit(range(len(df_realized)), df_realized[realized_col], 1)
            p = np.poly1d(z)
            ax.plot(df_realized['trading_day'], p(range(len(df_realized))), 'blue',
                   linestyle='--', alpha=0.5, linewidth=1.5)

        if rnd_col and rnd_col in df.columns and len(df_rnd) > 5:
            # Add trend for RND
            z = np.polyfit(range(len(df_rnd)), df_rnd[rnd_col], 1)
            p = np.poly1d(z)
            ax.plot(df_rnd['trading_day'], p(range(len(df_rnd))), 'red',
                   linestyle='--', alpha=0.5, linewidth=1.5)

        # Set APA-compliant labels and title
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{moment_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{ticker}: {moment_name.capitalize()} Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)

        # Format dates
        fig.autofmt_xdate()

        # Add analysis information as footnotes
        fig.text(0.5, 0.01, f"Data generated: {TIMESTAMP} | Outliers beyond 3σ removed | User: {USERNAME}",
                 ha='center', fontsize=8, color='black')

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig_path = f"{save_dir}/{ticker}_{moment_name}_time_series.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved {moment_name} time series plot to {fig_path}")

        return fig

    except Exception as e:
        print(f"Error plotting {moment_name} time series: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_variance_comparison(stats_data, ticker, save_dir=None):
    """
    Plot implied vs realized annualized variance with flexible column names
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(stats_data)

        # Convert trading_day to datetime and sort
        df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
        df = df.sort_values('trading_day')

        # Check for different possible column names
        realized_var_candidates = [
            'realized_annualized_variance',
            'annualized_variance',
            'realized_variance'
        ]

        rnd_var_candidates = [
            'rnd_annualized_variance',
            'rnd_variance_annualized',
            'rnd_variance'
        ]

        # Find first matching column
        realized_var_col = next((col for col in realized_var_candidates if col in df.columns), None)
        rnd_var_col = next((col for col in rnd_var_candidates if col in df.columns), None)

        if not realized_var_col:
            print("No realized variance column found. Cannot create comparison plot.")
            return None

        if not rnd_var_col:
            print("No RND variance column found. Will only plot realized variance.")

        # If using raw variance, apply annualization
        if realized_var_col == 'realized_variance' and 'days_to_expiry' in df.columns:
            print("Annualizing realized variance using days_to_expiry")
            df['realized_annualized_variance'] = df['realized_variance'] * (252.0 / df['days_to_expiry'])
            realized_var_col = 'realized_annualized_variance'

        if rnd_var_col == 'rnd_variance' and 'days_to_expiry' in df.columns:
            print("Annualizing RND variance using days_to_expiry")
            df['rnd_annualized_variance'] = df['rnd_variance'] * (252.0 / df['days_to_expiry'])
            rnd_var_col = 'rnd_annualized_variance'

        # Print stats for debugging
        print(f"Realized variance column: {realized_var_col}")
        print(f"  Mean: {df[realized_var_col].mean()}")
        print(f"  Min: {df[realized_var_col].min()}")
        print(f"  Max: {df[realized_var_col].max()}")

        if rnd_var_col:
            print(f"RND variance column: {rnd_var_col}")
            print(f"  Mean: {df[rnd_var_col].mean()}")
            print(f"  Min: {df[rnd_var_col].min()}")
            print(f"  Max: {df[rnd_var_col].max()}")

        # Remove outliers from each series separately
        df_realized = remove_outliers_from_moments(df, realized_var_col, n_std=3)

        if rnd_var_col:
            df_rnd = remove_outliers_from_moments(df, rnd_var_col, n_std=3)

        # Create APA-compliant plot
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        # Plot realized variance
        ax.plot(df_realized['trading_day'], df_realized[realized_var_col], 'blue',
                linewidth=2, label='Realized Variance (Annualized)')

        # Plot RND variance if available
        if rnd_var_col:
            ax.plot(df_rnd['trading_day'], df_rnd[rnd_var_col], 'red',
                    linewidth=2, label='Risk-Neutral Variance (Annualized)')

        # Optional: Convert to volatility and add secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(df_realized['trading_day'], np.sqrt(df_realized[realized_var_col]), 'blue',
                 linestyle='--', linewidth=1.5, label='Realized Volatility (Annualized)')

        if rnd_var_col:
            ax2.plot(df_rnd['trading_day'], np.sqrt(df_rnd[rnd_var_col]), 'red',
                     linestyle='--', linewidth=1.5, label='Risk-Neutral Volatility (Annualized)')

        ax2.set_ylabel('Annualized Volatility', fontsize=12, fontweight='bold', color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

        # Add note about consistent annualization
        title = f'{ticker}: Implied vs Realized Annualized Variance/Volatility\n'
        title += f'Using consistent annualization: variance * (252/days_to_expiry)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set APA-compliant labels
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annualized Variance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Format dates
        fig.autofmt_xdate()

        # Add analysis information as footnotes
        fig.text(0.5, 0.01, f"Data generated: {TIMESTAMP} | Outliers beyond 3σ removed | User: {USERNAME}",
                 ha='center', fontsize=8, color='black')

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig_path = f"{save_dir}/{ticker}_variance_volatility_comparison.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved variance/volatility comparison plot to {fig_path}")

        return fig

    except Exception as e:
        print(f"Error plotting variance comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_price_with_moments(stats_data, ticker, moment_type='skewness', save_dir=None, use_adjusted_price=True):
    """
    Plot adjusted asset price alongside either skewness or kurtosis time series

    Parameters:
    stats_data: List of dictionaries with stats data
    ticker: Ticker symbol
    moment_type: Type of moment to plot ('skewness' or 'kurtosis')
    save_dir: Directory to save the plot
    use_adjusted_price: Whether to use adjusted price (accounting for dividends)

    Returns:
    matplotlib.figure.Figure: The created figure
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(stats_data)

        # Convert trading_day to datetime and sort
        df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
        df = df.sort_values('trading_day')

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)

        # Calculate adjusted price if not already adjusted and requested
        price_column = 'underlying_price'
        if use_adjusted_price and 'underlying_price_adjusted' not in df.columns:
            # If we don't have dividend data, we'll use the raw price
            # In a real implementation, you might fetch dividend data and apply adjustments
            print("Using raw price data - to add dividend adjustments, you need dividend payment data")
            df['underlying_price_adjusted'] = df['underlying_price']
            price_column = 'underlying_price_adjusted'
        elif use_adjusted_price and 'underlying_price_adjusted' in df.columns:
            price_column = 'underlying_price_adjusted'

        # Plot price on primary y-axis
        ax1.plot(df['trading_day'], df[price_column], 'black', linewidth=2.5,
                label=f'{"Adjusted " if use_adjusted_price else ""}Price')
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{ticker} {"Adjusted " if use_adjusted_price else ""}Price ($)', color='black',
                      fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # Create secondary y-axis for moments
        ax2 = ax1.twinx()

        # Select columns for plotting based on moment type
        if moment_type.lower() == 'skewness':
            realized_col = 'realized_skewness'
            rnd_col = 'rnd_skewness'
            hist_col = 'hist_skewness_raw' if 'hist_skewness_raw' in df.columns else None
            moment_label = 'Skewness'
        else:  # Default to kurtosis
            realized_col = 'realized_kurtosis'
            rnd_col = 'rnd_kurtosis'
            hist_col = 'hist_kurtosis_raw' if 'hist_kurtosis_raw' in df.columns else None
            moment_label = 'Kurtosis'

        # Don't remove outliers to preserve time series continuity
        # Plot moments on secondary y-axis using original data
        if realized_col in df.columns:
            ax2.plot(df['trading_day'], df[realized_col], 'blue',
                    linewidth=2, label=f'Realized {moment_label}')

        if rnd_col in df.columns:
            ax2.plot(df['trading_day'], df[rnd_col], 'red',
                    linewidth=2, label=f'Implied {moment_label}')

        if hist_col in df.columns:
            ax2.plot(df['trading_day'], df[hist_col], 'green',
                    linewidth=2, label=f'Historical {moment_label}')

        # Add reference lines if plotting kurtosis
        if moment_type.lower() == 'kurtosis':
            ax2.axhline(y=3, color='gray', linestyle='--', alpha=0.7, label='Normal Kurtosis (3)')
        elif moment_type.lower() == 'skewness':
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Zero Skewness')

        # Set labels for secondary axis
        ax2.set_ylabel(f'{moment_label}', color='blue', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Add title
        title = f'{ticker}: Price vs {moment_label} Time Series'
        if use_adjusted_price:
            title += ' (Dividend Adjusted)'
        plt.title(title, fontsize=14, fontweight='bold')

        # Add combined legend from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Format x-axis dates
        fig.autofmt_xdate()

        # Add footnote
        fig.text(0.5, 0.01, f"Generated: {TIMESTAMP} | User: {USERNAME} | Outliers beyond 3σ removed",
                ha='center', fontsize=8, color='black')

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"{ticker}_price_vs_{moment_type}.png"
            plt.savefig(os.path.join(save_dir, file_name), dpi=300, bbox_inches='tight')
            print(f"Saved price vs {moment_type} plot to {os.path.join(save_dir, file_name)}")

        return fig

    except Exception as e:
        print(f"Error plotting price with {moment_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_skewness_kurtosis_time_series(stats_data, ticker, save_dir=None):
    """
    Plot separate time series for each moment with outlier removal
    """
    try:
        if not stats_data or len(stats_data) < 2:
            print("Insufficient data for time series plots")
            return None

        # Create time series directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Plot each moment separately
        results = []

        # Plot asset price
        df = pd.DataFrame(stats_data)
        df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
        df = df.sort_values('trading_day')

        # APA-compliant asset price chart
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        ax.plot(df['trading_day'], df['underlying_price'], 'blue', linewidth=2)
        ax.set_title(f'{ticker}: Asset Price', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        # Add footnote
        fig.text(0.5, 0.01, f"Generated: {TIMESTAMP} | User: {USERNAME}",
                 ha='center', fontsize=8, color='black')

        if save_dir:
            fig_path = f"{save_dir}/{ticker}_price_time_series.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved price time series plot to {fig_path}")

        results.append(fig)

        # Plot returns
        returns_fig = plot_moment_time_series(stats_data, ticker, 'return', save_dir)
        if returns_fig:
            results.append(returns_fig)

        # Plot variance
        variance_fig = plot_moment_time_series(stats_data, ticker, 'variance', save_dir)
        if variance_fig:
            results.append(variance_fig)

        # Plot annualized variance/volatility
        variance_comparison_fig = plot_variance_comparison(stats_data, ticker, save_dir)
        if variance_comparison_fig:
            results.append(variance_comparison_fig)

        # Plot skewness
        skewness_fig = plot_moment_time_series(stats_data, ticker, 'skewness', save_dir)
        if skewness_fig:
            results.append(skewness_fig)

        # Plot kurtosis
        kurtosis_fig = plot_moment_time_series(stats_data, ticker, 'kurtosis', save_dir)
        if kurtosis_fig:
            results.append(kurtosis_fig)

        # Add combined price and moments plots
        try:
            # Plot price with skewness
            price_skew_fig = plot_price_with_moments(
                stats_data, ticker, 'skewness', save_dir, use_adjusted_price=True)
            if price_skew_fig:
                results.append(price_skew_fig)

            # Plot price with kurtosis
            price_kurt_fig = plot_price_with_moments(
                stats_data, ticker, 'kurtosis', save_dir, use_adjusted_price=True)
            if price_kurt_fig:
                results.append(price_kurt_fig)

        except Exception as e:
            print(f"Error creating combined price/moments plots: {str(e)}")

        return results

    except Exception as e:
        print(f"Error plotting time series: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def add_iv_curves(ax1, volatility_data, trading_day, first_expiration, strike_range, underlying_price, iv_type="standard"):
    """Add IV curves to plot if available - prioritizing no arbitrage smoothed IV"""

    # Check for smoothed IV availability
    if 'iv_smooth_no_arb' in volatility_data[trading_day][first_expiration]:
        # Use no arbitrage smoothed IV
        iv_smooth = np.array(volatility_data[trading_day][first_expiration]['iv_smooth_no_arb'])
        iv_label = 'No-Arbitrage Smoothed IV'
    elif 'iv_smooth' in volatility_data[trading_day][first_expiration]:
        # Fall back to regular smoothed IV
        iv_smooth = np.array(volatility_data[trading_day][first_expiration]['iv_smooth'])
        iv_label = 'Smoothed IV'
    else:
        # No smoothed IV available
        return

    # Create proper x coordinates for IV data
    # Create proper x coordinates for IV data
    iv_strikes = np.linspace(min(strike_range), max(strike_range), len(iv_smooth))
    iv_returns = iv_strikes / underlying_price - 1.0

    # Create a secondary y-axis for top plot
    ax1_2 = ax1.twinx()

    # Plot total IV curve
    ax1_2.plot(iv_returns, iv_smooth, 'green', alpha=0.8, linewidth=2, label=iv_label)

    # Add call and put IVs as scatter points
    # Plot call IVs
    if 'call_ivs' in volatility_data[trading_day][first_expiration]:
        call_ivs = np.array(volatility_data[trading_day][first_expiration]['call_ivs'])
        # Filter out NaNs or zeros
        valid_indices = ~np.isnan(call_ivs) & (call_ivs > 0)
        if np.any(valid_indices):
            call_strikes = strike_range[valid_indices]
            call_returns = call_strikes / underlying_price - 1.0
            call_iv_values = call_ivs[valid_indices]
            ax1_2.scatter(call_returns, call_iv_values, c='blue', s=25, alpha=0.6, label='Call IVs', marker='^')

    # Plot put IVs
    if 'put_ivs' in volatility_data[trading_day][first_expiration]:
        put_ivs = np.array(volatility_data[trading_day][first_expiration]['put_ivs'])
        # Filter out NaNs or zeros
        valid_indices = ~np.isnan(put_ivs) & (put_ivs > 0)
        if np.any(valid_indices):
            put_strikes = strike_range[valid_indices]
            put_returns = put_strikes / underlying_price - 1.0
            put_iv_values = put_ivs[valid_indices]
            ax1_2.scatter(put_returns, put_iv_values, c='red', s=25, alpha=0.6, label='Put IVs', marker='v')

    # Set labels for IV axis
    ax1_2.set_ylabel('Implied Volatility (annualized)', color='darkgreen', fontweight='bold')
    ax1_2.tick_params(axis='y', labelcolor='darkgreen')

    # Add a legend that includes both lines and IV points
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Optional: Set y-axis range for more consistent IV visualization
    try:
        max_iv = max(np.nanmax(iv_smooth),
                     np.nanmax(call_ivs[valid_indices]) if 'call_ivs' in locals() and np.any(valid_indices) else 0,
                     np.nanmax(put_ivs[valid_indices]) if 'put_ivs' in locals() and np.any(valid_indices) else 0)
        ax1_2.set_ylim(0, min(max_iv * 1.1, 2.0))  # Cap at 200% IV for readability
    except (ValueError, TypeError):
        pass  # If there's an issue with finding max values, don't set limits

def plot_iv_surface(volatility_data, trading_day, ticker, save_dir=None):
    """
    Plot implied volatility surface for a specific trading day
    """
    try:
        # Check if we have multiple expirations
        if len(volatility_data[trading_day]) < 2:
            print("Need at least 2 expirations to plot IV surface")
            return None

        # Get underlying price
        first_exp = next(iter(volatility_data[trading_day]))
        underlying_price = volatility_data[trading_day][first_exp]['underlyingPrice']

        # Collect data for all expirations
        expirations = []
        times_to_expiry = []
        strike_ranges = []
        iv_curves = []

        for exp in volatility_data[trading_day]:
            # First check for no arbitrage IV curve
            if 'iv_smooth_no_arb' in volatility_data[trading_day][exp]:
                iv_curve_key = 'iv_smooth_no_arb'
            # Fall back to standard IV curve
            elif 'iv_smooth' in volatility_data[trading_day][exp]:
                iv_curve_key = 'iv_smooth'
            else:
                # Skip expirations without IV data
                continue

            # Get time to expiration
            time_to_exp = volatility_data[trading_day][exp]['time_to_expiration']

            # Get IV curve
            iv_curve = np.array(volatility_data[trading_day][exp][iv_curve_key])

            # Generate proper strike range that matches iv_curve length exactly
            strike_range_raw = np.array(volatility_data[trading_day][exp]['strike_range'])
            strike_range = np.linspace(min(strike_range_raw), max(strike_range_raw), len(iv_curve))

            # Convert to returns
            strike_returns = strike_range / underlying_price - 1.0

            # Store data
            times_to_expiry.append(time_to_exp)
            expirations.append(exp)
            strike_ranges.append(strike_returns)
            iv_curves.append(iv_curve)

        if len(expirations) < 2:
            print("Need at least 2 valid expirations with IV data to plot surface")
            return None

        # Sort by time to expiration
        sorted_indices = np.argsort(times_to_expiry)
        expirations = [expirations[i] for i in sorted_indices]
        times_to_expiry = [times_to_expiry[i] for i in sorted_indices]
        strike_ranges = [strike_ranges[i] for i in sorted_indices]
        iv_curves = [iv_curves[i] for i in sorted_indices]

        # Create IV surface figure (APA-compliant)
        fig = plt.figure(figsize=(12, 10), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Create common return grid for all expirations
        min_return = min([min(sr) for sr in strike_ranges])
        max_return = max([max(sr) for sr in strike_ranges])
        common_returns = np.linspace(min_return, max_return, 100)

        # Plot IV surface
        from matplotlib import cm
        colors = [cm.Blues(0.7), cm.Reds(0.7)]  # Blue for short term, red for long term

        for i, (exp, time_to_exp, strike_returns, iv_curve) in enumerate(zip(
                expirations, times_to_expiry, strike_ranges, iv_curves)):

            # Make sure arrays have the same length
            if len(strike_returns) != len(iv_curve):
                print(f"Warning: Length mismatch for {exp} - strikes: {len(strike_returns)}, iv: {len(iv_curve)}")
                continue

            # Interpolate IV curve to common return grid
            from scipy.interpolate import interp1d
            iv_interp = interp1d(strike_returns, iv_curve,
                                bounds_error=False, fill_value='extrapolate')
            iv_values = iv_interp(common_returns)

            # Create mesh for this expiration
            X, Y = np.meshgrid([time_to_exp * 365] * len(common_returns), common_returns)

            # Plot surface segment
            if i == 0:
                color = 'blue'
                alpha = 0.7
            elif i == len(expirations) - 1:
                color = 'red'
                alpha = 0.7
            else:
                color = 'purple'
                alpha = 0.5

            ax.plot_surface(X, Y, iv_values.reshape(X.shape),
                           color=color, alpha=alpha, rstride=1, cstride=1)

        # Set labels and title (APA-compliant)
        ax.set_xlabel('Days to Expiration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (S_t/S_0 - 1)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Implied Volatility', fontsize=12, fontweight='bold')
        ax.set_title(f'{ticker}: Implied Volatility Surface on {trading_day}\nUsing No-Arbitrage IV when available',
                    fontsize=14, fontweight='bold')

        # Add colorbar
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(np.array([min([min(iv) for iv in iv_curves]),
                             max([max(iv) for iv in iv_curves])]))
        cbar = fig.colorbar(m, ax=ax, label='Implied Volatility')
        cbar.ax.set_ylabel('Implied Volatility', fontsize=10, fontweight='bold')

        # Set view angle
        ax.view_init(elev=30, azim=-45)

        # Add footnote
        fig.text(0.5, 0.01, f"Generated: {TIMESTAMP} | User: {USERNAME}",
                 ha='center', fontsize=8, color='black')

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig_path = f"{save_dir}/{ticker}_iv_surface_{trading_day.replace('/', '_')}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved IV surface plot to {fig_path}")

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error plotting IV surface: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def clear_results(ticker=None):
    """
    Clear all plots and data for a specific ticker or all tickers

    Parameters:
    ticker (str): Specific ticker to clear, or None to clear all
    """
    try:
        if ticker:
            results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
                print(f"Cleared results for {ticker}")
            else:
                print(f"No results directory found for {ticker}")
        else:
            for ticker in valid_tickers:
                results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    print(f"Cleared results for {ticker}")
    except Exception as e:
        print(f"Error clearing results: {str(e)}")

def find_last_processed_trading_day(ticker, valid_trading_days):
    """
    Find the last trading day that was processed for a specific ticker
    by checking existing plot files in the results directory.

    Parameters:
    ticker (str): Ticker symbol
    valid_trading_days (list): List of valid trading days in chronological order

    Returns:
    str: Last processed trading day or None if no processing has started
    """
    results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
    plots_dir = f"{results_dir}/plots"

    # If directory doesn't exist, no trading days have been processed
    if not os.path.exists(plots_dir):
        return None

    # Get list of all plot files
    plot_files = os.listdir(plots_dir)
    processed_days = []

    # Extract trading days from filenames
    for file in plot_files:
        if file.startswith(ticker) and file.endswith('.png'):
            # Extract date part (format ticker_MM_DD_YYYY.png)
            date_part = file.replace(ticker + '_', '').replace('.png', '')
            if '_' in date_part:
                try:
                    # Convert back to MM/DD/YYYY format
                    parts = date_part.split('_')
                    if len(parts) >= 3:  # Account for timestamp in filename
                        trading_day = f"{parts[0]}/{parts[1]}/{parts[2]}"
                        if trading_day in valid_trading_days:
                            processed_days.append(trading_day)
                except:
                    continue

    # Return the last processed day if any
    if processed_days:
        # Sort to find the most recent processed day
        processed_days.sort(key=lambda x: valid_trading_days.index(x) if x in valid_trading_days else -1)
        return processed_days[-1]

    return None

def get_valid_trading_days(ticker_data, target_expiry_days=[6, 7, 8, 9, 10], allowed_range=0):
    """
    Get valid trading days for a ticker with options EXACTLY matching the target days

    Parameters:
    ticker_data: Dictionary with volatility data
    target_expiry_days: List of target days to expiration [6, 7, 8, 9, 10]
    allowed_range: Must be 0 for exact matching only

    Returns:
    list: List of valid trading days
    """
    valid_days = []

    for day in ticker_data:
        print(day)
        for expiration in ticker_data[day]:
            print(f"{expiration}")
            days_to_expiry = int(ticker_data[day][expiration]['time_to_expiration'] * 365)
            print(days_to_expiry)

            # Check if this EXACTLY matches any of our target days (no tolerance)
            if days_to_expiry in target_expiry_days:
                valid_days.append(day)
                break  # Found a match for this day, move to next day

    return sorted(valid_days)

def find_progress_for_all_tickers(valid_tickers, all_volatility_data=None):
    """
    Find progress for all tickers and determine which to process next based on completion

    Parameters:
    valid_tickers (list): List of valid tickers
    all_volatility_data (dict): Dictionary containing volatility data for each ticker

    Returns:
    list: Tickers sorted by processing progress (least complete first)
    """
    progress = {}

    for ticker in valid_tickers:
        # Check if data file exists without loading it
        filepath = f'/content/drive/MyDrive/{ticker}analyzed4.2_optimized_data.json'
        if not os.path.exists(filepath):
            print(f"No data file exists for {ticker}")
            progress[ticker] = -1  # Mark as not processable
            continue

        results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
        plots_dir = f"{results_dir}/plots"

        # If directory doesn't exist, no processing has started
        if not os.path.exists(plots_dir):
            progress[ticker] = 0
            processed_days = 0
        else:
            # Count processed days only if the directory exists
            processed_days = len([f for f in os.listdir(plots_dir)
                                 if f.startswith(ticker) and f.endswith('.png')])

            # Try to get total days
            if all_volatility_data and ticker in all_volatility_data:
                all_trading_days = get_valid_trading_days(all_volatility_data[ticker])
                total_days = len(all_trading_days)
            else:
                # Estimate total days (can be refined if needed)
                total_days = 1000  # Approximate number of trading days per ticker

            # Calculate progress percentage
            progress[ticker] = processed_days / total_days if total_days > 0 else 0

    # Sort tickers by progress (least complete first, but skip those with no data)
    sorted_tickers = sorted([t for t in valid_tickers if progress[t] >= 0],
                           key=lambda x: progress.get(x, 0))

    print("Ticker processing progress:")
    for ticker in valid_tickers:
        if progress[ticker] < 0:
            print(f"  {ticker}: No data file available")
        else:
            plots_dir = f'/content/drive/MyDrive/{ticker}_analysis_results/plots'
            if os.path.exists(plots_dir):
                processed_days = len([f for f in os.listdir(plots_dir)
                                    if f.startswith(ticker) and f.endswith('.png')])
            else:
                processed_days = 0

            pct = progress.get(ticker, 0) * 100
            print(f"  {ticker}: {processed_days} days processed (~{pct:.1f}% complete)")

    return sorted_tickers

def readjust_time_series_data():
    """
    Readjust time series data to fix any issues or recalculate moments
    """
    print("\nTime Series Data Readjustment Tool")
    print("---------------------------------")
    print("This tool helps fix issues with time series data and recalculate moments.")

    # Get list of tickers with saved data
    ticker_options = []
    for ticker in valid_tickers:
        results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'

        # Check if directory exists and has time series data
        if os.path.exists(results_dir):
            # Check for standard moments file
            if os.path.exists(f"{results_dir}/{ticker}_moments_summary.csv"):
                ticker_options.append((ticker, f"{results_dir}/{ticker}_moments_summary.csv", "all expirations"))

            # Check for expiration-specific files
            exp_dir = f"{results_dir}/expiration_specific"
            if os.path.exists(exp_dir):
                # Look for specific day files
                for day_length in [6, 7, 8, 9, 10]:
                    exp_file = f"{exp_dir}/{ticker}_{day_length}day_options_summary.csv"
                    if os.path.exists(exp_file):
                        ticker_options.append((ticker, exp_file, f"{day_length}-day"))

    if not ticker_options:
        print("No ticker time series data found to readjust.")
        return

    # Display available tickers
    print("\nAvailable ticker data:")
    for i, (ticker, file_path, expiry_info) in enumerate(ticker_options):
        print(f"{i+1}. {ticker} ({expiry_info})")

    # Get user selection
    selection = input("\nSelect data to readjust (number) or 'all' for all: ")

    files_to_process = []
    if selection.lower() == 'all':
        files_to_process = ticker_options
    else:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(ticker_options):
                files_to_process = [ticker_options[idx]]
            else:
                print("Invalid selection. Please try again.")
                return
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")
            return

    # Process each selected file
    for ticker, csv_file, expiry_info in files_to_process:
        print(f"\nProcessing {ticker} ({expiry_info})...")

        # Determine output directories
        if "all expirations" in expiry_info:
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/time_series"
            day_length = None
        else:
            # Extract days from expiry_info
            day_length = int(expiry_info.split("-")[0])
            time_series_dir = f"/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific/{day_length}day_time_series"

        # Create time series directory if it doesn't exist
        os.makedirs(time_series_dir, exist_ok=True)

        try:
            # Load the data
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} records from {csv_file}")

            # Make a backup
            backup_file = csv_file.replace('.csv', '_backup.csv')
            df.to_csv(backup_file, index=False)
            print(f"Created backup at {backup_file}")

            # Fix any issues with the data - examples:

            # 1. Remove extreme outliers
            for col in ['realized_skewness', 'realized_kurtosis', 'rnd_skewness', 'rnd_kurtosis']:
                if col in df.columns:
                    old_count = len(df)
                    df = remove_outliers_from_moments(df, col, n_std=5)  # Using 5 std for extreme outliers only
                    new_count = len(df)
                    if old_count != new_count:
                        print(f"  Removed {old_count - new_count} extreme outliers from {col}")

            # 2. Convert trading_day if not already datetime
            if 'trading_day' in df.columns and not pd.api.types.is_datetime64_dtype(df['trading_day']):
                df['trading_day'] = pd.to_datetime(df['trading_day'], format='%m/%d/%Y')
                print("  Converted trading_day to datetime format")

            # 3. Ensure consistent annualization
            if 'realized_variance' in df.columns and 'days_to_expiry' in df.columns and 'realized_annualized_variance' in df.columns:
                # Check if annualization needs to be fixed
                sample = df.iloc[0]
                expected_annualized = sample['realized_variance'] * (252.0 / sample['days_to_expiry'])
                if abs(expected_annualized - sample['realized_annualized_variance']) / expected_annualized > 0.01:  # >1% difference
                    print("  Fixing inconsistent annualization...")
                    df['realized_annualized_variance'] = df['realized_variance'] * (252.0 / df['days_to_expiry'])
                    df['realized_annualized_vol'] = np.sqrt(df['realized_annualized_variance'])

            # 4. Sort by trading day
            if 'trading_day' in df.columns:
                df = df.sort_values('trading_day')
                print("  Sorted data by trading day")

            # Save the fixed data
            df.to_csv(csv_file, index=False)
            print(f"Saved fixed data to {csv_file}")

            # Regenerate time series plots
            print("Regenerating time series plots with fixed data...")

            # Convert back to list of dicts for plotting function
            plot_skewness_kurtosis_time_series(
                df.to_dict('records'),
                f"{ticker} ({expiry_info})",
                save_dir=time_series_dir
            )

            print(f"Successfully readjusted data and regenerated plots for {ticker} ({expiry_info})")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nData readjustment completed.")

def check_expiration_specific_progress(valid_tickers):
    """
    Check progress for expiration-specific processing and determine which ticker needs processing next

    Parameters:
    valid_tickers (list): List of valid tickers

    Returns:
    tuple: (ticker_to_process, resume_day, needs_processing, needs_plots, completed_days)
    """
    ticker_progress = {}

    for ticker in valid_tickers:
        # Check if data file exists
        data_file = f'/content/drive/MyDrive/{ticker}analyzed4.2_optimized_data.json'
        if not os.path.exists(data_file):
            ticker_progress[ticker] = {
                'data_available': False,
                'completed_days': [],
                'processing_needed': False,
                'plots_needed': False,
                'last_day': None
            }
            continue

        # Check for expiration-specific directory
        exp_dir = f'/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific'

        ticker_progress[ticker] = {
            'data_available': True,
            'completed_days': [],
            'processing_needed': False,
            'plots_needed': False,
            'last_day': None
        }

        # Check which day lengths are already processed
        for day_length in [6, 7, 8, 9, 10]:
            csv_file = f"{exp_dir}/{ticker}_{day_length}day_options_summary.csv"
            if os.path.exists(csv_file):
                ticker_progress[ticker]['completed_days'].append(day_length)

                # Check if time series plots exist
                plots_dir = f"{exp_dir}/{day_length}day_time_series"
                if not os.path.exists(plots_dir) or len(os.listdir(plots_dir)) < 3:
                    ticker_progress[ticker]['plots_needed'] = True

                # Find the last processed day for this length
                try:
                    df = pd.read_csv(csv_file)
                    last_day = df['trading_day'].iloc[-1] if len(df) > 0 else None
                    if last_day:
                        ticker_progress[ticker]['last_day'] = last_day
                except Exception:
                    pass

        # If not all day lengths are processed, processing is needed
        if len(ticker_progress[ticker]['completed_days']) < 5:  # 5 = all day lengths
            ticker_progress[ticker]['processing_needed'] = True

    # Find the ticker with the most progress that still needs work
    ticker_to_process = None
    max_completed_days = -1
    resume_day = None

    for ticker, progress in ticker_progress.items():
        if not progress['data_available']:
            continue

        if progress['processing_needed'] or progress['plots_needed']:
            if len(progress['completed_days']) > max_completed_days:
                ticker_to_process = ticker
                max_completed_days = len(progress['completed_days'])
                resume_day = progress['last_day']

    if ticker_to_process:
        return (
            ticker_to_process,
            resume_day,
            ticker_progress[ticker_to_process]['processing_needed'],
            ticker_progress[ticker_to_process]['plots_needed'],
            ticker_progress[ticker_to_process]['completed_days']
        )
    else:
        return None, None, False, False, []

import logging

def setup_logging():
    """Set up comprehensive logging for the analysis process"""
    log_dir = '/content/drive/MyDrive/qvc_analysis_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"{log_dir}/qvc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== QVC Analysis Session Started ===")
    logger.info(f"User: {USERNAME}")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Log file: {log_filename}")
    
    return logger

def get_process_state():
    """
    Check the current state of processing for all tickers and return detailed status
    
    Returns:
    dict: Comprehensive state information for all tickers
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Process State ===")
    
    state = {
        'overall_status': 'checking',
        'tickers': {},
        'next_actions': [],
        'completion_summary': {}
    }
    
    target_days = [6, 7, 8, 9, 10]
    
    for ticker in valid_tickers:
        logger.info(f"Checking state for {ticker}...")
        
        ticker_state = {
            'data_file_exists': False,
            'minute_data_available': False,
            'expiration_processing': {},
            'time_series_plots': {},
            'completion_percentage': 0,
            'current_step': 'not_started',
            'issues': [],
            'next_action': None
        }
        
        # Check if data file exists
        data_file = f'/content/drive/MyDrive/{ticker}analyzed4.2_optimized_data.json'
        if os.path.exists(data_file):
            ticker_state['data_file_exists'] = True
            logger.info(f"  ✓ Data file exists for {ticker}")
        else:
            ticker_state['issues'].append('Data file missing')
            logger.warning(f"  ✗ Data file missing for {ticker}")
        
        # Check minute data availability
        minute_file = f"/content/drive/MyDrive/{ticker}_minute_data.csv"
        if os.path.exists(minute_file):
            ticker_state['minute_data_available'] = True
            logger.info(f"  ✓ Minute data available for {ticker}")
        else:
            ticker_state['issues'].append('Minute data missing')
            logger.warning(f"  ✗ Minute data missing for {ticker}")
        
        # Check expiration-specific processing
        exp_dir = f'/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific'
        completed_days = 0
        
        for day_length in target_days:
            day_state = {
                'csv_exists': False,
                'csv_record_count': 0,
                'time_series_plots_exist': False,
                'last_trading_day': None,
                'status': 'not_started'
            }
            
            # Check CSV file
            csv_file = f"{exp_dir}/{ticker}_{day_length}day_options_summary.csv"
            if os.path.exists(csv_file):
                day_state['csv_exists'] = True
                try:
                    df = pd.read_csv(csv_file)
                    day_state['csv_record_count'] = len(df)
                    if len(df) > 0:
                        day_state['last_trading_day'] = df['trading_day'].iloc[-1]
                        day_state['status'] = 'processing_complete'
                        completed_days += 1
                        logger.info(f"    ✓ {day_length}-day: {len(df)} records, last day: {day_state['last_trading_day']}")
                except Exception as e:
                    day_state['status'] = 'csv_corrupted'
                    ticker_state['issues'].append(f'{day_length}-day CSV corrupted: {str(e)}')
                    logger.error(f"    ✗ {day_length}-day CSV corrupted: {str(e)}")
            else:
                logger.info(f"    - {day_length}-day: Not started")
            
            # Check time series plots
            plots_dir = f"{exp_dir}/{day_length}day_time_series"
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                if len(plot_files) >= 3:  # Expect at least 3 plot files
                    day_state['time_series_plots_exist'] = True
                    logger.info(f"      ✓ Time series plots exist ({len(plot_files)} files)")
                else:
                    logger.info(f"      - Incomplete time series plots ({len(plot_files)} files)")
            
            ticker_state['expiration_processing'][day_length] = day_state
        
        # Calculate completion percentage
        ticker_state['completion_percentage'] = (completed_days / len(target_days)) * 100
        
        # Determine current step and next action
        if not ticker_state['data_file_exists']:
            ticker_state['current_step'] = 'data_missing'
            ticker_state['next_action'] = 'skip_no_data'
        elif not ticker_state['minute_data_available']:
            ticker_state['current_step'] = 'minute_data_missing'
            ticker_state['next_action'] = 'skip_no_minute_data'
        elif completed_days == 0:
            ticker_state['current_step'] = 'ready_to_start'
            ticker_state['next_action'] = 'start_processing'
        elif completed_days < len(target_days):
            ticker_state['current_step'] = 'partial_processing'
            ticker_state['next_action'] = 'continue_processing'
        else:
            # Check if time series plots need generation
            plots_needed = False
            for day_length in target_days:
                if not ticker_state['expiration_processing'][day_length]['time_series_plots_exist']:
                    plots_needed = True
                    break
            
            if plots_needed:
                ticker_state['current_step'] = 'plots_needed'
                ticker_state['next_action'] = 'generate_plots'
            else:
                ticker_state['current_step'] = 'complete'
                ticker_state['next_action'] = 'none'
        
        state['tickers'][ticker] = ticker_state
        
        # Log summary for this ticker
        logger.info(f"  Status: {ticker_state['current_step']} ({ticker_state['completion_percentage']:.1f}% complete)")
        if ticker_state['issues']:
            logger.warning(f"  Issues: {', '.join(ticker_state['issues'])}")
    
    # Determine overall status and next actions
    processable_tickers = [t for t in valid_tickers 
                          if state['tickers'][t]['data_file_exists'] and 
                             state['tickers'][t]['minute_data_available']]
    
    if not processable_tickers:
        state['overall_status'] = 'no_processable_tickers'
        logger.error("No tickers have both data file and minute data available")
    else:
        # Find tickers that need processing
        need_processing = [t for t in processable_tickers 
                          if state['tickers'][t]['next_action'] in ['start_processing', 'continue_processing']]
        need_plots = [t for t in processable_tickers 
                     if state['tickers'][t]['next_action'] == 'generate_plots']
        
        if need_processing:
            state['overall_status'] = 'processing_needed'
            state['next_actions'] = need_processing
        elif need_plots:
            state['overall_status'] = 'plots_needed'
            state['next_actions'] = need_plots
        else:
            state['overall_status'] = 'all_complete'
            state['next_actions'] = []
    
    # Create completion summary
    for ticker in valid_tickers:
        ticker_info = state['tickers'][ticker]
        state['completion_summary'][ticker] = {
            'status': ticker_info['current_step'],
            'completion_pct': ticker_info['completion_percentage'],
            'issues': len(ticker_info['issues'])
        }
    
    logger.info(f"=== Process State Check Complete ===")
    logger.info(f"Overall Status: {state['overall_status']}")
    logger.info(f"Next Actions: {state['next_actions']}")
    
    return state

def process_ticker_comprehensive(ticker, state, treasury_rates, logger):
    """
    Process a single ticker comprehensively with detailed logging
    
    Parameters:
    ticker (str): Ticker symbol to process
    state (dict): Current process state
    treasury_rates: Treasury rates data
    logger: Logger instance
    
    Returns:
    dict: Processing results
    """
    logger.info(f"=== Starting Comprehensive Processing for {ticker} ===")
    
    results = {
        'ticker': ticker,
        'success': False,
        'steps_completed': [],
        'steps_failed': [],
        'trading_days_processed': 0,
        'trading_days_failed': 0,
        'detailed_log': []
    }
    
    try:
        # Step 1: Load data files
        logger.info(f"Step 1: Loading data files for {ticker}")
        
        # Load volatility data
        data_file = f'/content/drive/MyDrive/{ticker}analyzed4.2_optimized_data.json'
        logger.info(f"Loading volatility data from: {data_file}")
        with open(data_file, "r") as file:
            volatility_data = json.load(file)
        logger.info(f"✓ Loaded volatility data: {len(volatility_data)} trading days")
        results['steps_completed'].append('load_volatility_data')
        
        # Load minute data
        minute_data = load_minute_data(ticker)
        if len(minute_data) == 0:
            raise Exception("No minute data available")
        logger.info(f"✓ Loaded minute data: {len(minute_data)} records")
        results['steps_completed'].append('load_minute_data')
        
        # Step 2: Set up directories
        logger.info(f"Step 2: Setting up directories for {ticker}")
        results_dir = f'/content/drive/MyDrive/{ticker}_analysis_results'
        expiration_dir = f"{results_dir}/expiration_specific"
        os.makedirs(expiration_dir, exist_ok=True)
        logger.info(f"✓ Results directory: {results_dir}")
        results['steps_completed'].append('setup_directories')
        
        # Step 3: Process each expiration length
        target_days = [6, 7, 8, 9, 10]
        
        for day_length in target_days:
            logger.info(f"Step 3.{day_length}: Processing {day_length}-day expirations for {ticker}")
            
            # Check if already completed
            csv_file = f"{expiration_dir}/{ticker}_{day_length}day_options_summary.csv"
            if os.path.exists(csv_file):
                try:
                    existing_df = pd.read_csv(csv_file)
                    if len(existing_df) > 0:
                        logger.info(f"  ✓ {day_length}-day already processed ({len(existing_df)} records)")
                        results['steps_completed'].append(f'process_{day_length}_day')
                        continue
                except Exception as e:
                    logger.warning(f"  Existing CSV corrupted, reprocessing: {str(e)}")
            
            # Get trading days for this expiration length
            expiration_days = get_specific_expiration_days(volatility_data, [day_length], tolerance=0)
            trading_days_list = expiration_days.get(day_length, [])
            
            if not trading_days_list:
                logger.warning(f"  No trading days found for exactly {day_length}-day expirations")
                results['steps_failed'].append(f'no_data_{day_length}_day')
                continue
            
            logger.info(f"  Found {len(trading_days_list)} trading days for {day_length}-day expirations")
            
            # Process each trading day with detailed logging
            day_moments = []
            processed_count = 0
            failed_count = 0
            
            for i, trading_day_tuple in enumerate(trading_days_list):
                trading_day, expiration_date, exact_days = trading_day_tuple
                
                try:
                    # Log every 20th trading day for progress tracking
                    if i % 20 == 0:
                        logger.info(f"    Processing trading day {i+1}/{len(trading_days_list)}: {trading_day} (expires in {exact_days} days)")
                    
                    # Calculate comprehensive moments
                    result = compare_rnd_to_realized_comprehensive(
                        volatility_data,
                        trading_day,
                        ticker,
                        minute_data,
                        treasury_rates=treasury_rates,
                        x_range_limits=(-0.5, 0.5),
                        include_overnight=True,
                        show_plot=False,
                        save_results=False  # Don't save individual results to speed up
                    )
                    
                    if result is not None:
                        _, _, stats_dict = result
                        
                        # Add Amaya scaling
                        trading_day_dt = pd.to_datetime(trading_day, format='%m/%d/%Y')
                        expiration_day_dt = pd.to_datetime(expiration_date, format='%m/%d/%Y')
                        
                        amaya_moments = calculate_realized_higher_moments_amaya(
                            minute_data, trading_day_dt, expiration_day_dt, stats_dict
                        )
                        
                        if amaya_moments:
                            stats_dict['original_realized_skewness'] = stats_dict.get('realized_skewness', 0)
                            stats_dict['original_realized_kurtosis'] = stats_dict.get('realized_kurtosis', 3)
                            stats_dict['realized_skewness'] = amaya_moments['realized_skewness']
                            stats_dict['realized_kurtosis'] = amaya_moments['realized_kurtosis']
                            stats_dict['amaya_scaling_applied'] = True
                        else:
                            stats_dict['amaya_scaling_applied'] = False
                        
                        day_moments.append(stats_dict)
                        processed_count += 1
                        
                    else:
                        failed_count += 1
                        error_msg = f"No result returned for {trading_day}"
                        if failed_count <= 5:  # Only log first 5 failures to avoid spam
                            logger.debug(f"    Failed: {error_msg}")
                        results['detailed_log'].append(f"{trading_day}: {error_msg}")
                        
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    if failed_count <= 5:  # Only log first 5 failures to avoid spam
                        logger.debug(f"    Failed {trading_day}: {error_msg}")
                    results['detailed_log'].append(f"{trading_day}: {error_msg}")
            
            # Save results for this day length
            if day_moments:
                df = pd.DataFrame(day_moments)
                df.to_csv(csv_file, index=False)
                logger.info(f"  ✓ Saved {len(day_moments)} records to {csv_file}")
                
                results['trading_days_processed'] += processed_count
                results['trading_days_failed'] += failed_count
                
                # Log processing summary
                total_days = processed_count + failed_count
                success_rate = (processed_count / total_days) * 100 if total_days > 0 else 0
                logger.info(f"  Summary: {processed_count}/{total_days} successful ({success_rate:.1f}%)")
                
                # Log reasons for failures if there are many
                if failed_count > 10:
                    failure_sample = results['detailed_log'][-min(3, failed_count):]
                    logger.info(f"  Sample failures: {failure_sample}")
                
            else:
                logger.error(f"  No valid results for {day_length}-day expirations")
                results['steps_failed'].append(f'process_{day_length}_day')
                continue
            
            results['steps_completed'].append(f'process_{day_length}_day')
        
        # Step 4: Generate time series plots for all completed expiration lengths
        logger.info(f"Step 4: Generating time series plots for {ticker}")
        
        for day_length in target_days:
            csv_file = f"{expiration_dir}/{ticker}_{day_length}day_options_summary.csv"
            if os.path.exists(csv_file):
                try:
                    logger.info(f"  Creating time series plots for {day_length}-day options...")
                    
                    # Load data
                    df = pd.read_csv(csv_file)
                    
                    # Create time series directory
                    time_series_dir = f"{expiration_dir}/{day_length}day_time_series"
                    os.makedirs(time_series_dir, exist_ok=True)
                    
                    # Create plots
                    plot_skewness_kurtosis_time_series(
                        df.to_dict('records'),
                        f"{ticker} ({day_length}-day)",
                        save_dir=time_series_dir
                    )
                    
                    # Create Amaya comparison plots if applicable
                    if 'original_realized_skewness' in df.columns:
                        plot_amaya_comparison(
                            df.to_dict('records'),
                            ticker,
                            day_length,
                            save_dir=time_series_dir
                        )
                    
                    logger.info(f"  ✓ Created time series plots for {day_length}-day options")
                    results['steps_completed'].append(f'plots_{day_length}_day')
                    
                except Exception as e:
                    logger.error(f"  Failed to create plots for {day_length}-day: {str(e)}")
                    results['steps_failed'].append(f'plots_{day_length}_day')
        
        results['success'] = True
        logger.info(f"=== Successfully completed processing for {ticker} ===")
        
    except Exception as e:
        logger.error(f"=== Failed processing for {ticker}: {str(e)} ===")
        results['steps_failed'].append('major_error')
        results['detailed_log'].append(f"Major error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return results

def main():
    """
    Streamlined main function that automatically processes all tickers for specific expiration lengths
    with comprehensive logging and resume capability
    """
    # Set up logging
    logger = setup_logging()
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Analysis started at {start_time}")
    
    # Update global variables with current timestamp
    global TIMESTAMP, USERNAME
    TIMESTAMP = "2025-05-31 19:06:43"  # Current UTC time
    USERNAME = "testtesttest703"
    
    logger.info(f"Processing expiration-specific analysis for tickers: {valid_tickers}")
    logger.info(f"Target expiration days: [6, 7, 8, 9, 10]")
    
    try:
        # Step 1: Check current process state
        logger.info("=== STEP 1: Checking Process State ===")
        state = get_process_state()
        
        # Display current state summary
        logger.info("=== Current State Summary ===")
        for ticker in valid_tickers:
            ticker_state = state['tickers'][ticker]
            logger.info(f"{ticker}: {ticker_state['current_step']} "
                       f"({ticker_state['completion_percentage']:.1f}% complete) "
                       f"- Next: {ticker_state['next_action']}")
            
            if ticker_state['issues']:
                logger.warning(f"  Issues: {', '.join(ticker_state['issues'])}")
        
        # Step 2: Load Treasury rates data (needed for all processing)
        logger.info("=== Loading Treasury Rates Data ===")
        treasury_rates = load_treasury_rates()
        if treasury_rates is None:
            logger.warning("Treasury rates not available, using default rates")
        
        # Step 3: Process tickers based on their state
        logger.info("=== STEP 2: Processing Tickers ===")
        
        if state['overall_status'] == 'no_processable_tickers':
            logger.error("No tickers can be processed due to missing data files")
            return
        
        elif state['overall_status'] == 'all_complete':
            logger.info("All tickers are fully processed! No further action needed.")
            
            # Still generate time series plots if any are missing
            logger.info("Checking for missing time series plots...")
            plots_generated = 0
            for ticker in valid_tickers:
                ticker_state = state['tickers'][ticker]
                if ticker_state['data_file_exists'] and ticker_state['minute_data_available']:
                    expiration_dir = f'/content/drive/MyDrive/{ticker}_analysis_results/expiration_specific'
                    
                    for day_length in [6, 7, 8, 9, 10]:
                        csv_file = f"{expiration_dir}/{ticker}_{day_length}day_options_summary.csv"
                        time_series_dir = f"{expiration_dir}/{day_length}day_time_series"
                        
                        if os.path.exists(csv_file) and not os.path.exists(f"{time_series_dir}/{ticker}_{day_length}day_skewness_time_series.png"):
                            logger.info(f"Generating missing plots for {ticker} {day_length}-day...")
                            try:
                                df = pd.read_csv(csv_file)
                                os.makedirs(time_series_dir, exist_ok=True)
                                
                                plot_skewness_kurtosis_time_series(
                                    df.to_dict('records'),
                                    f"{ticker} ({day_length}-day)",
                                    save_dir=time_series_dir
                                )
                                
                                if 'original_realized_skewness' in df.columns:
                                    plot_amaya_comparison(
                                        df.to_dict('records'),
                                        ticker,
                                        day_length,
                                        save_dir=time_series_dir
                                    )
                                
                                plots_generated += 1
                                logger.info(f"✓ Generated plots for {ticker} {day_length}-day")
                                
                            except Exception as e:
                                logger.error(f"Failed to generate plots for {ticker} {day_length}-day: {str(e)}")
            
            if plots_generated > 0:
                logger.info(f"Generated {plots_generated} missing plot sets")
            else:
                logger.info("All plots are already complete")
            return
        
        else:
            # Process tickers that need work
            tickers_to_process = state['next_actions']
            logger.info(f"Processing {len(tickers_to_process)} tickers: {tickers_to_process}")
            
            processing_results = {}
            
            for ticker in tickers_to_process:
                logger.info(f"\n{'='*50}")
                logger.info(f"PROCESSING TICKER: {ticker}")
                logger.info(f"{'='*50}")
                
                # Process this ticker comprehensively
                results = process_ticker_comprehensive(ticker, state, treasury_rates, logger)
                processing_results[ticker] = results
                
                # Log results summary
                if results['success']:
                    logger.info(f"✓ {ticker} completed successfully")
                    logger.info(f"  Steps completed: {len(results['steps_completed'])}")
                    logger.info(f"  Trading days processed: {results['trading_days_processed']}")
                    logger.info(f"  Trading days failed: {results['trading_days_failed']}")
                    if results['trading_days_processed'] > 0:
                        success_rate = (results['trading_days_processed'] / 
                                      (results['trading_days_processed'] + results['trading_days_failed'])) * 100
                        logger.info(f"  Success rate: {success_rate:.1f}%")
                else:
                    logger.error(f"✗ {ticker} failed")
                    logger.error(f"  Steps failed: {results['steps_failed']}")
                    if results['detailed_log']:
                        logger.error(f"  First few errors: {results['detailed_log'][:3]}")
        
        # Step 4: Final state check and summary
        logger.info("=== STEP 3: Final State Check ===")
        final_state = get_process_state()
        
        # Create final summary
        logger.info("=== FINAL PROCESSING SUMMARY ===")
        total_processed = 0
        total_failed = 0
        
        for ticker in valid_tickers:
            final_ticker_state = final_state['tickers'][ticker]
            logger.info(f"{ticker}: {final_ticker_state['current_step']} "
                       f"({final_ticker_state['completion_percentage']:.1f}% complete)")
            
            if ticker in processing_results:
                results = processing_results[ticker]
                logger.info(f"  This session - Processed: {results['trading_days_processed']} days, "
                           f"Failed: {results['trading_days_failed']} days")
                total_processed += results['trading_days_processed']
                total_failed += results['trading_days_failed']
        
        # Record end time
        end_time = datetime.now()
        runtime = end_time - start_time
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Started: {start_time}")
        logger.info(f"Ended: {end_time}")
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Total trading days processed this session: {total_processed}")
        logger.info(f"Total trading days failed this session: {total_failed}")
        if total_processed + total_failed > 0:
            session_success_rate = (total_processed / (total_processed + total_failed)) * 100
            logger.info(f"Session success rate: {session_success_rate:.1f}%")
        logger.info(f"User: {USERNAME}")
        
        # Check if any tickers still need work
        if final_state['overall_status'] != 'all_complete':
            incomplete_tickers = [t for t in valid_tickers 
                                if final_state['tickers'][t]['completion_percentage'] < 100]
            logger.info(f"Note: {len(incomplete_tickers)} tickers still need processing: {incomplete_tickers}")
            logger.info("Run the script again to continue processing.")
        else:
            logger.info("🎉 ALL TICKERS ARE NOW FULLY PROCESSED!")
            logger.info("All expiration-specific analysis complete with time series plots generated.")
            
    except Exception as e:
        logger.error(f"Fatal error in main processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
