from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import date, timedelta, datetime # Added datetime for type checking
import logging
from PIL import Image
import io
import holidays as pyholidays # For public holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Helper function to get holidays for Prophet ---
def get_market_holidays(years, country='US'):
    market_holidays_lib = pyholidays.CountryHoliday(country, years=years, prov=None, state=None)
    df_holidays = pd.DataFrame(columns=['holiday', 'ds'])
    if market_holidays_lib:
        holiday_dates = []
        holiday_names = []
        for dt, name in sorted(market_holidays_lib.items()):
            holiday_dates.append(dt)
            holiday_names.append(name)
        df_holidays['ds'] = pd.to_datetime(holiday_dates)
        df_holidays['holiday'] = holiday_names
        # Optional: Filter out Saturdays and Sundays if the holidays library includes them
        # df_holidays = df_holidays[df_holidays['ds'].dt.dayofweek < 5]
        logger.info(f"Generated {len(df_holidays)} holidays for {country} for years: {years}")
    return df_holidays

# --- Image Analysis (Placeholder) ---
def analyze_chart_image(image_bytes):
    if not image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        width, height = img.size
        if (width * height) % 100 < 33:
             logger.info("Mock Image Analysis: Trend UP")
             return "up"
        elif (width * height) % 100 < 66:
             logger.info("Mock Image Analysis: Trend NEUTRAL")
             return "neutral"
        else:
             logger.info("Mock Image Analysis: Trend DOWN")
             return "down"
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return None

# --- Stock Data Fetching ---
def get_stock_data(ticker_symbol, period="5y"):
    try:
        logger.info(f"Fetching data for ticker: {ticker_symbol}")
        stock = yf.Ticker(ticker_symbol)
        hist_data = stock.history(period=period)
        if hist_data.empty:
            logger.warning(f"No historical data found for {ticker_symbol} via yfinance history.")
            # Fallback: try to get some info if history is empty
            info = stock.info
            if info and info.get('regularMarketPrice') is not None and info.get('previousClose') is not None:
                 logger.warning(f"History empty for {ticker_symbol}, but info found. This might indicate issues.")
                 # This path means we can't train, but it's good to know.
            return None # Return None if no actual OHLCV history
            
        hist_data.reset_index(inplace=True)
        hist_data['ds'] = pd.to_datetime(hist_data['Date']).dt.tz_localize(None)
        hist_data['y'] = hist_data['Close']
        logger.info(f"Successfully fetched and processed data for {ticker_symbol}")
        return hist_data[['ds', 'y']]
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}", exc_info=True)
        return None

# --- Model Training and Prediction Logic ---
def train_and_forecast(data, days_to_forecast_daily, days_to_forecast_weekly, 
                       image_trend_regressor=None, country_holidays='US'):
    if data is None or data.empty or len(data) < 2: # Prophet needs at least 2 data points
        logger.warning("Not enough data to train Prophet model.")
        return None, None, None

    min_hist_year = data['ds'].min().year
    max_forecast_date_approx = data['ds'].max() + pd.Timedelta(days=max(days_to_forecast_daily, days_to_forecast_weekly) + 60) # Increased buffer
    max_hist_year = max_forecast_date_approx.year
    holiday_years = list(range(min_hist_year, max_hist_year + 1))
    holidays_df = get_market_holidays(years=holiday_years, country=country_holidays)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        holidays=holidays_df if not holidays_df.empty else None
    )

    if image_trend_regressor is not None and image_trend_regressor in ["up", "down", "neutral"]:
        if image_trend_regressor == "up": trend_value = 1
        elif image_trend_regressor == "down": trend_value = -1
        else: trend_value = 0
        data['image_sentiment'] = trend_value
        model.add_regressor('image_sentiment')
        logger.info(f"Prepared 'image_sentiment' regressor for fitting with value: {trend_value}")
    
    try:
        model.fit(data)
    except Exception as e:
        logger.error(f"Error during model.fit: {e}", exc_info=True)
        return None, None, None

    future_daily_df = model.make_future_dataframe(periods=days_to_forecast_daily)
    future_weekly_df = model.make_future_dataframe(periods=days_to_forecast_weekly)

    if 'image_sentiment' in data.columns:
        if image_trend_regressor == "up": future_trend_value = 1
        elif image_trend_regressor == "down": future_trend_value = -1
        else: future_trend_value = 0
        if image_trend_regressor in ["up", "down", "neutral"]:
            future_daily_df['image_sentiment'] = future_trend_value
            future_weekly_df['image_sentiment'] = future_trend_value

    try:
        forecast_daily = model.predict(future_daily_df)
        forecast_weekly = model.predict(future_weekly_df)
    except Exception as e:
        logger.error(f"Error during model.predict: {e}", exc_info=True)
        return None, None, None

    return model, forecast_daily, forecast_weekly

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast_with_image', methods=['POST'])
def forecast_stock_with_image():
    # Initialize variables to ensure they have default values
    image_trend = None
    image_sentiment_for_response = "Not Provided"
    stock_data_df = None
    current_price = None # Will be float or None
    last_date = None     # Will be datetime or None
    three_day_forecast_results = []
    week_ahead_pred_details = {
        "date": "N/A", "price": "N/A", 
        "lower_bound": "N/A", "upper_bound": "N/A", 
        "recommendation": "N/A"
    }
    ticker_symbol = "N/A" # Default if not found in form
    market_country = "US" # Default if not found

    try:
        ticker_symbol = request.form.get('ticker', 'N/A') # Provide default
        image_file = request.files.get('chartImage')
        market_country = request.form.get('market_country', 'US')

        if not ticker_symbol or ticker_symbol == 'N/A':
            return jsonify({"error": "Ticker symbol is required"}), 400

        logger.info(f"Received forecast request for ticker: {ticker_symbol} with image: {image_file.filename if image_file else 'No Image'}, Market: {market_country}")

        if image_file and image_file.filename != '':
            try:
                image_bytes = image_file.read()
                analyzed_trend = analyze_chart_image(image_bytes)
                if analyzed_trend in ["up", "down", "neutral"]:
                    image_trend = analyzed_trend
                # image_trend remains None if analysis is inconclusive or returns unexpected
                image_sentiment_for_response = image_trend if image_trend else "Analysis Failed/Neutral"
                logger.info(f"Analyzed image trend result: {analyzed_trend}, Used image_trend: {image_trend}")
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}", exc_info=True)
                image_sentiment_for_response = "Error Processing Image"
                # image_trend remains None
        
        stock_data_df = get_stock_data(ticker_symbol)
        if stock_data_df is None or stock_data_df.empty:
            logger.error(f"Failed to fetch historical data for {ticker_symbol}. Cannot proceed with forecast.")
            return jsonify({"error": f"Could not fetch historical data for {ticker_symbol}. It might be an invalid ticker or have no data."}), 404

        current_price = stock_data_df['y'].iloc[-1]
        last_date = stock_data_df['ds'].iloc[-1]

        # We need to forecast enough periods to find 3 trading days and 1 week trading day
        days_for_daily_forecast_periods = 10 
        days_for_week_ahead_periods = 15   

        model, forecast_output_daily_df, forecast_output_weekly_df = train_and_forecast(
            stock_data_df.copy(),
            days_for_daily_forecast_periods,
            days_for_week_ahead_periods,
            image_trend_regressor=image_trend,
            country_holidays=market_country
        )

        if forecast_output_daily_df is None or forecast_output_weekly_df is None :
            logger.error(f"Model training or prediction failed for {ticker_symbol}.")
            return jsonify({"error": "Failed to generate forecast (model error)."}), 500
            
        min_hist_year_check = last_date.year # Start checking holidays from last known date's year
        max_forecast_date_approx_check = last_date + pd.Timedelta(days=max(days_for_daily_forecast_periods, days_for_week_ahead_periods) + 30)
        max_hist_year_check = max_forecast_date_approx_check.year
        holiday_years_for_check = list(range(min_hist_year_check, max_hist_year_check + 1))
        custom_holidays_dates = get_market_holidays(years=holiday_years_for_check, country=market_country)['ds'].tolist()

        trading_days_found = 0
        future_calendar_dates_in_forecast = forecast_output_daily_df[forecast_output_daily_df['ds'] > last_date]['ds']

        for potential_trading_date in future_calendar_dates_in_forecast:
            if potential_trading_date.dayofweek >= 5: continue
            is_holiday = any(potential_trading_date.normalize() == hol_date.normalize() for hol_date in custom_holidays_dates)
            if is_holiday: continue

            prediction_row = forecast_output_daily_df[forecast_output_daily_df['ds'] == potential_trading_date].head(1)
            if not prediction_row.empty:
                price = round(prediction_row['yhat'].iloc[0], 2)
                lower = round(prediction_row['yhat_lower'].iloc[0], 2)
                upper = round(prediction_row['yhat_upper'].iloc[0], 2)
                recommendation = "Hold/Neutral" # Add full recommendation logic
                image_factor = 0
                if image_trend == "up": image_factor = 0.002
                elif image_trend == "down": image_factor = -0.002
                if isinstance(price, (int, float)) and isinstance(current_price, (int, float)) and current_price > 0:
                    if price > current_price * (1 + 0.01 + image_factor): recommendation = "Consider Buying"
                    elif price < current_price * (1 - 0.01 + image_factor): recommendation = "Consider Selling/Not Buying"

                three_day_forecast_results.append({
                    "date": potential_trading_date.strftime('%Y-%m-%d'), "price": price,
                    "lower_bound": lower, "upper_bound": upper, "recommendation": recommendation
                })
                trading_days_found += 1
                if trading_days_found >= 3: break
        
        target_week_ahead_search_date = last_date + timedelta(days=7)
        for pot_wk_date in forecast_output_weekly_df[forecast_output_weekly_df['ds'] >= target_week_ahead_search_date]['ds']:
            if pot_wk_date.dayofweek >= 5: continue
            is_holiday_wk = any(pot_wk_date.normalize() == hol_date.normalize() for hol_date in custom_holidays_dates)
            if is_holiday_wk: continue
            
            wk_pred_row = forecast_output_weekly_df[forecast_output_weekly_df['ds'] == pot_wk_date].head(1)
            if not wk_pred_row.empty:
                wk_price = round(wk_pred_row['yhat'].iloc[0], 2)
                wk_rec = "Hold/Neutral" # Add full recommendation logic
                image_factor_wk = 0
                if image_trend == "up": image_factor_wk = 0.004 # different factor for week
                elif image_trend == "down": image_factor_wk = -0.004
                if isinstance(wk_price, (int, float)) and isinstance(current_price, (int, float)) and current_price > 0:
                    if wk_price > current_price * (1 + 0.02 + image_factor_wk): wk_rec = "Consider Buying (Week Ahead)"
                    elif wk_price < current_price * (1 - 0.02 + image_factor_wk): wk_rec = "Consider Selling/Not Buying (Week Ahead)"

                week_ahead_pred_details = {
                    "date": pot_wk_date.strftime('%Y-%m-%d'), "price": wk_price,
                    "lower_bound": round(wk_pred_row['yhat_lower'].iloc[0], 2), 
                    "upper_bound": round(wk_pred_row['yhat_upper'].iloc[0], 2),
                    "recommendation": wk_rec
                }
            break 
        
        current_price_display = f"{current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
        last_date_display = last_date.strftime('%Y-%m-%d') if isinstance(last_date, (date, pd.Timestamp, datetime)) else "undefined"
        week_ahead_price_display = week_ahead_pred_details.get('price', "N/A") # Use .get for safety

        logger.info(f"Forecast for {ticker_symbol} (Image Trend: {str(image_trend)}, Market: {market_country}): Current ${current_price_display} (as of {last_date_display}), "
                    f"3-Day Forecast: {[(d['date'], d['price']) for d in three_day_forecast_results]}, "
                    f"Week Ahead Prediction ({week_ahead_pred_details.get('date', 'N/A')}): ${week_ahead_price_display}")

        return jsonify({
            "ticker": ticker_symbol,
            "current_price": current_price_display if current_price_display != "N/A" else None, # Send None if N/A for JS
            "last_known_date": last_date_display,
            "image_analysis_sentiment": image_sentiment_for_response,
            "three_day_forecast": three_day_forecast_results,
            "forecast_week_ahead": week_ahead_pred_details
        })

    except Exception as e:
        # Log crucial variables at the point of error
        current_image_trend_on_error = "variable 'image_trend' not in scope"
        if 'image_trend' in locals(): current_image_trend_on_error = str(image_trend)
        
        current_ticker_on_error = "variable 'ticker_symbol' not in scope"
        if 'ticker_symbol' in locals(): current_ticker_on_error = str(ticker_symbol)

        logger.error(f"Error in /forecast_with_image endpoint for ticker '{current_ticker_on_error}' (image_trend state: {current_image_trend_on_error}): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check logs."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
