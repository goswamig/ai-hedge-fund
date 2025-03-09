from flask import Flask, request, jsonify
from dotenv import load_dotenv
from main import run_hedge_fund
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load environment variables (e.g., API keys used in main.py)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

@app.route('/run-hedge-fund', methods=['GET'])
def run_hedge_fund_endpoint():
    # Get the 'tickers' query parameter
    tickers_str = request.args.get('tickers')
    if not tickers_str:
        return jsonify({"error": "Tickers are required"}), 400

    # Convert comma-separated tickers string to a list
    tickers = [ticker.strip() for ticker in tickers_str.split(",")]

    # Optional parameters with defaults
    end_date = request.args.get('end_date', datetime.now().strftime("%Y-%m-%d"))
    start_date = request.args.get('start_date')
    show_reasoning = request.args.get('show_reasoning', '0') == '1'

    # If start_date is not provided, default to 3 months before end_date
    if not start_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")

    # Initialize portfolio with default values (as in the original code)
    initial_cash = 100000.0
    margin_requirement = 0.0
    portfolio = {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for ticker in tickers},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
    }

    try:
        # Call the run_hedge_fund function from main.py
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=None,  # Use all analysts by default
            model_name="gpt-4o-mini",  # Default model from main.py
            model_provider="OpenAI"
        )

        # Check if decisions were parsed successfully
        if result["decisions"] is None:
            return jsonify({"error": "Failed to parse hedge fund response"}), 500

        # Return the result as JSON
        return jsonify({
            "decisions": result["decisions"],
            "analyst_signals": result["analyst_signals"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
