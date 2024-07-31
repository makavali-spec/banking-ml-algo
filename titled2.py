from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.ticktype import TickType
import threading
import time
import numpy as np
import pandas as pd
from scipy import stats
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading_bot.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Define the pairs
pairs = [
    ('MA', 'V'),
    ('NVO', 'LLY'),
    ('BLK', 'BX'),
    ('EWC', 'EWA'),
    ('GDX', 'GDXJ'),
    ('KO', 'PEP'),
    ('XOM', 'CVX'),
    ('CAT', 'DE'),
    ('T', 'VZ'),
    ('DAL', 'UAL'),
    ('XLK', 'VGT'),
    ('SMH', 'SOXX'),
    ('QQQ', 'VOO'),
    ('ONLN', 'EBIZ'),
    ('WM', 'RSG'),
    ('XLP', 'XLU'),
    ('LQD', 'AGG'),
    ('HYG', 'JNK'),
    ('TLT', 'IEF'),
    ('COIN', 'IBIT')
]

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.prices = {}
        self.current_req_id = 1
        self.symbol_to_reqId = {}
        self.reqId_to_symbol = {}
        self.lock = threading.Lock()
        self.connected = False
        self.connect_attempt = 0
        self.last_connection_time = None

    def error(self, reqId, errorCode, errorString):
        logging.error(f"Error: {reqId} - {errorCode} - {errorString}")
        if errorCode in [502, 504, 1100, 1101, 1102, 2108]:
            self.connected = False
            self.reconnect()

    def connectAck(self):
        self.connected = True
        self.last_connection_time = datetime.now()
        logging.info("Successfully connected to TWS/IB Gateway")

    def connectionClosed(self):
        self.connected = False
        logging.warning("Connection to TWS/IB Gateway closed")
        self.reconnect()

    def reconnect(self):
        if self.connect_attempt < 5:
            self.connect_attempt += 1
            logging.info(f"Attempting to reconnect (attempt {self.connect_attempt})...")
            time.sleep(10)  # Wait before reconnecting
            self.connect("127.0.0.1", 7497, clientId=1)
        else:
            logging.error("Max reconnection attempts reached. Please check your TWS/IB Gateway settings.")

    def historicalData(self, reqId, bar):
        with self.lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append(bar.close)
            logging.debug(f"Received historical data for reqId {reqId}: {bar.close}")

    def historicalDataEnd(self, reqId, start, end):
        logging.info(f"Historical data request finished for reqId {reqId}")
        symbol = self.reqId_to_symbol.get(reqId, "Unknown")
        logging.info(f"Completed historical data for {symbol}: {self.data.get(reqId, [])}")

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType in [1, 2, 4]:  # 1: BID, 2: ASK, 4: LAST
            with self.lock:
                old_price = self.prices.get(reqId)
                self.prices[reqId] = price
                symbol = self.reqId_to_symbol.get(reqId, None)
                if symbol:
                    if old_price is not None:
                        change = price - old_price
                        logging.info(f"Price update for {symbol}: {price} (Change: {change:+.2f})")
                    else:
                        logging.info(f"Initial price for {symbol}: {price}")

def get_contract(symbol):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract

class StatisticalArbitrage:
    def __init__(self, pairs):
        self.pairs = pairs
        self.ib = IBApi()
        self.connect_to_tws()
        self.data_lock = threading.Lock()
        self.pair_data = {pair: {'spread': [], 'hedge_ratio': None, 'z_score': None} for pair in pairs}
        self.open_arbitrages = 0
        self.open_trades = {}
        self.prices = {}
        self.symbol_to_reqId = {}
        self.total_equity = 1000000  # Assume $1M total equity
        self.max_position_size = 0.05  # Maximum 5% of total equity per position
        self.stop_loss_threshold = 2  # Stop loss at 2 standard deviations

    def connect_to_tws(self):
        self.ib.connect("127.0.0.1", 7497, clientId=1)
        ib_thread = threading.Thread(target=self.ib.run, daemon=True)
        ib_thread.start()
        
        timeout = 30
        start_time = time.time()
        while not self.ib.connected and time.time() - start_time < timeout:
            time.sleep(1)
        
        if not self.ib.connected:
            raise ConnectionError("Failed to connect to TWS/IB Gateway")

    def fetch_historical_data(self, ticker):
        contract = get_contract(ticker)
        reqId = self.ib.current_req_id
        self.ib.current_req_id += 1

        logging.info(f"Requesting historical data for {ticker} with reqId {reqId}")

        self.ib.reqHistoricalData(
            reqId,
            contract,
            "",
            "60 D",
            "1 day",
            "MIDPOINT",
            1,
            1,
            False,
            []
        )

        start_time = time.time()
        while time.time() - start_time < 30:  # 30-second timeout
            time.sleep(1)
            if reqId in self.ib.data:
                break

        with self.data_lock:
            prices = self.ib.data.get(reqId, [])
            if not prices:
                logging.warning(f"No historical data received for {ticker}")
            else:
                logging.info(f"Fetched historical data for {ticker}: {prices}")
            return prices

    def calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        prices = np.array(prices)
        if len(prices) < window:
            return None, None, None

        ma = np.mean(prices[-window:])
        std_dev = np.std(prices[-window:])
        upper_band = ma + (std_dev * num_std_dev)
        lower_band = ma - (std_dev * num_std_dev)
        return ma, upper_band, lower_band

    def calculate_hedge_ratio(self, prices1, prices2):
        if len(prices1) != len(prices2):
            min_length = min(len(prices1), len(prices2))
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
        
        df = pd.DataFrame({'y': prices1, 'x': prices2})
        result = stats.linregress(df['x'], df['y'])
        return result.slope

    def calculate_spread(self, prices1, prices2, hedge_ratio):
        return np.array(prices1) - hedge_ratio * np.array(prices2)

    def calculate_z_score(self, spread):
        return (spread[-1] - np.mean(spread)) / np.std(spread)

    def update_pair_data(self):
        for pair in self.pairs:
            stock1, stock2 = pair
            logging.info(f"Fetching historical data for pair: {pair}")
            stock1_prices = self.fetch_historical_data(stock1)
            stock2_prices = self.fetch_historical_data(stock2)

            if len(stock1_prices) < 60 or len(stock2_prices) < 60:
                logging.warning(f"Insufficient data for pair {pair}")
                continue

            hedge_ratio = self.calculate_hedge_ratio(stock1_prices, stock2_prices)
            spread = self.calculate_spread(stock1_prices, stock2_prices, hedge_ratio)
            z_score = self.calculate_z_score(spread)
            
            self.pair_data[pair]['hedge_ratio'] = hedge_ratio
            self.pair_data[pair]['spread'] = spread.tolist()
            self.pair_data[pair]['z_score'] = z_score

            logging.info(f"Updated pair data for {pair}: hedge_ratio={hedge_ratio}, z_score={z_score}")

    def monitor_and_trade_pairs(self):
        for pair in self.pairs:
            stock1, stock2 = pair
            logging.info(f"Monitoring pair: {pair}")
            
            z_score = self.pair_data[pair]['z_score']
            if z_score is None:
                logging.warning(f"Z-score not available for pair {pair}")
                continue

            logging.info(f"Current z-score for pair {pair}: {z_score}")

            if z_score > 2 and pair not in self.open_trades:
                self.execute_trade(pair, "short_stock1_long_stock2")
            elif z_score < -2 and pair not in self.open_trades:
                self.execute_trade(pair, "long_stock1_short_stock2")
            elif abs(z_score) < 0.5 and pair in self.open_trades:
                self.close_trade(pair)

            self.check_stop_loss(pair)

    def execute_trade(self, pair, signal):
        if self.open_arbitrages >= 5:
            logging.warning("Maximum number of arbitrage opportunities reached. No new trades will be executed.")
            return

        stock1, stock2 = pair
        hedge_ratio = self.pair_data[pair]['hedge_ratio']

        # Calculate position sizes
        stock1_price = self.prices[self.symbol_to_reqId[stock1]]
        stock2_price = self.prices[self.symbol_to_reqId[stock2]]
        
        position_value = min(self.total_equity * self.max_position_size, self.total_equity * 0.5 / (self.open_arbitrages + 1))
        stock1_quantity = int(position_value / (2 * stock1_price))
        stock2_quantity = int(hedge_ratio * stock1_quantity)

        # Place orders
        if signal == "long_stock1_short_stock2":
            self.place_order(stock1, "BUY", stock1_quantity)
            self.place_order(stock2, "SELL", stock2_quantity)
        elif signal == "short_stock1_long_stock2":
            self.place_order(stock1, "SELL", stock1_quantity)
            self.place_order(stock2, "BUY", stock2_quantity)

        self.open_trades[pair] = {
            'signal': signal,
            'stock1_quantity': stock1_quantity,
            'stock2_quantity': stock2_quantity,
            'entry_z_score': self.pair_data[pair]['z_score']
        }
        self.open_arbitrages += 1

        logging.info(f"Executed {signal} trade for pair {pair}")

    def close_trade(self, pair):
        if pair not in self.open_trades:
            logging.warning(f"No open trade found for pair {pair}")
            return

        stock1, stock2 = pair
        trade_info = self.open_trades[pair]

        if trade_info['signal'] == "long_stock1_short_stock2":
            self.place_order(stock1, "SELL", trade_info['stock1_quantity'])
            self.place_order(stock2, "BUY", trade_info['stock2_quantity'])
        elif trade_info['signal'] == "short_stock1_long_stock2":
            self.place_order(stock1, "BUY", trade_info['stock1_quantity'])
            self.place_order(stock2, "SELL", trade_info['stock2_quantity'])

        del self.open_trades[pair]
        self.open_arbitrages -= 1

        logging.info(f"Closed trade for pair {pair}")

    def check_stop_loss(self, pair):
        if pair not in self.open_trades:
            return

        current_z_score = self.pair_data[pair]['z_score']
        entry_z_score = self.open_trades[pair]['entry_z_score']

        if abs(current_z_score - entry_z_score) > self.stop_loss_threshold:
            logging.warning(f"Stop loss triggered for pair {pair}")
            self.close_trade(pair)

    def place_order(self, symbol, action, quantity):
        contract = get_contract(symbol)
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "MKT"
        
        self.ib.placeOrder(self.ib.current_req_id, contract, order)
        self.ib.current_req_id += 1

        logging.info(f"Placed order: {action} {quantity} shares of {symbol}")

    def init_price_streaming(self):
        for pair in self.pairs:
            for stock in pair:
                contract = get_contract(stock)
                reqId = self.ib.current_req_id
                self.ib.current_req_id += 1

                self.ib.symbol_to_reqId[stock] = reqId
                self.ib.reqId_to_symbol[reqId] = stock

                self.ib.reqMktData(reqId, contract, "", False, False, [])
                logging.info(f"Requested market data for {stock} with reqId {reqId}")

    def check_connection(self):
        if not self.ib.connected:
            logging.warning("Connection lost. Attempting to reconnect...")
            self.ib.reconnect()
        elif self.ib.last_connection_time and (datetime.now() - self.ib.last_connection_time) > timedelta(hours=1):
            logging.info("Refreshing connection...")
            self.ib.disconnect()
            time.sleep(1)
            self.ib.connect("127.0.0.1", 7497, clientId=1)

    def run(self):
        self.update_pair_data()
        self.init_price_streaming()
        while True:
            self.check_connection()
            self.update_pair_data()
            self.monitor_and_trade_pairs()
            time.sleep(60)  # Run the monitoring every 60 seconds

    def backtest(self, start_date, end_date):
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest results
        backtest_results = {
            'trades': [],
            'pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }
        
        # Fetch historical data for the backtest period
        for pair in self.pairs:
            stock1, stock2 = pair
            stock1_prices = self.fetch_historical_data_for_backtest(stock1, start_date, end_date)
            stock2_prices = self.fetch_historical_data_for_backtest(stock2, start_date, end_date)
            
            if len(stock1_prices) != len(stock2_prices):
                logging.warning(f"Mismatched data lengths for pair {pair}. Skipping.")
                continue
            
            # Calculate hedge ratio and spread
            hedge_ratio = self.calculate_hedge_ratio(stock1_prices, stock2_prices)
            spread = self.calculate_spread(stock1_prices, stock2_prices, hedge_ratio)
            
            # Initialize variables for this pair
            position = 0
            entry_price = 0
            entry_date = None
            daily_returns = []
            
            # Simulate trading
            for i in range(20, len(spread)):  # Start after the initial window
                z_score = (spread[i] - np.mean(spread[i-20:i])) / np.std(spread[i-20:i])
                
                if position == 0:
                    if z_score > 2:
                        position = -1
                        entry_price = spread[i]
                        entry_date = start_date + timedelta(days=i)
                    elif z_score < -2:
                        position = 1
                        entry_price = spread[i]
                        entry_date = start_date + timedelta(days=i)
                elif position == 1 and z_score >= 0:
                    # Close long position
                    pnl = spread[i] - entry_price
                    backtest_results['pnl'] += pnl
                    backtest_results['trades'].append({
                        'pair': pair,
                        'entry_date': entry_date,
                        'exit_date': start_date + timedelta(days=i),
                        'pnl': pnl
                    })
                    daily_returns.append(pnl)
                    position = 0
                elif position == -1 and z_score <= 0:
                    # Close short position
                    pnl = entry_price - spread[i]
                    backtest_results['pnl'] += pnl
                    backtest_results['trades'].append({
                        'pair': pair,
                        'entry_date': entry_date,
                        'exit_date': start_date + timedelta(days=i),
                        'pnl': pnl
                    })
                    daily_returns.append(pnl)
                    position = 0
                
                # Check for stop loss
                if position != 0 and abs(spread[i] - entry_price) > self.stop_loss_threshold * np.std(spread[i-20:i]):
                    pnl = (spread[i] - entry_price) * position
                    backtest_results['pnl'] += pnl
                    backtest_results['trades'].append({
                        'pair': pair,
                        'entry_date': entry_date,
                        'exit_date': start_date + timedelta(days=i),
                        'pnl': pnl,
                        'stop_loss': True
                    })
                    daily_returns.append(pnl)
                    position = 0
            
            # Calculate performance metrics
            if daily_returns:
                backtest_results['sharpe_ratio'] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                cumulative_returns = np.cumsum(daily_returns)
                backtest_results['max_drawdown'] = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
        
        logging.info(f"Backtest completed. Total PNL: {backtest_results['pnl']}")
        logging.info(f"Sharpe Ratio: {backtest_results['sharpe_ratio']}")
        logging.info(f"Max Drawdown: {backtest_results['max_drawdown']}")
        logging.info(f"Total trades: {len(backtest_results['trades'])}")
        
        return backtest_results


if __name__ == "__main__":
    bot = StatisticalArbitrage(pairs)

    # Allow time for connection and API setup
    time.sleep(5)

    try:
        # Update pair data and start monitoring trades
        bot.update_pair_data()
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Ensure all open positions are closed before exiting
        for pair in bot.open_trades:
            bot.close_trade(pair)
        logging.info("All positions closed. Bot shutting down.")
        bot.ib.disconnect()