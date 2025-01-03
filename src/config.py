import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DIR_NAME = os.path.dirname(ROOT_FOLDER)
TRADING_DATA_DIR = os.path.join(DIR_NAME, "trading_data")

LOOKER_URL = os.environ.get('LOOKER_URL', '')

BINANCE_API_KEY_TEST = os.environ.get('BINANCE_API_KEY_TEST')
BINANCE_API_SECRET_TEST = os.environ.get('BINANCE_API_SECRET_TEST')
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')

KUCOIN_API_KEY = os.environ.get('KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.environ.get('KUCOIN_API_SECRET')
KUCOIN_PASS = os.environ.get('KUCOIN_PASS')

MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_API_SECRET = os.environ.get('MEXC_API_SECRET')

BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET')
BYBIT_API_KEY_SUB1 = os.environ.get('BYBIT_API_KEY_SUB1')
BYBIT_API_SECRET_SUB1 = os.environ.get('BYBIT_API_SECRET_SUB1')

LEVERAGE = os.environ.get('LEVERAGE', 1)

# exchanges
BINANCE_CONFIG_TEST = {
    'apiKey': BINANCE_API_KEY_TEST,
    'secret': BINANCE_API_SECRET_TEST,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'leverage': LEVERAGE
    },
    'base_currency': 'USDC',
}

BINANCE_CONFIG_PROD = {
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'leverage': LEVERAGE
    },
    'base_currency': 'USDC',
}

KUCOIN_CONFIG_PROD = {
    'apiKey': KUCOIN_API_KEY,
    'secret': KUCOIN_API_SECRET,
    'password': KUCOIN_PASS,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'leverage': LEVERAGE
    },
    'base_currency': 'USDT',
}

MEXC_CONFIG_PROD = {
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {
        'leverage': LEVERAGE,
        'defaultType': 'swap'
    },
    'base_currency': 'USDT',
    'timeout': 30_000
}

BYBIT_CONFIG_PROD = {
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_API_SECRET,
    'enableRateLimit': True,
    'options': {
        'leverage': LEVERAGE,
        'defaultType': 'swap'
    },
    'base_currency': 'USDT',
    'sub_accounts': {
        'subAccount1': {
            'apiKey': BYBIT_API_KEY_SUB1,
            'secret': BYBIT_API_SECRET_SUB1,
            'leverage': 2
        }
    }
}

SLACK_URL = os.environ.get("SLACK_URL")
LLM_TRADER_SLACK_URL = os.environ.get("LLM_TRADER_SLACK_URL")

# turtle strategy
# risks
TRADE_RISK_ALLOCATION = float(os.environ.get('TRADE_RISK_ALLOCATION', 0.01))  # one trade risk capital allocation
MAX_ONE_ASSET_RISK_ALLOCATION = float(os.environ.get(
    'MAX_ONE_ASSET_RISK_ALLOCATION',
    0.5
))  # maximum of capital in one asset traded

# timeframes
ATR_PERIOD = int(os.environ.get('ATR_PERIOD', 20))  # 20 for slow, 50 for fast
TURTLE_ENTRY_DAYS = int(os.environ.get('TURTLE_ENTRY_DAYS', ATR_PERIOD))  # 20 for fast, 50 for slow
TURTLE_EXIT_DAYS = int(os.environ.get('TURTLE_EXIT_DAYS', 10))  # 10 for fast, 20 for slow
MIN_POSITION_THRESHOLD = int(os.environ.get('MIN_POSITION_THRESHOLD', 20))
VALIDATOR_REPEATED_CALL_TIME_TEST_MIN = int(os.environ.get('VALIDATOR_REPEATED_CALL_TIME_TEST_MIN', 60))
ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD = int(os.environ.get('ACCEPTABLE_ROUNDING_PERCENT_THRESHOLD', 5))

APP_SETTINGS = os.environ.get("APP_SETTINGS", "DevConfig")


class Config:
    DEBUG = False
    DEVELOPMENT = True
    USE_SANDBOX = True
    EXCHANGES = {
        'binance': BINANCE_CONFIG_TEST
    }


class DevConfig(Config):
    pass


class ProdConfig(Config):
    DEBUG = False
    DEVELOPMENT = False
    USE_SANDBOX = False
    EXCHANGES = {
        'binance': BINANCE_CONFIG_PROD,
        'kucoinfutures': KUCOIN_CONFIG_PROD,
        'mexc': MEXC_CONFIG_PROD,
        'bybit': BYBIT_CONFIG_PROD
    }


app_config = eval(APP_SETTINGS)
