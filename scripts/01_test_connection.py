"""IBKR Gateway connection test — strictly read-only.

Connects to IBKR Gateway on port 4001 (live), queries contract details + head
timestamp for AAPL, then disconnects. No order code paths. Run from project root:

    .venv/bin/python scripts/01_test_connection.py
"""

import sys
import time
from ib_async import IB, Stock, util

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001  # 4001 = live gateway, 4002 = paper, 7496/7497 = TWS
CLIENT_ID = 11  # arbitrary; stays distinct from any TWS/Gateway default

util.logToConsole(level="WARNING")


def main() -> int:
    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}")
    t0 = time.perf_counter()
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=10)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"connected in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"server version: {ib.client.serverVersion()}")
    print(f"accounts: {ib.managedAccounts()}")

    aapl = Stock("AAPL", "SMART", "USD", primaryExchange="NASDAQ")
    details = ib.reqContractDetails(aapl)
    if not details:
        print("reqContractDetails returned empty for AAPL — connection works but data may be unsubscribed")
        ib.disconnect()
        return 2
    cd = details[0]
    print(
        f"AAPL contract: conId={cd.contract.conId} "
        f"longName={cd.longName!r} timeZoneId={cd.timeZoneId}"
    )

    head = ib.reqHeadTimeStamp(aapl, whatToShow="TRADES", useRTH=False, formatDate=1)
    print(f"AAPL earliest TRADES tick available: {head}")

    ib.disconnect()
    print("disconnected cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
