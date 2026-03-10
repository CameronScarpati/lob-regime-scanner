"""Allow running the dashboard with ``python -m dashboard``.

CLI usage:
    python -m dashboard --symbol BTCUSDT --start 2025-01-01 --end 2025-01-14
    python -m dashboard --demo
"""

import logging

from dashboard.app import create_app, parse_args

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app(args)
    app.run(debug=args.debug, host=args.host, port=args.port)
