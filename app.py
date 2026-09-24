#!/usr/bin/env python3
"""Start the Tai Chi balance analysis dashboard.

    python3 app.py            then open http://127.0.0.1:8051 (a tab opens by itself)

Everything is done from the browser: the Pipeline page computes whatever is
missing, the Event editor curates the windows, and the Results page shows what
came out.  ``python3 app.py --help`` lists the options.
"""

if __name__ == "__main__":
    # Imported here rather than at the top: the orientation filter runs in
    # worker processes that re-import this file, and they need none of it.
    from analysis.dashboard.app import main

    main()
