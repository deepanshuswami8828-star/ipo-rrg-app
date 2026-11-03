"""
Daily data refresh of the IPO RRG app (no Streamlit). Adds every completed NSE session from NSE's official files
for all IPO stocks + NIFTY 50; prints a report of anything unusual. One stock's problem never fails the run.
Exit code: 0 = SUCCESS / NO_NEW_SESSION / SKIPPED, 1 = FAILED (only when NIFTY 50 itself could not be updated or
something unexpected broke the run).
"""
import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import data_pipeline as dp


def main():
    parser = argparse.ArgumentParser(description="Refresh the IPO RRG app's prices from NSE's official files.")
    parser.add_argument("--force", action="store_true", help="Run even if the last refresh was under 15 minutes ago.")
    args = parser.parse_args()

    print("=== IPO RRG data refresh (NSE official files) ===")
    res = dp.refresh_data(force=args.force)
    print(f"Status:  {res.get('status')}")
    print(f"Message: {res.get('message')}")
    rep = res.get("report") or {}
    for key, label in (("fallback", "Yahoo fallback used (replaced by NSE later)"), ("held", "HELD for review"),
                       ("rebased", "NSE re-based (split/bonus)"), ("errors", "errors (stored data kept)"),
                       ("promoted", "joined the RRG pages (60 sessions)"), ("missing_files", "NSE files not out yet")):
        if rep.get(key):
            print(f"{label}: {rep[key]}")
    sys.exit(0 if res.get("status") in ("SUCCESS", "NO_NEW_SESSION", "SKIPPED") else 1)


if __name__ == "__main__":
    main()
