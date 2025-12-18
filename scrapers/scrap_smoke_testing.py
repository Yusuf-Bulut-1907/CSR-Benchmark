import requests
import time
from companies_to_scrape import companies_to_scrape

OUTPUT_FILE = "smoke_test_results.txt"

def test_company_urls(companies_dict):
    success = {}
    blocked  = {}
    error    = {}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    for name, url in companies_dict.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200 and len(response.text) > 500:
                success[name] = url

            elif response.status_code in [401, 403]:
                blocked[name] = url

            else:
                error[name] = url

        except Exception:
            error[name] = url

        time.sleep(0.5)  

    return success, blocked, error


def save_results(success, blocked, error):
    with open("smoke_test_results.txt", "w", encoding="utf-8") as f:
        f.write("SMOKE TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("SUCCESS:\n")
        f.write("\n".join(f"{name}: {url}" for name, url in success.items()))
        f.write("\n\nBLOCKED:\n")
        f.write("\n".join(f"{name}: {url}" for name, url in blocked.items()))
        f.write("\n\nERRORS:\n")
        f.write("\n".join(f"{name}: {url}" for name, url in error.items()))

    print("\n📊 TEST COMPLETED")
    print(f"✅ Success  : {len(success)}")
    print(f"🚫 Blocked  : {len(blocked)}")
    print(f"❌ Errors   : {len(error)}")
    print(f"\n📁 RResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    print("🚀 STARTING SMOKE TEST ON COMPANY URLS...\n")
    success, blocked, error = test_company_urls(companies_to_scrape)
    save_results(success, blocked, error)