"""Capture GUI screenshots for the README (needs: pip install playwright && playwright install chromium).

    python -m app.server &      # in another terminal
    python scripts/screenshots.py
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(URL)
        await page.wait_for_selector("#patchRows tr")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=OUT / "gui_overview.png")

        # forest map layer + latest interval
        await page.check("#lyForest"); await page.uncheck("#lyLoss"); await page.uncheck("#lyPoly")
        await page.wait_for_timeout(800)
        await page.screenshot(path=OUT / "gui_forest_map.png")
        await page.uncheck("#lyForest"); await page.check("#lyLoss"); await page.check("#lyPoly")

        # swipe comparison
        await page.check("#lySwipe")
        await page.wait_for_timeout(800)
        await page.screenshot(path=OUT / "gui_swipe_compare.png")
        await page.uncheck("#lySwipe")

        # click the largest clearing → zoom + popup
        await page.click("#patchRows tr[data-id]")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=OUT / "gui_patch_zoom.png")

        # new-analysis tab with preset
        await page.click(".tab[data-tab=new]")
        await page.click(".chip[data-preset=rondonia]")
        await page.wait_for_timeout(800)
        await page.screenshot(path=OUT / "gui_new_analysis.png")

        await page.click(".tab[data-tab=model]")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=OUT / "gui_model_tab.png")
        await b.close()
        print("JS errors:", errors or "none")


asyncio.run(main())
