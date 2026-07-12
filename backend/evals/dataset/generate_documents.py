"""Deterministically regenerate the eval document set.

The PDFs are committed so evals are reproducible without running this, but
the script is the source of truth for their content. Each page carries
distinct facts so retrieval labels (file + page) are unambiguous.

Usage:
    python evals/dataset/generate_documents.py
"""

from __future__ import annotations

from pathlib import Path

import fitz

DOCS_DIR = Path(__file__).resolve().parent / "documents"

DOCUMENTS: dict[str, list[tuple[str, str]]] = {
    "northwind_handbook.pdf": [
        (
            "Company Overview",
            "Northwind Robotics was founded in 2017 in Tallinn, Estonia. The "
            "company builds autonomous warehouse robots and employs 240 people "
            "across three offices. Its flagship product line is the Atlas "
            "series of picking robots.\n\n"
            "The company operates on a four-day work week and is "
            "employee-owned.",
        ),
        (
            "Vacation Policy",
            "All full-time employees receive 27 days of paid vacation per "
            "year, in addition to public holidays. Unused vacation days can "
            "be carried over into the first quarter of the following year, up "
            "to a maximum of 5 days.\n\n"
            "Vacation requests longer than two consecutive weeks require "
            "approval from a department head.",
        ),
        (
            "Security Practices",
            "All employees must use hardware security keys for authentication "
            "to production systems. Passwords alone are never sufficient for "
            "access to customer data.\n\n"
            "Security incidents must be reported to the on-call security "
            "engineer within 30 minutes of discovery.",
        ),
        (
            "Remote Work",
            "Employees working remotely receive a one-time home office "
            "stipend of $1,200 to purchase equipment. The stipend renews "
            "every three years.\n\n"
            "Remote employees must be reachable during core hours, defined as "
            "10:00 to 15:00 in the Tallinn time zone.",
        ),
        (
            "On-Call Rotation",
            "The engineering on-call rotation lasts one week and includes a "
            "compensation bonus of 8% of weekly salary. Engineers may swap "
            "rotations with at least 48 hours notice.\n\n"
            "Escalations that remain unresolved for more than one hour are "
            "paged to the engineering director.",
        ),
    ],
    "atlas_spec.pdf": [
        (
            "Atlas P-100 Overview",
            "The Atlas P-100 is Northwind's flagship autonomous picking "
            "robot, launched in 2021. It carries payloads of up to 15 kg and "
            "travels at a top speed of 2.5 m/s in mixed pedestrian "
            "environments.\n\n"
            "Over 3,000 units are deployed across Europe and North America.",
        ),
        (
            "Battery and Charging",
            "The P-100 uses a lithium iron phosphate (LiFePO4) battery pack "
            "rated for 8 hours of continuous operation. A fast-charge dock "
            "restores 80% capacity in 45 minutes.\n\n"
            "Battery packs are hot-swappable; a trained operator can exchange "
            "a pack in under 90 seconds.",
        ),
        (
            "Sensing and Navigation",
            "Navigation combines six stereo camera pairs with two solid-state "
            "lidar units with a range of 100 meters. The chassis is rated "
            "IP54 for dust and splash resistance.\n\n"
            "Localization accuracy is within 2 centimeters using pre-mapped "
            "warehouse floor plans.",
        ),
        (
            "Software Platform",
            "The P-100 runs on a ROS 2 based software stack. Fleet "
            "coordination is handled by Northwind's Hive fleet manager, which "
            "supports up to 500 robots per site.\n\n"
            "Over-the-air software updates are released on a monthly cadence.",
        ),
        (
            "Safety Systems",
            "The P-100 complies with the ISO 3691-4 standard for driverless "
            "industrial trucks. The emergency stop system halts the robot "
            "within 250 milliseconds.\n\n"
            "A protective safety zone of 1.2 meters is enforced around "
            "detected humans; entering it triggers an immediate slowdown.",
        ),
    ],
    "q3_financial_report.pdf": [
        (
            "Q3 2025 Executive Summary",
            "Northwind Robotics reports third-quarter 2025 revenue of EUR "
            "18.4 million, an increase of 22% year over year. Gross margin "
            "improved to 41%.\n\n"
            "The quarter closed with a cash position of EUR 31 million and no "
            "outstanding debt.",
        ),
        (
            "Segment Performance",
            "Robotics hardware contributed EUR 12.1 million in revenue, "
            "software subscriptions EUR 4.2 million, and professional "
            "services EUR 2.1 million.\n\n"
            "Software subscription revenue grew 38% year over year, the "
            "fastest of the three segments.",
        ),
        (
            "Operating Costs",
            "Research and development spending reached EUR 3.8 million, "
            "representing 21% of quarterly revenue. Total headcount grew from "
            "240 to 265 during the quarter.\n\n"
            "Sales and marketing costs were flat at EUR 2.4 million.",
        ),
        (
            "Q4 Outlook",
            "Management guides fourth-quarter revenue of EUR 20 to 21 "
            "million. A new engineering office in Munich is planned to open "
            "in the first quarter of 2026.\n\n"
            "The Hive fleet manager version 4.0 launch is scheduled for "
            "December 2025.",
        ),
        (
            "Risk Factors",
            "Lead times for solid-state lidar units remain elevated at 26 "
            "weeks, posing a supply chain risk for hardware deliveries.\n\n"
            "Approximately 35% of revenue is denominated in US dollars, "
            "creating currency exposure against the euro.",
        ),
    ],
    "rotterdam_pilot_study.pdf": [
        (
            "Pilot Setup",
            "A 12-week automation pilot was conducted at the Rotterdam "
            "distribution center of a major grocery retailer, deploying 40 "
            "Atlas P-100 robots across two warehouse zones.\n\n"
            "The pilot ran from March through May 2025.",
        ),
        (
            "Methodology",
            "The study used an A/B design: Zone A operated with robot "
            "assistance while Zone B continued manual picking as a control. "
            "The primary metric was picks per hour per worker.\n\n"
            "Error rates were measured as mis-picks per 1,000 order lines.",
        ),
        (
            "Results",
            "Zone A throughput increased by 34% compared to the control "
            "zone. The mis-pick error rate fell from 1.8% to 0.4% over the "
            "pilot period.\n\n"
            "Average order cycle time dropped from 42 minutes to 28 minutes.",
        ),
        (
            "Incidents and Reliability",
            "Three minor stoppages occurred during the pilot, all resolved "
            "within 20 minutes. No safety incidents involving personnel were "
            "recorded.\n\n"
            "Mean time between failures (MTBF) across the fleet was 410 "
            "hours.",
        ),
        (
            "Conclusions",
            "The projected return on investment period is 14 months. The "
            "study recommends rolling out the system to three additional "
            "distribution sites in 2026.\n\n"
            "Worker satisfaction surveys showed a 12-point improvement after "
            "robot deployment.",
        ),
    ],
}


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, pages in DOCUMENTS.items():
        doc = fitz.open()
        for title, body in pages:
            page = doc.new_page()
            page.insert_text((72, 90), title, fontsize=18)
            page.insert_textbox(fitz.Rect(72, 130, 540, 750), body, fontsize=11)
        out = DOCS_DIR / filename
        doc.save(str(out))
        doc.close()
        print(f"Wrote {out.name}: {len(pages)} pages")


if __name__ == "__main__":
    main()
