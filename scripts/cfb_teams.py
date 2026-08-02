"""
College football team crosswalk. The first CFB deliverable, and deliberately so.

Why this comes first
--------------------
The NFL side lost real time to a single naming mismatch -- nfl_data_py calls the
Rams "LA" while everything else said "LAR", and every Rams game silently
vanished until someone noticed. That was 32 teams. College football is 275
entities in the odds feed alone, and the naming is genuinely hostile:

  * "Miami Hurricanes" and "Miami (OH) RedHawks" are different schools
  * "Ole Miss", not Mississippi. "Mississippi State" is someone else
  * "Louisiana Ragin Cajuns" (Lafayette) vs "UL Monroe Warhawks"
  * "Youngstown St Penguins" -- St, not State, and no period
  * "Yale University Bulldogs", "Southern University Jaguars",
    "Presbyterian College Blue Hose" -- University/College inconsistently present
  * schools that renamed mid-sample and appear under BOTH names:
    Dixie State -> Utah Tech, Houston Baptist -> Houston Christian,
    Sam Houston State -> Sam Houston, Texas A&M-Commerce -> East Texas A&M

Mascot-stripping by algorithm would look clever and fail silently on most of
these. So the FBS mapping is explicit, and anything not in it is treated as
non-FBS rather than guessed at.

Non-FBS matters: roughly a hundred games a year are played against FCS
opponents who have no usable stats. Those games must be identifiable so they
can be excluded, not quietly zero-filled into the feature matrix.

Usage:
    from cfb_teams import to_key, is_fbs, assert_all_mapped
    python cfb_teams.py          # validate against the live odds feed
"""

import json
import sys
from pathlib import Path

# Canonical key -> the exact full_name(s) The Odds API uses. Aliases exist for
# schools that renamed inside our 2020-2025 sample window; both spellings must
# resolve to the same key or the same program would split into two teams.
FBS = {
    "AIR_FORCE": ["Air Force Falcons"], "AKRON": ["Akron Zips"],
    "ALABAMA": ["Alabama Crimson Tide"], "APP_STATE": ["Appalachian State Mountaineers"],
    "ARIZONA": ["Arizona Wildcats"], "ARIZONA_ST": ["Arizona State Sun Devils"],
    "ARKANSAS": ["Arkansas Razorbacks"], "ARKANSAS_ST": ["Arkansas State Red Wolves"],
    "ARMY": ["Army Black Knights"], "AUBURN": ["Auburn Tigers"],
    "BALL_ST": ["Ball State Cardinals"], "BAYLOR": ["Baylor Bears"],
    "BOISE_ST": ["Boise State Broncos"], "BOSTON_COLLEGE": ["Boston College Eagles"],
    "BOWLING_GREEN": ["Bowling Green Falcons"], "BUFFALO": ["Buffalo Bulls"],
    "BYU": ["BYU Cougars"], "CALIFORNIA": ["California Golden Bears"],
    "CENTRAL_MICHIGAN": ["Central Michigan Chippewas"], "CHARLOTTE": ["Charlotte 49ers"],
    "CINCINNATI": ["Cincinnati Bearcats"], "CLEMSON": ["Clemson Tigers"],
    "COASTAL_CAROLINA": ["Coastal Carolina Chanticleers"], "COLORADO": ["Colorado Buffaloes"],
    "COLORADO_ST": ["Colorado State Rams"], "UCONN": ["UConn Huskies"],
    "DELAWARE": ["Delaware Blue Hens"], "DUKE": ["Duke Blue Devils"],
    "EAST_CAROLINA": ["East Carolina Pirates"], "EASTERN_MICHIGAN": ["Eastern Michigan Eagles"],
    "FLORIDA": ["Florida Gators"], "FAU": ["Florida Atlantic Owls"],
    "FIU": ["Florida International Panthers"], "FLORIDA_ST": ["Florida State Seminoles"],
    "FRESNO_ST": ["Fresno State Bulldogs"], "GEORGIA": ["Georgia Bulldogs"],
    "GEORGIA_SOUTHERN": ["Georgia Southern Eagles"], "GEORGIA_ST": ["Georgia State Panthers"],
    "GEORGIA_TECH": ["Georgia Tech Yellow Jackets"], "HAWAII": ["Hawaii Rainbow Warriors"],
    "HOUSTON": ["Houston Cougars"], "ILLINOIS": ["Illinois Fighting Illini"],
    "INDIANA": ["Indiana Hoosiers"], "IOWA": ["Iowa Hawkeyes"],
    "IOWA_ST": ["Iowa State Cyclones"], "JACKSONVILLE_ST": ["Jacksonville State Gamecocks"],
    "JAMES_MADISON": ["James Madison Dukes"], "KANSAS": ["Kansas Jayhawks"],
    "KANSAS_ST": ["Kansas State Wildcats"], "KENNESAW_ST": ["Kennesaw State Owls"],
    "KENT_ST": ["Kent State Golden Flashes"], "KENTUCKY": ["Kentucky Wildcats"],
    "LIBERTY": ["Liberty Flames"], "LOUISIANA": ["Louisiana Ragin Cajuns"],
    "LOUISIANA_TECH": ["Louisiana Tech Bulldogs"], "LOUISVILLE": ["Louisville Cardinals"],
    "LSU": ["LSU Tigers"], "MARSHALL": ["Marshall Thundering Herd"],
    "MARYLAND": ["Maryland Terrapins"], "MEMPHIS": ["Memphis Tigers"],
    "MIAMI_FL": ["Miami Hurricanes"], "MIAMI_OH": ["Miami (OH) RedHawks"],
    "MICHIGAN": ["Michigan Wolverines"], "MICHIGAN_ST": ["Michigan State Spartans"],
    "MIDDLE_TENNESSEE": ["Middle Tennessee Blue Raiders"], "MINNESOTA": ["Minnesota Golden Gophers"],
    "MISSISSIPPI_ST": ["Mississippi State Bulldogs"], "MISSOURI": ["Missouri Tigers"],
    "MISSOURI_ST": ["Missouri State Bears"], "NAVY": ["Navy Midshipmen"],
    "NC_STATE": ["NC State Wolfpack"], "NEBRASKA": ["Nebraska Cornhuskers"],
    "NEVADA": ["Nevada Wolf Pack"], "NEW_MEXICO": ["New Mexico Lobos"],
    "NEW_MEXICO_ST": ["New Mexico State Aggies"], "NORTH_CAROLINA": ["North Carolina Tar Heels"],
    "NORTH_TEXAS": ["North Texas Mean Green"], "NORTHERN_ILLINOIS": ["Northern Illinois Huskies"],
    "NORTHWESTERN": ["Northwestern Wildcats"], "NOTRE_DAME": ["Notre Dame Fighting Irish"],
    "OHIO": ["Ohio Bobcats"], "OHIO_ST": ["Ohio State Buckeyes"],
    "OKLAHOMA": ["Oklahoma Sooners"], "OKLAHOMA_ST": ["Oklahoma State Cowboys"],
    "OLD_DOMINION": ["Old Dominion Monarchs"], "OLE_MISS": ["Ole Miss Rebels"],
    "OREGON": ["Oregon Ducks"], "OREGON_ST": ["Oregon State Beavers"],
    "PENN_ST": ["Penn State Nittany Lions"], "PITTSBURGH": ["Pittsburgh Panthers"],
    "PURDUE": ["Purdue Boilermakers"], "RICE": ["Rice Owls"],
    "RUTGERS": ["Rutgers Scarlet Knights"],
    "SAM_HOUSTON": ["Sam Houston State Bearkats", "Sam Houston Bearkats"],
    "SAN_DIEGO_ST": ["San Diego State Aztecs"], "SAN_JOSE_ST": ["San Jose State Spartans"],
    "SMU": ["SMU Mustangs"], "SOUTH_ALABAMA": ["South Alabama Jaguars"],
    "SOUTH_CAROLINA": ["South Carolina Gamecocks"], "SOUTH_FLORIDA": ["South Florida Bulls"],
    "SOUTHERN_MISS": ["Southern Mississippi Golden Eagles"], "STANFORD": ["Stanford Cardinal"],
    "SYRACUSE": ["Syracuse Orange"], "TCU": ["TCU Horned Frogs"],
    "TEMPLE": ["Temple Owls"], "TENNESSEE": ["Tennessee Volunteers"],
    "TEXAS": ["Texas Longhorns"], "TEXAS_AM": ["Texas A&M Aggies"],
    "TEXAS_ST": ["Texas State Bobcats"], "TEXAS_TECH": ["Texas Tech Red Raiders"],
    "TOLEDO": ["Toledo Rockets"], "TROY": ["Troy Trojans"],
    "TULANE": ["Tulane Green Wave"], "TULSA": ["Tulsa Golden Hurricane"],
    "UAB": ["UAB Blazers"], "UCF": ["UCF Knights"], "UCLA": ["UCLA Bruins"],
    "UL_MONROE": ["UL Monroe Warhawks"], "UMASS": ["UMass Minutemen"],
    "UNLV": ["UNLV Rebels"], "USC": ["USC Trojans"], "UTAH": ["Utah Utes"],
    "UTAH_ST": ["Utah State Aggies"], "UTEP": ["UTEP Miners"], "UTSA": ["UTSA Roadrunners"],
    "VANDERBILT": ["Vanderbilt Commodores"], "VIRGINIA": ["Virginia Cavaliers"],
    "VIRGINIA_TECH": ["Virginia Tech Hokies"], "WAKE_FOREST": ["Wake Forest Demon Deacons"],
    "WASHINGTON": ["Washington Huskies"], "WASHINGTON_ST": ["Washington State Cougars"],
    "WEST_VIRGINIA": ["West Virginia Mountaineers"], "WESTERN_KENTUCKY": ["Western Kentucky Hilltoppers"],
    "WESTERN_MICHIGAN": ["Western Michigan Broncos"], "WISCONSIN": ["Wisconsin Badgers"],
    "WYOMING": ["Wyoming Cowboys"],
}

# Reverse index, built once. Lowercased so casing drift cannot break a join.
_ODDS_TO_KEY = {alias.lower(): key for key, aliases in FBS.items() for alias in aliases}


def to_key(odds_name: str) -> str | None:
    """Odds API full_name -> canonical key, or None if not an FBS program."""
    if not odds_name:
        return None
    return _ODDS_TO_KEY.get(odds_name.strip().lower())


def is_fbs(odds_name: str) -> bool:
    return to_key(odds_name) is not None


def assert_all_mapped(names, context: str = "") -> None:
    """Raise on any name that should be FBS but is not in the crosswalk.

    Call this wherever odds data enters the pipeline. An unmapped FBS school
    that is silently skipped is the Rams bug again, and it is invisible until
    someone counts games.
    """
    unmapped = sorted({n for n in names if n and to_key(n) is None})
    if unmapped:
        raise SystemExit(
            f"UNMAPPED TEAMS{' in ' + context if context else ''} "
            f"({len(unmapped)}):\n  " + "\n  ".join(unmapped) +
            "\n\nAdd to FBS in cfb_teams.py, or confirm each is non-FBS and "
            "filter it out upstream with is_fbs()."
        )


def main():
    """Validate the crosswalk against the live participant list."""
    path = Path(__file__).parent / "_ncaaf_participants.json"
    if not path.exists():
        print("Missing _ncaaf_participants.json — fetch the participants list first")
        return
    names = json.loads(path.read_text())

    mapped = [n for n in names if to_key(n)]
    unmapped = [n for n in names if not to_key(n)]
    keys_hit = {to_key(n) for n in mapped}
    missing = sorted(set(FBS) - keys_hit)

    print(f"crosswalk: {len(FBS)} FBS programs, {len(_ODDS_TO_KEY)} name aliases")
    print(f"  participants in feed : {len(names)}")
    print(f"  mapped to FBS        : {len(mapped)}")
    print(f"  treated as non-FBS   : {len(unmapped)}")

    if missing:
        print(f"\n  FBS keys with NO matching participant ({len(missing)}):")
        for k in missing:
            print(f"    {k:<22}{FBS[k]}")
        print("  ^ these would never join. Fix the alias or drop the key.")

    # Eyeball check: anything here that is actually FBS is a silent data loss.
    print(f"\n  first 20 treated as non-FBS (should be FCS/D2/renamed):")
    for n in unmapped[:20]:
        print(f"    {n}")

    if "--strict" in sys.argv and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
