"""Gradual English coding-agent dialogue using Pi's standard tools."""

OPENING = [
    "hi",
    "I want a small offline-first conference schedule web app in this empty workspace. Before coding, use your built-in powershell tool to search the live GitHub repository API for iCalendar examples (https://api.github.com/search/repositories?per_page=5&q=icalendar, with a User-Agent and -TimeoutSec 20). Give the tool call a 30-second timeout and print at least one returned repository URL. Fetch a current W3C accessibility reference too. Cite URLs you actually retrieved and state a compact implementation plan.",
    "Implement the first vertical slice: sessions, rooms, speakers, and a day view. Use local files and run the tests you create. Keep the app disposable and self-contained.",
    "Add import of iCalendar events. Use the powershell tool to fetch current RFC 5545 guidance live, cite what you actually found, and test folded lines, escaping, and malformed input.",
    "A user in a different time zone must see the correct local date. Fetch a current source live with the powershell tool, then implement and test a DST boundary case.",
    "Add keyboard-only navigation and visible focus. Fetch a live accessibility source with the powershell tool and make the test assertions concrete.",
    "Now add an offline update queue. Show what happens when the same session is edited on two devices, and add a conflict test.",
    "I changed my mind: sessions can belong to tracks and a speaker can appear in multiple tracks. Migrate the data model without breaking existing imports.",
    "Investigate a performance issue when the schedule contains thousands of sessions. Profile the existing code before changing it; preserve behavior.",
    "Add export back to iCalendar. Verify a round trip with Unicode text and a recurring event, stating any unsupported recurrence cases.",
    "Review the earlier decisions and identify one regression risk caused by the track change. Add a test that fails without the fix.",
]
RESEARCH_TURN = 1

AREAS = [
    "time-zone conversion", "recurring sessions", "search filters", "speaker profiles",
    "room capacity", "schedule conflicts", "offline persistence", "accessibility",
    "calendar import", "calendar export", "data migration", "error recovery",
]
PERSONAS = [
    "a keyboard-only attendee", "a speaker travelling across time zones",
    "an organizer working offline", "a volunteer correcting an import",
    "an attendee using a screen reader", "an organizer merging duplicate sessions",
    "a returning user with older local data", "a coordinator handling a cancelled room",
]


def prompt_for_turn(index: int) -> str:
    if index < len(OPENING):
        return OPENING[index]
    n = index - len(OPENING)
    area = AREAS[n % len(AREAS)]
    persona = PERSONAS[(n // len(AREAS)) % len(PERSONAS)]
    cycle = n // (len(AREAS) * len(PERSONAS)) + 1
    research = (
        "Use the built-in powershell tool to fetch one current source relevant to this change, and cite its URL. "
        if n % 5 == 0 else ""
    )
    return (
        f"Follow-up {index}: {persona} reports an edge case in {area} "
        f"(iteration {cycle}). {research}Inspect the current implementation, "
        "make the smallest coherent improvement, run a relevant test, and explain "
        "how it interacts with earlier requirements. Do not erase previous work."
    )
