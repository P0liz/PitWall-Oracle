"""Pure display formatting helpers for Streamlit tables."""


def percentage(value: float) -> str:
    """Format a probability as a one-decimal percentage."""
    return f"{value * 100:.1f}%"


def position_delta_label(delta: int | None) -> str:
    """Describe the actual-versus-predicted position change in English."""
    if delta is None:
        return "Not classified"
    if delta < 0:
        return f"↑ {abs(delta)} better"
    if delta > 0:
        return f"↓ {delta} worse"
    return "= as predicted"


def prediction_rows(document: dict) -> list[dict]:
    """Return sorted, friendly prediction rows without internal model scores."""
    drivers = sorted(document["drivers"], key=lambda driver: driver["predicted_position"])
    return [
        {
            "Position": driver["predicted_position"],
            "Driver": driver["display_name"],
            "Team": driver["team_name"],
            "Win": percentage(driver["win_probability"]),
            "Podium": percentage(driver["podium_probability"]),
            "Points": percentage(driver["points_probability"]),
            "DNF": percentage(driver["dnf_probability"]),
        }
        for driver in drivers
    ]


def history_rows(document: dict) -> list[dict]:
    """Return history comparisons with readable actual-versus-predicted deltas."""
    return [
        {
            "Driver": comparison["display_name"],
            "Predicted": comparison["predicted_position"],
            "Actual": comparison["actual_position"],
            "Difference": position_delta_label(comparison["position_difference"]),
        }
        for comparison in document["comparisons"]
    ]
