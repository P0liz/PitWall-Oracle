"""Pure display formatting helpers for Streamlit tables."""


def percentage(value: float) -> str:
    """Format a probability as a one-decimal percentage."""
    return f"{value * 100:.1f}%"


def position_delta_label(delta: int | None) -> str:
    """Describe the actual-versus-predicted position change in Italian."""
    if delta is None:
        return "Non classificato"
    if delta < 0:
        return f"↑ {abs(delta)} meglio"
    if delta > 0:
        return f"↓ {delta} peggio"
    return "= come previsto"


def prediction_rows(document: dict) -> list[dict]:
    """Return sorted, friendly prediction rows without internal model scores."""
    drivers = sorted(document["drivers"], key=lambda driver: driver["predicted_position"])
    return [
        {
            "Posizione": driver["predicted_position"],
            "Pilota": driver["display_name"],
            "Team": driver["team_name"],
            "Vittoria": percentage(driver["win_probability"]),
            "Podio": percentage(driver["podium_probability"]),
            "Punti": percentage(driver["points_probability"]),
            "DNF": percentage(driver["dnf_probability"]),
            "Posizione media": f"{driver['expected_position']:.1f}",
        }
        for driver in drivers
    ]


def history_rows(document: dict) -> list[dict]:
    """Return history comparisons with readable actual-versus-predicted deltas."""
    return [
        {
            "Prevista": comparison["predicted_position"],
            "Pilota": comparison["display_name"],
            "Reale": comparison["actual_position"],
            "Differenza": position_delta_label(comparison["position_difference"]),
        }
        for comparison in document["comparisons"]
    ]
