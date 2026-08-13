"""Streamlit page for comparing published predictions with race results."""

from datetime import datetime

import streamlit as st

from webapp.ui.api_client import ApiDataError, ApiUnavailable, PitWallApiClient
from webapp.ui.formatting import history_rows


def history_options(index_document: dict) -> dict[str, tuple[int, int, str]]:
    """Return race labels in reverse round order with an explicit publication type."""
    publication_labels = {
        "backtest": "HISTORICAL BACKTEST",
        "live": "LIVE PREDICTION",
    }
    races = sorted(index_document["races"], key=lambda race: race["round"], reverse=True)
    return {
        (
            f"Round {race['round']} · {race['name']} · {race['session_type'].upper()} · "
            f"{publication_labels[race['publication_type']]}"
        ): (race["season"], race["round"], race["session_type"])
        for race in races
    }


def summary_messages(history_document: dict) -> list[str]:
    """Return the prediction-versus-result summary in non-technical language."""
    summary = history_document["summary"]
    mean_error = summary["mean_absolute_position_error"]
    mean_error_label = "unavailable" if mean_error is None else f"{mean_error:.1f}"
    return [
        f"Average error: {mean_error_label}{'' if mean_error is None else ' positions'}",
        f"Predicted podium: {summary['podium_hits']} correct drivers out of {summary['podium_total']}",
        f"Predicted top 5: {summary['top_five_hits']} correct drivers out of {summary['top_five_total']}",
    ]


def _format_datetime(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.strftime("%d/%m/%Y, %H:%M UTC")


def _render_api_unavailable(message: str, key: str) -> None:
    st.error(message)
    if st.button("Retry", key=key):
        st.rerun()


def _display_history_rows(history_document: dict) -> list[dict]:
    """Replace missing numeric finishes with their published race status."""
    rows = history_rows(history_document)
    for row, comparison in zip(rows, history_document["comparisons"], strict=True):
        actual_position = comparison["actual_position"]
        row["Actual"] = (
            str(actual_position)
            if actual_position is not None
            else comparison["status"]
        )
    return rows


def _render_model_details(history_document: dict) -> None:
    publication = history_document["publication"]
    with st.expander("Model details"):
        st.write(f"**Model artifact:** {publication['model_artifact']}")
        st.write(f"**DNF strategy:** {publication['dnf_strategy']}")
        st.write(f"**Simulations:** {publication['simulations']:,}")
        st.write(f"**Seed:** {publication['seed']}")
        st.write(f"**Generated at:** {_format_datetime(publication['generated_at'])}")
        st.write(f"**Data available through:** {_format_datetime(publication['data_cutoff'])}")


def render_history(client: PitWallApiClient) -> None:
    """Render the selected completed race with predicted-versus-real outcomes."""
    st.title("Prediction history")
    try:
        index_document = client.list_history(season=2026)
    except ApiUnavailable:
        _render_api_unavailable(
            "The results service is currently unavailable.",
            "retry_history_index",
        )
        return
    except ApiDataError:
        st.error("The published history is not available in a valid format.")
        return

    options = history_options(index_document)
    if not options:
        st.info("There are no completed races to compare yet.")
        return

    selected_label = st.selectbox("Select a race", list(options), key="history_race")
    season, round_number, session_type = options[selected_label]
    try:
        history_document = client.get_history(season, round_number, session_type)
    except ApiUnavailable:
        _render_api_unavailable(
            "The results service is currently unavailable.",
            "retry_history_detail",
        )
        return
    except ApiDataError:
        st.error("The comparison for this race is not available in a valid format.")
        return

    race = history_document["race"]
    publication_type = history_document["publication"]["type"]
    badge = "LIVE PREDICTION" if publication_type == "live" else "HISTORICAL BACKTEST"
    st.subheader(race["name"])
    st.badge(f"{race['session_type'].upper()} · {badge}")
    st.caption(f"Round {race['round']} · {race['circuit']}")

    for column, message in zip(st.columns(3), summary_messages(history_document), strict=True):
        column.info(message)

    st.subheader("Prediction and actual result")
    st.dataframe(_display_history_rows(history_document), hide_index=True, use_container_width=True)
    _render_model_details(history_document)


def main() -> None:
    """Run the page when Streamlit executes it through the navigation entry point."""
    base_url = st.session_state.get("pitwall_api_url", "http://127.0.0.1:8000")
    render_history(PitWallApiClient(base_url))


if __name__ == "__main__":
    main()
