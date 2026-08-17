"""Streamlit page for comparing published predictions with race results."""

from datetime import datetime

import streamlit as st

from webapp.ui.api_client import ApiDataError, ApiUnavailable, PitWallApiClient
from webapp.ui.formatting import history_rows

GLOBAL_STATISTICS = (
    ("winner_accuracy", "Winner accuracy"),
    ("podium_hit_rate", "Podium hit rate"),
    ("pairwise_accuracy", "Pairwise accuracy"),
)
MAE_LABEL = "Mean absolute position error"


def history_options(index_document: dict) -> dict[str, tuple[int, int, str]]:
    """Return race labels in reverse round order with an explicit publication type."""
    races = sorted(index_document["races"], key=lambda race: race["round"], reverse=True)
    return {
        (f"Round {race['round']} · {race['name']} · {race['session_type'].upper()}"): (
            race["season"],
            race["round"],
            race["session_type"],
        )
        for race in races
    }


def summary_messages(history_document: dict) -> list[str]:
    """Return the prediction-versus-result summary in non-technical language."""
    summary = history_document["summary"]
    mean_error = summary["mean_absolute_position_error"]
    mean_error_label = "unavailable" if mean_error is None else f"{mean_error:.1f}"
    return [
        f"Predicted podium: {summary['podium_hits']} correct drivers out of {summary['podium_total']}",
        f"Predicted top 5: {summary['top_five_hits']} correct drivers out of {summary['top_five_total']}",
        f"Average error: {mean_error_label}{'' if mean_error is None else ' positions'}",
    ]


def global_stat_metrics(index_document: dict) -> list[tuple[str, str]]:
    """Return global History statistics formatted for metric cards."""
    statistics = index_document["global_statistics"]
    return [(label, f"{statistics[key] * 100:.1f}%") for key, label in GLOBAL_STATISTICS]


def global_trend_rows(index_document: dict) -> list[dict]:
    """Return cumulative History statistics in chart-ready percentages."""
    return [
        {
            "Event": f"R{point['round']} {point['session_type'].title()}",
            "Event order": event_order,
            **{label: round(point[key] * 100, 1) for key, label in GLOBAL_STATISTICS},
            MAE_LABEL: (
                None
                if point["mean_absolute_position_error"] is None
                else round(point["mean_absolute_position_error"], 1)
            ),
        }
        for event_order, point in enumerate(index_document["global_statistics"]["timeline"])
    ]


def accuracy_chart_spec(rows: list[dict]) -> dict:
    """Return the percentage chart with an explicit chronological event order."""
    event_labels = [row["Event"] for row in sorted(rows, key=lambda row: row["Event order"])]
    statistic_labels = [label for _, label in GLOBAL_STATISTICS]
    return {
        "data": {"values": rows},
        "transform": [{"fold": statistic_labels, "as": ["Metric", "Percentage"]}],
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "Event", "type": "ordinal", "sort": event_labels, "axis": {"title": None}},
            "y": {
                "field": "Percentage",
                "type": "quantitative",
                "scale": {"domain": [0, 100]},
                "axis": {"title": None},
            },
            "color": {"field": "Metric", "type": "nominal", "sort": statistic_labels, "legend": {"title": None}},
            "tooltip": [
                {"field": "Event", "type": "nominal"},
                {"field": "Metric", "type": "nominal"},
                {"field": "Percentage", "type": "quantitative", "format": ".1f"},
            ],
        },
        "height": 350,
    }


def mae_chart_spec(rows: list[dict]) -> dict:
    """Return the cumulative MAE chart on its own position scale."""
    event_labels = [row["Event"] for row in sorted(rows, key=lambda row: row["Event order"])]
    return {
        "data": {"values": rows},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "Event", "type": "ordinal", "sort": event_labels, "axis": {"title": None}},
            "y": {"field": MAE_LABEL, "type": "quantitative", "axis": {"title": MAE_LABEL}},
            "tooltip": [
                {"field": "Event", "type": "nominal"},
                {"field": MAE_LABEL, "type": "quantitative", "format": ".1f"},
            ],
        },
        "height": 220,
    }


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
        row["Predicted"] = str(row["Predicted"])
        row["Actual"] = str(actual_position) if actual_position is not None else comparison["status"]
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
        _render_api_unavailable("The results service is currently unavailable.", "retry_history_index")
        return
    except ApiDataError:
        st.error("The published history is not available in a valid format.")
        return

    if index_document.get("global_statistics") is not None:
        for column, (label, value) in zip(st.columns(3), global_stat_metrics(index_document), strict=True):
            column.metric(label, value)
        trend_rows = global_trend_rows(index_document)
        with st.expander("Performance over time", expanded=False):
            st.vega_lite_chart(spec=accuracy_chart_spec(trend_rows), width="stretch")
            st.vega_lite_chart(spec=mae_chart_spec(trend_rows), width="stretch")

    options = history_options(index_document)
    if not options:
        st.info("There are no completed races to compare yet.")
        return

    st.header("Race predictions")
    selected_label = st.selectbox("Select a race", list(options), key="history_race")
    season, round_number, session_type = options[selected_label]
    try:
        history_document = client.get_history(season, round_number, session_type)
    except ApiUnavailable:
        _render_api_unavailable("The results service is currently unavailable.", "retry_history_detail")
        return
    except ApiDataError:
        st.error("The comparison for this race is not available in a valid format.")
        return

    race = history_document["race"]
    st.subheader(race["name"])
    st.caption(f"Round {race['round']} · {race['circuit']}")

    for column, message in zip(st.columns(3), summary_messages(history_document), strict=True):
        column.info(message)

    st.subheader("Prediction and actual result")
    st.dataframe(_display_history_rows(history_document), hide_index=True, use_container_width=True, height="content")


def main() -> None:
    """Run the page when Streamlit executes it through the navigation entry point."""
    base_url = st.session_state.get("pitwall_api_url", "http://127.0.0.1:8000")
    render_history(PitWallApiClient(base_url))


if __name__ == "__main__":
    main()
