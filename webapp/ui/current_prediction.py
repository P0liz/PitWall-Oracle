"""Current-race Streamlit page for published PitWall Oracle predictions."""

from datetime import datetime

import streamlit as st

from webapp.ui.api_client import ApiDataError, ApiUnavailable, PitWallApiClient, PredictionUnavailable
from webapp.ui.formatting import percentage, prediction_rows


def head_to_head_options(document: dict) -> dict[str, str]:
    """Map driver labels to the stable IDs required by the API."""
    return {driver["display_name"]: driver["driver_id"] for driver in document["drivers"]}


def selected_head_to_head_pair(driver_a_id: str, driver_b_id: str) -> tuple[str, str] | None:
    """Return a valid comparison pair, or no pair for a self-comparison."""
    if driver_a_id == driver_b_id:
        return None
    return driver_a_id, driver_b_id


def fetch_head_to_head(client: PitWallApiClient, driver_a_id: str, driver_b_id: str) -> dict | None:
    """Fetch a comparison only when the selected drivers are distinct."""
    pair = selected_head_to_head_pair(driver_a_id, driver_b_id)
    if pair is None:
        return None
    return client.get_head_to_head(*pair)


def _format_datetime(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.strftime("%d/%m/%Y, %H:%M UTC")


def _render_api_unavailable(message: str, key: str) -> None:
    st.error(message)
    if st.button("Retry", key=key):
        st.rerun()


def _render_prediction_header(document: dict) -> None:
    race = document["race"]
    publication = document["publication"]
    publication_label = "Live" if publication["type"] == "live" else "Backtest"
    session_label = "Sprint" if race["session_type"] == "sprint" else "Grand Prix"

    st.title(f"Next {session_label}")
    st.subheader(race["name"])
    st.badge(f"{session_label} · {publication_label}")
    details, generated = st.columns(2)
    details.write(f"**Circuit:** {race['circuit']}")
    details.write(f"**Start:** {_format_datetime(race['start_time'])}")
    generated.write(f"**Prediction generated:** {_format_datetime(publication['generated_at'])}")

    if publication["type"] == "live":
        st.info("Prediction generated after qualifying")


def _render_head_to_head(client: PitWallApiClient, document: dict) -> None:
    st.subheader("Head-to-Head")
    st.caption("Select two drivers to compare their probability of finishing ahead of each other.")

    labels_to_ids = head_to_head_options(document)
    driver_ids = list(labels_to_ids.values())
    ids_to_labels = {driver_id: label for label, driver_id in labels_to_ids.items()}
    first_driver, second_driver = st.columns(2)
    driver_a_id = first_driver.selectbox(
        "First driver", driver_ids, format_func=ids_to_labels.__getitem__, key="head_to_head_driver_a"
    )
    driver_b_id = second_driver.selectbox(
        "Second driver",
        driver_ids,
        index=1 if len(driver_ids) > 1 else 0,
        format_func=ids_to_labels.__getitem__,
        key="head_to_head_driver_b",
    )
    pair = selected_head_to_head_pair(driver_a_id, driver_b_id)
    if pair is None:
        st.info("Select two different drivers to compare them.")

    if not st.button("Compare drivers", disabled=pair is None, key="compare_drivers"):
        return

    try:
        comparison = fetch_head_to_head(client, driver_a_id, driver_b_id)
    except PredictionUnavailable:
        st.info("The prediction will be available after qualifying.")
        return
    except ApiUnavailable:
        _render_api_unavailable("The results service is currently unavailable.", "retry_head_to_head")
        return
    except ApiDataError:
        st.error("The published results are not available in a valid format.")
        return

    if comparison is None:
        return

    first_probability, second_probability = st.columns(2)
    first_probability.metric(comparison["driver_a_name"], percentage(comparison["driver_a_probability"]))
    first_probability.progress(comparison["driver_a_probability"])
    second_probability.metric(comparison["driver_b_name"], percentage(comparison["driver_b_probability"]))
    second_probability.progress(comparison["driver_b_probability"])


def render_current_prediction(client: PitWallApiClient) -> None:
    """Render the published current-race prediction."""
    try:
        document = client.get_current_prediction()
    except PredictionUnavailable:
        st.title("Next race")
        st.info("The prediction will be available after qualifying.")
        return
    except ApiUnavailable:
        st.title("Next race")
        _render_api_unavailable("The results service is currently unavailable.", "retry_current_prediction")
        return
    except ApiDataError:
        st.title("Next race")
        st.error("The published results are not available in a valid format.")
        return

    _render_prediction_header(document)
    st.subheader("Predicted finishing order")
    st.dataframe(prediction_rows(document), hide_index=True, use_container_width=True)
    _render_head_to_head(client, document)


def main() -> None:
    """Run the page when Streamlit executes it through the navigation entry point."""
    base_url = st.session_state.get("pitwall_api_url", "http://127.0.0.1:8000")
    render_current_prediction(PitWallApiClient(base_url))


if __name__ == "__main__":
    main()
