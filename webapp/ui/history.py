"""Streamlit page for comparing published predictions with race results."""

from datetime import datetime

import streamlit as st

from webapp.ui.api_client import ApiDataError, ApiUnavailable, PitWallApiClient
from webapp.ui.formatting import history_rows


def history_options(index_document: dict) -> dict[str, tuple[int, int]]:
    """Return race labels in reverse round order with an explicit publication type."""
    publication_labels = {
        "backtest": "BACKTEST STORICO",
        "live": "PREVISIONE LIVE",
    }
    races = sorted(index_document["races"], key=lambda race: race["round"], reverse=True)
    return {
        (
            f"Round {race['round']} · {race['name']} · "
            f"{publication_labels[race['publication_type']]}"
        ): (race["season"], race["round"])
        for race in races
    }


def summary_messages(history_document: dict) -> list[str]:
    """Return the prediction-versus-result summary in non-technical language."""
    summary = history_document["summary"]
    mean_error = summary["mean_absolute_position_error"]
    mean_error_label = "non disponibile" if mean_error is None else f"{mean_error:.1f}".replace(".", ",")
    return [
        f"Errore medio: {mean_error_label}{'' if mean_error is None else ' posizioni'}",
        f"Podio previsto: {summary['podium_hits']} piloti corretti su {summary['podium_total']}",
        f"Top 5 prevista: {summary['top_five_hits']} piloti corretti su {summary['top_five_total']}",
    ]


def _format_datetime(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.strftime("%d/%m/%Y, %H:%M UTC")


def _render_api_unavailable(message: str, key: str) -> None:
    st.error(message)
    if st.button("Riprova", key=key):
        st.rerun()


def _display_history_rows(history_document: dict) -> list[dict]:
    """Replace missing numeric finishes with their published race status."""
    rows = history_rows(history_document)
    for row, comparison in zip(rows, history_document["comparisons"], strict=True):
        if row["Reale"] is None:
            row["Reale"] = comparison["status"]
    return rows


def _render_model_details(history_document: dict) -> None:
    publication = history_document["publication"]
    with st.expander("Dettagli del modello"):
        st.write(f"**Artefatto modello:** {publication['model_artifact']}")
        st.write(f"**Strategia DNF:** {publication['dnf_strategy']}")
        st.write(f"**Simulazioni:** {publication['simulations']:,}")
        st.write(f"**Seed:** {publication['seed']}")
        st.write(f"**Generata il:** {_format_datetime(publication['generated_at'])}")
        st.write(f"**Dati disponibili fino al:** {_format_datetime(publication['data_cutoff'])}")


def render_history(client: PitWallApiClient) -> None:
    """Render the selected completed race with predicted-versus-real outcomes."""
    st.title("Storico previsioni")
    try:
        index_document = client.list_history(season=2026)
    except ApiUnavailable:
        _render_api_unavailable(
            "Il servizio dei risultati non è raggiungibile in questo momento.",
            "retry_history_index",
        )
        return
    except ApiDataError:
        st.error("Lo storico pubblicato non è disponibile in un formato valido.")
        return

    options = history_options(index_document)
    if not options:
        st.info("Non ci sono ancora gare concluse da confrontare.")
        return

    selected_label = st.selectbox("Scegli una gara", list(options), key="history_race")
    season, round_number = options[selected_label]
    try:
        history_document = client.get_history(season, round_number)
    except ApiUnavailable:
        _render_api_unavailable(
            "Il servizio dei risultati non è raggiungibile in questo momento.",
            "retry_history_detail",
        )
        return
    except ApiDataError:
        st.error("Il confronto per questa gara non è disponibile in un formato valido.")
        return

    race = history_document["race"]
    publication_type = history_document["publication"]["type"]
    badge = "PREVISIONE LIVE" if publication_type == "live" else "BACKTEST STORICO"
    st.subheader(race["name"])
    st.badge(badge)
    st.caption(f"Round {race['round']} · {race['circuit']}")

    for column, message in zip(st.columns(3), summary_messages(history_document), strict=True):
        column.info(message)

    st.subheader("Previsione e risultato reale")
    st.dataframe(_display_history_rows(history_document), hide_index=True, use_container_width=True)
    _render_model_details(history_document)


def main() -> None:
    """Run the page when Streamlit executes it through the navigation entry point."""
    base_url = st.session_state.get("pitwall_api_url", "http://127.0.0.1:8000")
    render_history(PitWallApiClient(base_url))


if __name__ == "__main__":
    main()
