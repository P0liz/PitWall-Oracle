"""Streamlit entry point for the public PitWall Oracle results app."""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def api_base_url() -> str:
    """Read the API URL from Streamlit secrets, then the environment."""
    try:
        configured_url = st.secrets.get("PITWALL_API_URL")
    except StreamlitSecretNotFoundError:
        configured_url = None
    return configured_url or os.getenv("PITWALL_API_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Configure shared UI chrome and launch the selected page."""
    st.set_page_config(
        page_title="PitWall Oracle",
        page_icon="🏁",
        layout="wide",
    )
    st.session_state["pitwall_api_url"] = api_base_url()

    with st.sidebar:
        st.header("Come funziona")
        st.write(
            "Dopo le qualifiche, il modello pubblica l'ordine previsto e le "
            "probabilità di gara. Lo storico confronta le previsioni con i risultati reali."
        )

    navigation = st.navigation([
        st.Page("current_prediction.py", title="Prossima gara", icon="🏎️", default=True),
        st.Page("history.py", title="Storico", icon="📊"),
    ])
    navigation.run()


if __name__ == "__main__":
    main()
