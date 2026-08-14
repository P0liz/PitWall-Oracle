"""Streamlit entry point for the public PitWall Oracle results app."""

import os
import sys
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LOGO_PATH = Path(__file__).parent / "assets" / "pitwall-signal.png"


def api_base_url() -> str:
    """Read the API URL from Streamlit secrets, then the environment."""
    try:
        configured_url = st.secrets.get("PITWALL_API_URL")
    except StreamlitSecretNotFoundError:
        configured_url = None
    return configured_url or os.getenv("PITWALL_API_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Configure shared UI chrome and launch the selected page."""
    st.set_page_config(page_title="PitWall Oracle", page_icon="🏁", layout="wide")
    st.logo(LOGO_PATH, size="large")
    st.session_state["pitwall_api_url"] = api_base_url()
    with st.sidebar:
        st.header("PitWall Oracle")
        st.subheader("How it works")
        st.write(
            "A few hours before the race, the model publishes the predicted finishing order and race probabilities."
        )
        st.write("The history compares predictions with actual results.")
        st.subheader("My contacts")
        st.markdown(
            "- [polizzotto.gabriele7@gmail.com]()  \n"
            "- [GitHub](https://github.com/P0liz)  \n"
            "- [LinkedIn](https://www.linkedin.com/in/gabriele-polizzotto/)"
        )

    navigation = st.navigation(
        [
            st.Page("current_prediction.py", title="Next race", icon="🏎️", default=True),
            st.Page("history.py", title="History", icon="📊"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
