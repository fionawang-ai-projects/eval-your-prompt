"""Streamlit UI for Eval-Your-Prompt."""

import pandas as pd
import streamlit as st

from llm import generate_eval

CATEGORY_STYLE = {
    "ambiguous_input": ("blue", "Ambiguous input"),
    "edge_case": ("green", "Edge case"),
    "adversarial_injection": ("red", "Adversarial / injection"),
    "format_breaking": ("orange", "Format-breaking"),
}

st.set_page_config(page_title="Eval-Your-Prompt", page_icon="🧪", layout="wide")

st.title("🧪 Eval-Your-Prompt")
st.subheader(
    "Paste an LLM prompt and get back an adversarial test suite, a scoring rubric, "
    "and a failure-mode diagnosis — before your users find the cracks."
)

prompt = st.text_area(
    "Prompt to evaluate",
    height=220,
    placeholder="Paste the LLM prompt you want to stress-test...",
)
use_case = st.text_input(
    "Use-case context (optional)",
    placeholder="e.g., support chatbot for a banking app; replies are user-facing",
)
target_model = st.selectbox(
    "Target model (display only in v1)",
    ["Claude", "GPT-4", "Other"],
    help="Recorded for context; does not change the evaluation in v1.",
)

if st.button("Generate Eval", type="primary"):
    if not prompt.strip():
        st.warning("Please paste a prompt to evaluate.")
    else:
        with st.spinner("Designing adversarial tests, rubric, and diagnosis..."):
            try:
                st.session_state["result"] = generate_eval(prompt, use_case=use_case)
                st.session_state["target_model"] = target_model
            except Exception as exc:  # surface API/parse failures to the user
                st.session_state.pop("result", None)
                st.error(f"Could not generate the eval: {exc}")


def render(result: dict) -> None:
    tests = result["test_prompts"]

    st.header("Test Prompts")
    st.caption(
        f"Target model: {st.session_state.get('target_model', 'Claude')}  ·  "
        f"{len(tests)} adversarial inputs"
    )
    for i, test in enumerate(tests, 1):
        color, label = CATEGORY_STYLE.get(test["category"], ("gray", test["category"]))
        st.markdown(f"**{i}.** :{color}-badge[{label}]")
        st.code(test["prompt"], language=None)
        st.caption(f"Tests: {test['what_it_tests']}")

    st.header("Scoring Rubric")
    rubric_df = pd.DataFrame(
        [
            {"Criterion": r["criterion"], "Pass": r["pass"], "Fail": r["fail"]}
            for r in result["rubric"]
        ]
    )
    rubric_df.index = range(1, len(rubric_df) + 1)
    st.table(rubric_df)

    st.header("Diagnosis")
    st.info(result["diagnosis"])


if "result" in st.session_state:
    render(st.session_state["result"])
