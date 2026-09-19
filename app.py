import streamlit as st

from rag import (
    index_pdfs,
    retrieve,
    build_context,
    build_prompt,
)

from llm import generate_answer


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📚 Research Paper Assistant")

st.write(
    "Upload research papers and ask questions about them "
    "using a Retrieval-Augmented Generation (RAG) system."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    index_button = st.button(
        "🔍 Index / Re-index Papers",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Index papers
# ---------------------------------------------------------

if index_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

    else:

        with st.spinner(
            "Extracting, chunking and indexing papers..."
        ):

            try:

                total_chunks = index_pdfs(
                    uploaded_files
                )

                st.success(
                    f"Successfully indexed "
                    f"{total_chunks} chunks."
                )

            except Exception as error:

                st.error(
                    f"Indexing failed: {error}"
                )


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

st.header("💬 Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What methodology does this paper propose?"
    ),
    height=100,
)


ask_button = st.button(
    "Ask",
    type="primary",
)


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the papers..."
        ):

            try:

                retrieved_chunks = retrieve(
                    question
                )

            except Exception as error:

                st.error(
                    f"Retrieval failed: {error}"
                )

                st.stop()


        if not retrieved_chunks:

            st.warning(
                "No relevant information was found."
            )

        else:

            context = build_context(
                retrieved_chunks
            )

            prompt = build_prompt(
                question,
                context,
            )

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    result = generate_answer(
                        prompt
                    )

                except Exception as error:

                    st.error(
                        str(error)
                    )

                    st.stop()


            # -------------------------------------------------
            # Answer
            # -------------------------------------------------

            st.subheader("💡 Answer")

            st.write(
                result["answer"]
            )

            st.caption(
                f"Generated using: "
                f"{result['provider']}"
            )


            # -------------------------------------------------
            # Sources
            # -------------------------------------------------

            st.subheader(
                "📚 Retrieved Sources"
            )

            for index, item in enumerate(
                retrieved_chunks,
                start=1,
            ):

                with st.expander(
                    f"{index}. "
                    f"{item['filename']} "
                    f"— Page {item['page']}"
                ):

                    st.write(
                        item["text"]
                    )

                    st.caption(
                        f"Similarity distance: "
                        f"{item['distance']:.4f}"
                    )