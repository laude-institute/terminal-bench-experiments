"""Streamlit dashboard for importing jobs into the database."""

from pathlib import Path

import streamlit as st

# Set page config
st.set_page_config(
    page_title="Import Jobs",
    page_icon=":inbox_tray:",
    layout="wide",
)

# Get the submissions directory path
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def get_submission_folders() -> list[Path]:
    """Get all folders in the submissions directory."""
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(
        [f for f in SUBMISSIONS_DIR.iterdir() if f.is_dir()],
        key=lambda x: x.name,
    )


def is_job_path(folder: Path) -> bool:
    """
    Determine if a folder is a job path (has result.json at top level).

    A job_path is a single job directory that contains result.json.
    A jobs_dir is a directory containing multiple job subdirectories.
    """
    return (folder / "result.json").exists()


def get_subfolders(folder: Path) -> list[Path]:
    """Get all subfolders of a folder."""
    if not folder.exists():
        return []
    return sorted(
        [f for f in folder.iterdir() if f.is_dir()],
        key=lambda x: x.name,
    )


def run_import(
    path: Path,
    is_job: bool,
    override_model_name: list[str] | None,
    override_model_provider: list[str] | None,
    agent_display_name: str | None,
    agent_url: str | None,
    agent_org_display_name: str | None,
    model_display_name: str | None,
    model_org_display_name: str | None,
    include_on_leaderboard: bool,
):
    """Run the import_jobs function with the given parameters."""
    # Import the function from the CLI module
    from terminal_bench_experiments.cli.main import import_jobs

    try:
        if is_job:
            import_jobs(
                jobs_dir=None,
                job_path=[path],
                job_id=None,
                override_model_name=override_model_name
                if override_model_name
                else None,
                override_model_provider=override_model_provider
                if override_model_provider
                else None,
                agent_display_name=agent_display_name if agent_display_name else None,
                agent_url=agent_url if agent_url else None,
                agent_org_display_name=agent_org_display_name
                if agent_org_display_name
                else None,
                model_display_name=model_display_name if model_display_name else None,
                model_org_display_name=model_org_display_name
                if model_org_display_name
                else None,
                include_on_leaderboard=include_on_leaderboard,
            )
        else:
            import_jobs(
                jobs_dir=path,
                job_path=None,
                job_id=None,
                override_model_name=override_model_name
                if override_model_name
                else None,
                override_model_provider=override_model_provider
                if override_model_provider
                else None,
                agent_display_name=agent_display_name if agent_display_name else None,
                agent_url=agent_url if agent_url else None,
                agent_org_display_name=agent_org_display_name
                if agent_org_display_name
                else None,
                model_display_name=model_display_name if model_display_name else None,
                model_org_display_name=model_org_display_name
                if model_org_display_name
                else None,
                include_on_leaderboard=include_on_leaderboard,
            )
        return True, "Import completed successfully!"
    except Exception as e:
        return False, f"Import failed: {e}"


def main():
    st.title("Import Jobs")
    st.markdown("Import job data from the submissions directory into the database.")

    # Check if submissions directory exists
    if not SUBMISSIONS_DIR.exists():
        st.error(f"Submissions directory not found: {SUBMISSIONS_DIR}")
        return

    # Get top-level submission folders
    submission_folders = get_submission_folders()
    if not submission_folders:
        st.warning("No submission folders found.")
        return

    st.subheader("Select Folder")

    # First level: select submission folder
    folder_names = [f.name for f in submission_folders]
    selected_folder_name = st.selectbox(
        "Submission Folder",
        options=folder_names,
        help="Select a top-level submission folder",
    )

    if not selected_folder_name:
        return

    selected_folder = SUBMISSIONS_DIR / selected_folder_name

    # Check if this folder is a job path or contains subfolders
    subfolders = get_subfolders(selected_folder)

    # Determine the final path to import
    final_path = selected_folder
    is_job = is_job_path(selected_folder)

    # If there are subfolders, let user select one
    if subfolders:
        subfolder_names = ["(Use this folder)"] + [f.name for f in subfolders]
        selected_subfolder_name = st.selectbox(
            "Subfolder (optional)",
            options=subfolder_names,
            help="Optionally select a subfolder within the submission",
        )

        if selected_subfolder_name and selected_subfolder_name != "(Use this folder)":
            final_path = selected_folder / selected_subfolder_name
            is_job = is_job_path(final_path)

    # Show the selected path and its type
    st.markdown("---")
    st.subheader("Selected Path")
    st.code(str(final_path))

    if is_job:
        st.info(
            "This is a **job path** (contains result.json) - will import as a single job"
        )
    else:
        st.info("This is a **jobs directory** - will import all jobs in subdirectories")

    # Optional overrides
    st.markdown("---")
    st.subheader("Optional Overrides")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Overrides**")
        model_names_input = st.text_input(
            "Override Model Name(s)",
            help="Comma-separated list of model names to override",
        )
        model_providers_input = st.text_input(
            "Override Model Provider(s)",
            help="Comma-separated list of model providers (must match model names count)",
        )
        model_display_name = st.text_input(
            "Model Display Name",
            help="Display name for the model",
        )
        model_org_display_name = st.text_input(
            "Model Org Display Name",
            help="Organization display name for the model",
        )

    with col2:
        st.markdown("**Agent Overrides**")
        agent_display_name = st.text_input(
            "Agent Display Name",
            help="Display name for the agent",
        )
        agent_url = st.text_input(
            "Agent URL",
            help="URL for the agent",
        )
        agent_org_display_name = st.text_input(
            "Agent Org Display Name",
            help="Organization display name for the agent",
        )

    st.markdown("---")
    include_on_leaderboard = st.checkbox(
        "Include on Leaderboard",
        value=False,
        help="Include this job on the leaderboard",
    )

    # Parse model overrides
    override_model_name = None
    override_model_provider = None
    if model_names_input and model_providers_input:
        override_model_name = [
            n.strip() for n in model_names_input.split(",") if n.strip()
        ]
        override_model_provider = [
            p.strip() for p in model_providers_input.split(",") if p.strip()
        ]

        if len(override_model_name) != len(override_model_provider):
            st.error(
                f"Model names count ({len(override_model_name)}) must match "
                f"providers count ({len(override_model_provider)})"
            )
            return

    # Import button
    st.markdown("---")
    if st.button("Import Jobs", type="primary"):
        with st.spinner("Importing jobs..."):
            success, message = run_import(
                path=final_path,
                is_job=is_job,
                override_model_name=override_model_name,
                override_model_provider=override_model_provider,
                agent_display_name=agent_display_name or None,
                agent_url=agent_url or None,
                agent_org_display_name=agent_org_display_name or None,
                model_display_name=model_display_name or None,
                model_org_display_name=model_org_display_name or None,
                include_on_leaderboard=include_on_leaderboard,
            )

        if success:
            st.success(message)
        else:
            st.error(message)


if __name__ == "__main__":
    main()
